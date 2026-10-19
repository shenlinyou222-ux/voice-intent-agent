# -*- coding: utf-8 -*-
r"""DSH 会话：新建 / 继续（语音触发）。

与用户说定的模式（实测 dsh 0.2.0-rc.2）：

    dsh headless --json "<任务>"                   → 新建会话，第一行事件给出 sessionId
    dsh headless --session-id <id> --json "<任务>"  → 在【同一个】会话里继续

行为约定（用户原话）：
    "只需要正确按照我们说定的模式进行新建会话就行，然后除非明确说新建会话
     不然就一直在这个会话"
所以：
    · 默认（dsh_run）   → 有当前会话就 --session-id 续；没有就顺手建一个
    · 明确说新建（dsh_new）→ 强制不带 --session-id，建完把新 id 记下来

三个实测踩过的坑（都写在对应位置，别改回去）：
 ① 402 QUOTA：headless 的 cordis.patch.yml 是空的 []，会落到 bundle 默认的
    llm-deepseek(api.deepseek.com)，那个 key 余额 0 → 任务立刻 402。
    必须带 --patch dsh_model_patch.yml（从 desktop profile 抽出的模型层，
    指向 provider=opencode-go / model=deepseek-v4.1-flash）。
 ② cmd.exe 剥引号：路径含空格（…\DeepSeek Harness\…）时多引号命令行会被
    劈开，只有 `cmd /c call "<路径>" args…` 稳。
 ③ 同一会话同时只能有一个写者：两条任务挨着发，第二条报
    'session "…" is already owned by an active write handle'
    → 用【单个后台 worker + 队列】串行执行；会话 id 在【执行时】才解析，
      这样第一条刚新建的会话能被第二条正确续上。
"""
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

DEFAULT_CLI = (r"C:\Users\user\AppData\Local\Programs\DeepSeek Harness"
               r"\resources\runtime\cli\bin\dsh.cmd")
DEFAULT_WORKSPACE = r"C:\Users\user\Desktop\ds harness"

_STATE_LOCK = threading.Lock()
_Q = queue.Queue()
_WORKER = None
_WLOCK = threading.Lock()


def _log(msg, tag=None, sub=None):
    try:
        from dlog import log as _l
        _l(msg, tag, sub)
    except Exception:
        print(f"{('[' + tag + '] ') if tag else ''}{msg}", flush=True)


def _paths(cfg):
    b = getattr(cfg, "behavior", {}) or {}
    state = Path(b.get("dsh_state_file", "dsh_session.json"))
    if not state.is_absolute():
        state = Path(__file__).parent / state
    return state, state.parent / "dsh_logs"


def load_current(cfg):
    state, _ = _paths(cfg)
    try:
        return json.loads(state.read_text(encoding="utf-8")).get("session_id")
    except Exception:
        return None


def save_current(cfg, sid, task=""):
    state, _ = _paths(cfg)
    with _STATE_LOCK:
        try:
            old = json.loads(state.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        old.update({"session_id": sid,
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_task": (task or "")[:200]})
        state.write_text(json.dumps(old, ensure_ascii=False, indent=1),
                         encoding="utf-8")


def _build(cfg, task, force_new, cur=None):
    """拼命令。cur=None 且非强制新建时，读当前会话。"""
    b = getattr(cfg, "behavior", {}) or {}
    cli = str(b.get("dsh_cli", DEFAULT_CLI))
    patch = str(b.get("dsh_patch")
                or (Path(__file__).parent / "dsh_model_patch.yml"))
    if cur is None and not force_new:
        cur = load_current(cfg)
    args = ["headless"]
    if Path(patch).exists():                 # ⚠️ 坑①：不带就 402
        args += ["--patch", patch]
    args += ["--json"]
    if cur:
        args += ["--session-id", cur]
    args += [task]
    # ⚠️ 坑②：必须 cmd /c call "<路径>" —— 多引号时 cmd 会剥外层引号劈开路径
    return ["cmd.exe", "/c", "call", cli] + args


def _ensure_worker():
    global _WORKER
    with _WLOCK:
        if _WORKER is None or not _WORKER.is_alive():
            _WORKER = threading.Thread(target=_worker_loop, daemon=True,
                                       name="dsh-worker")
            _WORKER.start()


def _worker_loop():
    """⚠️ 坑③：一个 worker 串行跑，避免同一会话两个写者"""
    while True:
        cfg, task, force_new, logpath, fh = _Q.get()
        try:
            _execute(cfg, task, force_new, logpath, fh)
        except Exception as e:
            _log(f"   ❌ DSH 任务异常: {type(e).__name__}: {e}", "WARN")
        finally:
            _Q.task_done()


def _execute(cfg, task, force_new, logpath, fh):
    b = getattr(cfg, "behavior", {}) or {}
    ws = str(b.get("dsh_workspace", DEFAULT_WORKSPACE))
    prev_sid = load_current(cfg)             # 跑之前的会话，用来判断这次是新建还是续
    cmd = _build(cfg, task, force_new)       # 执行时才解析会话 id（见坑③）
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, cwd=ws, stdout=fh,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, close_fds=True)
        proc.wait()
    except Exception as e:
        fh.close()
        _log(f"   ❌ 起 dsh 失败: {type(e).__name__}: {e}", "WARN")
        return
    fh.close()

    sid, final, err = None, "", None
    try:
        for line in Path(logpath).read_text(encoding="utf-8",
                                            errors="ignore").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            t = ev.get("type")
            if t == "session" and ev.get("sessionId"):
                sid = ev["sessionId"]
            elif t == "final":
                final = (ev.get("text") or "").strip()
            elif t == "error":
                err = str(ev.get("message"))
            elif t == "turn_end":
                r = ev.get("reason") or {}
                if r.get("kind") == "error":
                    e2 = r.get("error") or {}
                    err = (f"{e2.get('message')}（code={e2.get('code')} "
                           f"status={e2.get('status')}）")
    except Exception:
        pass

    dt = time.time() - t0
    if sid:
        save_current(cfg, sid, task)
        _kind = "新建" if sid != prev_sid else "续"
        _log(f"   🆕 DSH 会话 {sid}（{_kind}）"
             + ("  以后默认都用它" if sid != prev_sid else ""), "SYS", sub=True)
    if err:
        _log(f"   ❌ DSH 任务失败：{err}  日志 {Path(logpath).name}", "WARN")
        if "QUOTA" in str(err) or "402" in str(err):
            _log("      ⚠️ 余额不足(402)：检查 --patch 是否生效"
                 "（headless 空 patch 会落到 api.deepseek.com）", "WARN")
    elif final:
        _log(f"   ✅ DSH 答完（{dt:.0f}s）：{final[:160]}"
             f"  日志 {Path(logpath).name}", "SYS", sub=True)
    else:
        _log(f"   ⚠️ DSH 结束但没有 final 文本（{dt:.0f}s），日志 "
             f"{Path(logpath).name}", "WARN")


def run_task(cfg, task, force_new=False, dry=False):
    """→ (ok, msg, info)。dry 只回命令预览；真跑则【排队】后台串行执行。"""
    b = getattr(cfg, "behavior", {}) or {}
    if not b.get("enable_dsh", True):
        return False, "dsh 功能已关闭（enable_dsh=false）", None
    task = (task or "").strip()
    if len(task) < 2:
        return False, "任务内容为空（要说清让 DSH 做什么）", None
    cli = str(b.get("dsh_cli", DEFAULT_CLI))
    if not Path(cli).exists():
        return False, f"找不到 dsh CLI: {cli}", None
    ws = str(b.get("dsh_workspace", DEFAULT_WORKSPACE))
    if not Path(ws).exists():
        return False, f"工作区不存在: {ws}", None

    cur = None if force_new else load_current(cfg)
    info = {"cmd": _build(cfg, task, force_new, cur), "workspace": ws,
            "session_id": cur, "new": not cur}
    if dry:
        where = "新建会话" if not cur else f"续会话 {cur[:20]}…"
        return True, f"[dry] DSH {where}：「{task}」  cwd={ws}", info

    _, logs = _paths(cfg)
    logs.mkdir(parents=True, exist_ok=True)
    logpath = logs / (f"dsh_{time.strftime('%m%d_%H%M%S')}"
                      f"_{abs(hash(task)) % 1000:03d}.jsonl")
    fh = open(logpath, "w", encoding="utf-8")
    _ensure_worker()
    _Q.put((cfg, task, force_new, str(logpath), fh))
    n = _Q.qsize()
    where = "新建会话" if not cur else f"续会话 {cur[:20]}…"
    tail = f"（前面还有 {n - 1} 个排队）" if n > 1 else ""
    return True, (f"已交给 DSH {where}：「{task}」{tail}  "
                  f"后台串行执行，日志 {logpath.name}"), info
