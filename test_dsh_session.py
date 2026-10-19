# -*- coding: utf-8 -*-
"""DSH 会话功能回归：分句/解析 + 新建 vs 续会话 + 状态文件。

不烧额度：用一个【假的 dsh CLI】(.cmd) 回放真实事件流格式（照着实测输出写的），
所以能完整验证"新建会话 → 记下 id → 下一条自动续同一会话"这条链路。
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import actions                                   # noqa: E402
from actions import Config, parse                # noqa: E402

HERE = Path(__file__).parent
REAL_STREAM = Path(tempfile.gettempdir()) / "fake_dsh_stream.txt"

# 真·实测的一段事件流（会话 id 每次不同，这里固定一个 UUID 便于断言）
FAKE_SID = "session-11111111-2222-3333-4444-555555555555"


def make_fake_cli(tmp: Path) -> Path:
    """假 CLI：把 --session-id 的值回显出来 + 回放事件流"""
    bat = tmp / "fake_dsh.cmd"
    stream = REAL_STREAM
    stream.write_text(
        json.dumps({"type": "session", "sessionId": FAKE_SID,
                    "cwd": str(tmp)}, ensure_ascii=False) + "\n"
        + json.dumps({"type": "status", "phase": "turn_start", "turn": 1}) + "\n"
        + json.dumps({"type": "text", "text": "收到。"}, ensure_ascii=False) + "\n"
        + json.dumps({"type": "status", "phase": "turn_end", "turn": 1,
                      "reason": {"kind": "completed"}}) + "\n"
        + json.dumps({"type": "final", "text": "收到。"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    # 记录收到的参数，便于断言"续会话带上了 --session-id"
    bat.write_text(
        "@echo off\r\n"
        f'echo %* > "{tmp}\\last_args.txt"\r\n'
        f'type "{stream}"\r\n', encoding="utf-8")
    return bat


def main():
    tmp = Path(tempfile.mkdtemp(prefix="dsh test "))
    fake = make_fake_cli(tmp)
    cfg_raw = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))
    cfg_raw["behavior"].update({
        "dsh_cli": str(fake),
        "dsh_workspace": str(tmp),
        "dsh_state_file": str(tmp / "dsh_session.json"),
        "dsh_patch": str(HERE / "dsh_model_patch.yml"),
        "dsh_timeout_s": 30,
    })
    cfgp = tmp / "cfg.json"
    cfgp.write_text(json.dumps(cfg_raw, ensure_ascii=False), encoding="utf-8")
    CFG = Config(str(cfgp))

    bad = 0

    # ① 规则解析
    print("① 规则解析（默认续会话 / 明确新建）")
    CASES = [
        ("让DSH跑一下测试", "dsh_run", "跑一下测试"),
        ("让dsh 看看这个bug", "dsh_run", "看看这个bug"),
        ("交给DSH 重构这个模块", "dsh_run", "重构这个模块"),
        ("新建DSH会话，跑一下测试", "dsh_new", "跑一下测试"),
        ("新建会话 帮我写个脚本", "dsh_new", "帮我写个脚本"),
        ("打开知乎", "open_url", "知乎"),
    ]
    for text, want_i, want_a in CASES:
        got_i, got_a = parse(text, CFG)
        ok = (got_i == want_i and got_a == want_a)
        if not ok:
            bad += 1
        print(f"  {'✅' if ok else '❌'} {text!r:26s} → {got_i:8s} {got_a!r}")
        if not ok:
            print(f"       期望 {want_i} {want_a!r}")

    # ② 第一次：没有会话 → 应新建
    # ⚠️ 不要 monkeypatch Popen：要让【假 CLI 真跑】（它把收到的参数写进
    #    last_args.txt，比抓 Popen 更接近真实链路），watcher 才能读到事件流。
    args_file = tmp / "last_args.txt"
    print("\n② 第一条任务（无当前会话）→ 应新建")
    ok, msg, info = actions.act_dsh_run("跑一下测试", CFG, dry=False)
    print(f"  ok={ok}  new={info['new']}  msg={msg}")
    for _ in range(20):
        if args_file.exists():
            break
        time.sleep(0.25)
    got_args = args_file.read_text(encoding="utf-8", errors="ignore").strip()
    print(f"  假 CLI 收到的参数: {got_args}")
    if "--session-id" in got_args:
        print("  ❌ 第一次不该带 --session-id")
        bad += 1
    if "--patch" not in got_args:
        print("  ❌ 必须带 --patch（否则 402）")
        bad += 1

    # ③ 等 watcher 抓到 sessionId
    from dsh_session import load_current
    for _ in range(40):
        if load_current(CFG):
            break
        time.sleep(0.25)
    sid = load_current(CFG)
    print(f"\n③ 状态文件里的会话 id: {sid}")
    if sid != FAKE_SID:
        print(f"  ❌ 期望 {FAKE_SID}")
        bad += 1

    # ④ 第二条：应续同一会话
    print("\n④ 第二条任务 → 应续同一会话（带 --session-id）")
    args_file.unlink(missing_ok=True)
    ok, msg, info = actions.act_dsh_run("再跑一遍", CFG, dry=False)
    print(f"  ok={ok}  msg={msg}")
    for _ in range(20):
        if args_file.exists():
            break
        time.sleep(0.25)
    got_args = args_file.read_text(encoding="utf-8", errors="ignore").strip()
    print(f"  假 CLI 收到的参数: {got_args}")
    if "--session-id" not in got_args or FAKE_SID not in got_args:
        print("  ❌ 没带上上一个会话 id")
        bad += 1

    # ⑤ 明确新建 → 必须不带 --session-id
    print("\n⑤ 明确「新建会话」→ 不该带 --session-id")
    args_file.unlink(missing_ok=True)
    ok, msg, info = actions.act_dsh_new("开个新活", CFG, dry=False)
    print(f"  ok={ok}  new={info['new']}  msg={msg}")
    for _ in range(20):
        if args_file.exists():
            break
        time.sleep(0.25)
    got_args = args_file.read_text(encoding="utf-8", errors="ignore").strip()
    print(f"  假 CLI 收到的参数: {got_args}")
    if "--session-id" in got_args:
        print("  ❌ 新建会话却带了 --session-id")
        bad += 1

    # ⑥ 空任务应被拒
    print("\n⑥ 空任务/太短任务应拒绝")
    ok, msg, _ = actions.act_dsh_run("好", CFG, dry=True)
    print(f"  ok={ok}（应为 False） msg={msg}")
    if ok:
        bad += 1

    n = len(CASES) + 5
    print(f"\n{'='*60}\n通过（除①的 {len(CASES)} 条外还有 5 项检查）"
          f"失败项 {bad}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
