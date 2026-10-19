# -*- coding: utf-8 -*-
"""运维语音域的【执行侧】：意图 → 执行命令 → 渲染输出。

与 aoe4/openra 的 game_actions.py 同构：每个动作 (arg, cfg, dry=False) -> (ok, msg, info)
末尾导出 GAME_ACTIONS，由 actions.py 用 ACTIONS.update() 接进全局动作表。

═══ 三层策略 ═══
    read    只读 → 直接执行，输出 rich 渲染；长输出可拉窗口
    change  变更 → **两步确认**：第一步只登记 + 回问，说「确认」才真执行
    ai      开放性排查 → 交给 DSH 会话（后台），**自动带上最近一次只读输出**

═══ 为什么变更要两步确认 ═══
    语音运维最怕的不是"识别不准"，是**"识别错了还执行了"**。
    所以「重启 nginx」这句话**不执行任何东西**，只回问；「确认」才执行。

═══ 为什么 AI 升级要带上下文 ═══
    用户说「查集群」→ 拿到 kubectl 输出；接着说「让 ds 看看」。
    如果只把「让 ds 看看」这句传给 DSH，它不知道要看什么。
    所以只读动作会把输出缓存进 _LAST_OUT，AI 升级时拼进任务 —— **用户不用复述**。
"""
from __future__ import annotations

import json
import re
import subprocess
import urllib.parse
import urllib.request

from . import render, window

WSL_DISTRO = "Ubuntu-24.04"
PROM = "http://127.0.0.1:30090"          # Prometheus NodePort（⚠️ 只在 WSL 内部有效，见 _prom）
# WSL 宿主 IP 的进程内缓存（重启 WSL 会变，所以别落盘、别写死）
_WSL_IP_CACHE = ""

# ── 变更类的两步确认状态（模块级，进程内）────────────────────────────
_PENDING: dict = {"intent": None, "arg": None}

# ── 最近一次只读输出（供 AI 升级拼上下文）────────────────────────────
_LAST_OUT: dict = {"intent": None, "cmd": None, "out": ""}


# ══════════════════════════════════════════════════════════════════════
#  执行底座
# ══════════════════════════════════════════════════════════════════════
def _wsl_candidates() -> list:
    """wsl.exe 的候选路径，**按优先级**排。

    ⚠️⚠️ 血债（2026-10-16 定位到根因）：新版的 WSL 是 **MSIX 商店包**
       （MicrosoftCorporationII.WindowsSubsystemForLinux）。这时
         · C:\\Windows\\System32\\wsl.exe      只是个 274KB 的【存根】，
           它要去【激活 MSIX 包】才能跑真正的 WSL
         · C:\\Users\\<u>\\AppData\\Local\\Microsoft\\WindowsApps\\wsl.exe  是 0 字节的
           App Execution Alias（重解析点）
         · C:\\Program Files\\WSL\\wsl.exe     才是真正的二进制（4.3MB / 2.7.x）
       实测：**PyInstaller 打出来的 frozen exe 里，System32 那个存根会返回
       「拒绝访问。错误代码: Wsl/E_ACCESSDENIED」**（MSIX 激活被拒），
       而同一台机器上 python.exe / cmd.exe / powershell.exe 跑同一个存根却没事，
       直接用 C:\\Program Files\\WSL\\wsl.exe 也没事。
       → 所以优先用真二进制，存根只当兜底。
    """
    import os
    cands = []
    real = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                        "WSL", "wsl.exe")
    if os.path.exists(real):
        cands.append(real)
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    stub = os.path.join(root, "System32", "wsl.exe")
    if os.path.exists(stub):
        cands.append(stub)
    cands.append("wsl.exe")          # 最后才走 PATH
    return cands


def _wsl_exe() -> str:
    """当前选中的 wsl.exe（= 候选列表第一个，供 doctor / 日志显示）。"""
    return _wsl_candidates()[0]


def _decode_wsl(b: bytes) -> str:
    """解 wsl.exe / bash 的输出。

    ⚠️ 血债（2026-10-16 实测）：wsl.exe **自己**的错误消息是 UTF-16LE，
       而 bash 里的命令输出是 UTF-8。原来一律按 utf-8 解，于是
       「拒绝访问。错误代码: Wsl/E_ACCESSDENIED」被解成
       「�b�~���0 ... ���Nx: W s l / E _ A C C E S S D E N I E D」——
       完全没法读，白排查半天。这里按「有没有大量 NUL 字节」自动判。
    """
    if not b:
        return ""
    # UTF-16LE 的 ASCII 字符每个后面跟一个 0x00 → NUL 占比很高
    if b.count(b"\x00") > len(b) // 4:
        try:
            return b.decode("utf-16-le", errors="replace")
        except Exception:
            pass
    return b.decode("utf-8", errors="replace")


def _run_wsl_files(cmd: str, timeout: int, exe: str):
    """策略 A：把 stdout/stderr 重定向到临时文件（**不用管道**）。

    ⚠️ 血债（2026-10-16 实测）：frozen exe（PyInstaller）里用管道接 wsl.exe 的输出，
       wsl.exe 直接回一句「拒绝访问。」并返回非 0 —— 而同一台机器上
       · 源码跑（python voice_agent.py）完全正常
       · 桌面域的 tasklist（同样用 capture_output=True）也正常
       · 最小 frozen 探针跑 wsl.exe 也正常
       只有「frozen + voice_agent + wsl.exe + 管道」这个组合炸。
       wsl.exe 要建 pty，对继承的管道句柄比普通控制台程序敏感得多。
       改成写临时文件后不再走管道，绕开这个组合。
    """
    import tempfile
    import os
    fo = fe = None
    try:
        fo = tempfile.NamedTemporaryFile(delete=False, suffix=".wslout")
        fe = tempfile.NamedTemporaryFile(delete=False, suffix=".wslerr")
        fo.close(); fe.close()
        with open(fo.name, "wb") as ho, open(fe.name, "wb") as he:
            p = subprocess.run(
                [exe, "-d", WSL_DISTRO, "--", "bash", "-lc", cmd],
                stdout=ho, stderr=he, timeout=timeout,
            )
        out = _decode_wsl(open(fo.name, "rb").read()).strip()
        err = _decode_wsl(open(fe.name, "rb").read()).strip()
        if p.returncode == 0:
            return True, out or err
        return False, (out + ("\n" if out and err else "") + err) or f"退出码 {p.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"⏱ 超时（>{timeout}s）：{cmd}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        for f in (fo, fe):
            if f is not None:
                try:
                    os.unlink(f.name)
                except Exception:
                    pass


def _run_wsl_pipes(cmd: str, timeout: int, exe: str):
    """策略 B：管道接输出（原来的做法；源码运行下一直好用）。"""
    try:
        p = subprocess.run(
            [exe, "-d", WSL_DISTRO, "--", "bash", "-lc", cmd],
            capture_output=True, timeout=timeout,   # 拿 bytes，自己解码
        )
    except subprocess.TimeoutExpired:
        return False, f"⏱ 超时（>{timeout}s）：{cmd}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    out = _decode_wsl(p.stdout).strip()
    err = _decode_wsl(p.stderr).strip()
    if p.returncode == 0:
        return True, out or err
    return False, (out + ("\n" if out and err else "") + err) or f"退出码 {p.returncode}"


def _run_wsl(cmd: str, timeout: int = 15, retries: int = 1):
    """在 WSL 里跑命令，返回 (ok, out)。

    ★ 用 subprocess 的 list 形式传参：命令作为**单个 argv** 交给 bash -lc，
      Windows 侧不做任何引号处理 —— 避开 cmd/PowerShell 剥引号那类坑。

    ★ 依次试：**wsl.exe 候选路径 × 接输出的策略**。
      第一优先是 C:\\Program Files\\WSL\\wsl.exe（真二进制，绕开 MSIX 存根），
      策略优先临时文件（不走管道）。原因见 _wsl_candidates / _run_wsl_files 的血债注释。
    """
    last = ""
    for exe in _wsl_candidates():
        for fn in (_run_wsl_files, _run_wsl_pipes):
            for attempt in range(retries + 1):
                ok, out = fn(cmd, timeout, exe)
                if ok:
                    return True, out
                last = out
                # 只有【WSL 服务级】的失败才重试；命令自己失败重试没意义
                if not _retryable(out):
                    return False, _wsl_hint(out)
                if attempt < retries:
                    import time as _t
                    _t.sleep(0.5)
            # 这条路整条失败 → 换下一种策略 / 换下一个 wsl.exe
    return False, _wsl_hint(last)


def _retryable(msg: str) -> bool:
    """这个失败值不值得重试（WSL 服务级 / 偶发权限拒绝）。"""
    m = msg or ""
    return ("Wsl/" in m
            or "拒绝访问" in m
            or "ACCESSDENIED" in m
            or "Access is denied" in m)


def _wsl_hint(msg: str) -> str:
    """把 wsl.exe 的裸报错补成【能照着修】的提示。

    ⚠️ 2026-10-16 实测：frozen exe 里 wsl.exe 只回一句「拒绝访问。」，
       看不出是 WSL 没装、发行版名错、还是调用方权限不够 —— 白排查半天。
       这里按关键字给出下一步。
    """
    m = msg or ""
    tips = []
    if "拒绝访问" in m or "ACCESSDENIED" in m or "Access is denied" in m:
        tips.append("WSL 拒绝访问 —— 常见原因：① 当前用户不在 "
                    "Administrators 组 / 被组策略限制；② 进程被以管理员身份运行而"
                    "WSL 服务不允许；③ 杀软拦了 wsl.exe。"
                    "先在同一个窗口手工跑 `wsl.exe -d " + WSL_DISTRO + " -- echo ok` 对照")
    if "Wsl/" in m and "E_ACCESSDENIED" not in m:
        tips.append("这是 WSL 服务级错误，通常是服务忙/刚唤醒 —— 重试一次多数会好")
    if "找不到" in m or "not found" in m.lower():
        tips.append(f"发行版名可能不对：当前写死 WSL_DISTRO={WSL_DISTRO!r}，"
                    "用 `wsl.exe -l -v` 看实际名字")
    if "超时" in m:
        tips.append("命令超时：kubectl 冷启动或集群不健康时会慢，"
                    "可手工跑同一条命令看要多久")
    if not tips:
        return m
    return m + "\n\n💡 " + "\n💡 ".join(tips)


# ══════════════════════════════════════════════════════════════════════
#  数据源标注 —— 每个只读动作都要说清「这个数从哪来」
# ══════════════════════════════════════════════════════════════════════
# ⚠️⚠️ 为什么必须标（2026-10-16 实测踩过）：
#   语音和 Grafana 面板必须是【同一个源】，否则会出现
#   「问语音说 14%、看面板显示 9%」这种自相矛盾 —— GPU 那边真差过：
#     nvidia-smi: 利用率 14% / 显存 5456 MiB / 功耗 16.44 W
#     Prometheus: 利用率  9% / 显存 5461 MiB / 功耗 16.45 W
#   因为那是两条采集路、采样时刻不同。用户看到两个数只会怀疑哪个是真的。
#   标出来之后，一眼能看出这个数是不是面板上的那个数。
SRC_PROM = "Prometheus（与 Grafana 面板同源）"
SRC_K8S = "kubectl（面板上没有）"
SRC_LIVE = "实时命令（面板上没有，无历史趋势）"


def _src_note(src: str | None, dash: str | None = None) -> str:
    """数据源标注 + 「看趋势去哪个面板」。"""
    if not src:
        return ""
    s = f"\n\n── 数据源：{src}"
    if dash:
        s += f"　·　看趋势：说「打开{dash}面板」"
    return s


def _fmt_num(v, unit: str) -> str:
    """按单位把数字格式化成人看得懂的样子。"""
    try:
        f = float(v)
    except Exception:
        return f"{v} {unit}"
    if unit in ("B", "Bps"):
        for div, u in ((1024 ** 4, "TiB"), (1024 ** 3, "GiB"), (1024 ** 2, "MiB"), (1024, "KiB")):
            if abs(f) >= div:
                return f"{f / div:,.1f} {u}" + ("/s" if unit == "Bps" else "")
        return f"{f:,.0f} B" + ("/s" if unit == "Bps" else "")
    # ⚠️ load 必须保留小数 —— 用 "short" 会被 f"{f:,.0f}" 取整成 1（实测 load1=1.37 显示成 1）
    if unit == "load":
        return f"{f:.2f}"
    if unit in ("MiB", "MHz", "W", "s", "short"):
        return f"{f:,.0f}" + (f" {unit}" if unit not in ("short", "s") else "")
    return f"{f:,.2f} {unit}".strip()


def _prom_read(intent: str, title: str, metrics, dash: str | None = None,
               timeout: int = 12, extra: str = ""):
    """Prometheus 只读动作的统一实现（**和 Grafana 面板同源**那一类）。

    metrics: [(标签, PromQL, 单位), ...] —— 每项取第一个序列。
    """
    lines, got, miss = [], 0, []
    for label, q, unit in metrics:
        res, err = _prom(q, timeout=timeout)
        if err or not res:
            miss.append(label)
            continue
        got += 1
        lines.append(f"  {label:<14} {_fmt_num(res[0]['value'][1], unit)}")
    if got == 0:
        msg = (f"{title}：Prometheus 查不到数据 —— 监控栈不可用。\n"
               f"（面板这会儿也是黑的，两边同源就是这个意思）\n"
               f"查 exporter：`kubectl -n monitoring get ds node-exporter`")
        render.render_result(False, f"{title} 失败", msg, None)
        return False, msg, {"promql": [m[1] for m in metrics]}
    body = extra + "\n".join(lines)
    if miss:
        body += f"\n\n  （缺 {len(miss)} 项：{', '.join(miss)}）"
    body += _src_note(SRC_PROM, dash)
    render.render_result(True, title, body, None)
    _LAST_OUT.update({"intent": intent, "cmd": f"PromQL {len(metrics)} 项", "out": body})
    return True, body, {"rows": got, "miss": miss}


def _read(intent: str, arg, cfg, cmd: str, title: str, timeout: int = 15,
          hl: str | None = None, window_if_long: int = 30,
          src: str | None = SRC_LIVE, dash: str | None = None):
    """只读动作的统一实现（跑命令那类）：执行 → 渲染 → 缓存（供 AI 升级）。

    ⚠️ 注意 src 默认是 SRC_LIVE（实时命令）—— 这类数据【面板上没有】，
       所以标注里明说，免得用户去面板上找不到对不上的数。
       Prometheus 那类走 _prom_read，标注 SRC_PROM。
    """
    ok, out = _run_wsl(cmd, timeout)
    note = _src_note(src, dash)
    n = len(out.splitlines())
    # 长输出 → 拉窗口（能滚动、能 grep、能接着手动操作）
    if ok and n > window_if_long and window.available():
        window.open_in_window(cmd, title, hl=hl)
        msg = f"{title}：{n} 行，已开窗口显示{note}"
        render.render_result(True, msg, "", None)
    else:
        msg = out + note
        render.render_result(ok, f"{title}" if ok else f"{title} 失败", msg, None)
    # ⚠️ 缓存进 _LAST_OUT 的是【原始输出】（不带标注）—— 那是给 AI 升级用的上下文，
    #    混进「数据源：…」这种自家文案会污染它的输入。
    _LAST_OUT.update({"intent": intent, "cmd": cmd, "out": out})
    return ok, msg, {"cmd": cmd, "lines": n}


# ══════════════════════════════════════════════════════════════════════
#  L1 · 只读（规则直通，直接执行）
# ══════════════════════════════════════════════════════════════════════
# ⚠️⚠️ 主机资源为什么改成走 Prometheus（2026-10-16）：
#   原来跑 df -h / free -h / uptime —— 那是【另一条采集路】，和面板无关，
#   而且面板上当时根本没有主机资源（没装 node-exporter）。
#   装完之后两边同源：面板看趋势，语音问当前值，数字必然一致。
#   ⚠️ CPU 用【比值式】而不是 100-avg(rate(idle))*100 —— WSL2 的 /proc/stat
#      让 20 核计数器合计涨到 22.13 s/s（偏快 1.107×），常见写法会算出负数
#      （实测 -8.17%）。比值式分子分母同样被放大 → 抵消。详见面板注解。
# 面板里按 `{mountpoint=~"/|/mnt/.*"}` 取 / 和 /mnt/*；语音侧按具体挂载点查（见 act_ops_disk）


def act_ops_disk(arg, cfg, dry=False):
    if dry: return True, "[dry] PromQL: node_filesystem_* （与面板同源）", {}
    return _prom_read("ops_disk", "磁盘占用（各挂载点）", [
        ("/ 使用率",      '100 * (1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"})', "%"),
        ("/ 可用",        'node_filesystem_avail_bytes{mountpoint="/"}', "B"),
        ("/mnt/e 使用率", '100 * (1 - node_filesystem_avail_bytes{mountpoint="/mnt/e"} / node_filesystem_size_bytes{mountpoint="/mnt/e"})', "%"),
        ("/mnt/e 可用",   'node_filesystem_avail_bytes{mountpoint="/mnt/e"}', "B"),
        ("/mnt/c 可用",   'node_filesystem_avail_bytes{mountpoint="/mnt/c"}', "B"),
        ("/mnt/d 可用",   'node_filesystem_avail_bytes{mountpoint="/mnt/d"}', "B"),
        ("读速率",        'sum(rate(node_disk_read_bytes_total[1m]))', "Bps"),
        ("写速率",        'sum(rate(node_disk_written_bytes_total[1m]))', "Bps"),
    ], dash="主机资源")


def act_ops_mem(arg, cfg, dry=False):
    if dry: return True, "[dry] PromQL: node_memory_* （与面板同源）", {}
    return _prom_read("ops_mem", "内存", [
        ("使用率",   '(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100', "%"),
        ("总量",     'node_memory_MemTotal_bytes', "B"),
        ("可用",     'node_memory_MemAvailable_bytes', "B"),
        ("已用",     'node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes', "B"),
        ("缓存",     'node_memory_Cached_bytes + node_memory_Buffers_bytes', "B"),
        ("Swap 已用", 'node_memory_SwapTotal_bytes - node_memory_SwapFree_bytes', "B"),
    ], dash="主机资源")


def act_ops_cpu(arg, cfg, dry=False):
    if dry: return True, "[dry] PromQL: node_cpu_* / node_load* （与面板同源）", {}
    return _prom_read("ops_cpu", "系统负载", [
        ("CPU 使用率", '100 * (1 - sum(rate(node_cpu_seconds_total{mode="idle"}[1m])) / sum(rate(node_cpu_seconds_total[1m])))', "%"),
        ("核数",       'count(count by (cpu) (node_cpu_seconds_total{mode="idle"}))', "short"),
        ("load1",      'node_load1', "load"),
        ("load5",      'node_load5', "load"),
        ("load15",     'node_load15', "load"),
        ("iowait",     '100 * sum(rate(node_cpu_seconds_total{mode="iowait"}[1m])) / sum(rate(node_cpu_seconds_total[1m]))', "%"),
        ("运行中进程", 'node_procs_running', "short"),
        ("阻塞进程",   'node_procs_blocked', "short"),
    ], dash="主机资源")


def act_ops_gpu(arg, cfg, dry=False):
    """★ 走 Prometheus（nvidia_smi_exporter），和 Grafana 面板【同一个源】。

    ⚠️⚠️ 血债（2026-10-16 实测）：原来跑 `nvidia-smi --query-gpu=...`，
       而面板走的是 exporter —— **两条采集路，采样时刻不同**
       （Prometheus scrape_interval=5s）。同一时刻实测：
           nvidia-smi: 利用率 14% / 显存 5456 MiB / 功耗 16.44 W
           Prometheus: 利用率  9% / 显存 5461 MiB / 功耗 16.45 W
       → 「问语音」和「看面板」得到不同数字，**这本身就是 bug**：
         用户会怀疑到底哪个是真的。统一走 Prometheus 后两边必然一致
         （面板刷新滞后 ≤5s，但值同源）。

    ★ 故意【不】静默回退到 nvidia-smi：监控栈挂了就该说挂了
      （面板那时也是黑的），不能悄悄给一个和面板不一样的值。
    """
    if dry:
        return True, "[dry] PromQL: nvidia_smi_* （与 Grafana 面板同源）", {}

    got, miss = {}, []
    for key, q, unit in GPU_METRICS:
        res, err = _prom(q)
        if err or not res:
            miss.append(key)
            continue
        m = res[0].get("metric", {})
        got[key] = (res[0]["value"][1], unit, m.get("name") or m.get("gpu", ""))

    if not got:
        return False, ("Prometheus 里没有 GPU 指标 —— 监控栈不可用，"
                       "面板这会儿也是黑的。\n"
                       "查 exporter：`systemctl status nvidia-smi-exporter`"
                       "（WSL 内，监听 :9400）"), {"promql": "nvidia_smi_*"}

    gpu = next((v[2] for v in got.values() if v[2]), "")
    lines = [f"GPU：{gpu}   （数据源 = Prometheus nvidia_smi_exporter，与面板同源）", ""]
    for key, q, unit in GPU_METRICS:
        if key not in got:
            lines.append(f"  {key:<10} ——")
            continue
        raw, u, _ = got[key]
        try:
            f = float(raw)
            s = f"{f:,.0f}" if u in ("MiB", "MHz", "W") else f"{f:.1f}"
        except Exception:
            s = str(raw)
        extra = ""
        # 显存顺手给个百分比（面板上就是这么显示的）
        if key == "显存已用" and "显存总量" in got:
            try:
                extra = f"   ({float(raw) / float(got['显存总量'][0]) * 100:.0f}%)"
            except Exception:
                pass
        lines.append(f"  {key:<10} {s:>10} {u}{extra}")
    if miss:
        lines.append("")
        lines.append(f"  （缺 {len(miss)} 项：{', '.join(miss)} —— exporter 可能少暴露）")
    lines.append(_src_note(SRC_PROM, "GPU"))
    return True, "\n".join(lines), {"promql": "nvidia_smi_*", "rows": len(got)}


def act_ops_ports(arg, cfg, dry=False):
    cmd = "ss -tlnp"
    if dry: return True, f"[dry] {cmd}", {}
    return _read("ops_ports", arg, cfg, cmd, "端口占用")


def act_ops_proc(arg, cfg, dry=False):
    cmd = "ps aux --sort=-%mem | head -20"
    if dry: return True, f"[dry] {cmd}", {}
    return _read("ops_proc", arg, cfg, cmd, "进程（按内存排序）")


def act_ops_service(arg, cfg, dry=False):
    if not arg:
        return False, "没说要查哪个服务", {}
    if dry: return True, f"[dry] systemctl status {arg}", {}
    return _read("ops_service", arg, cfg, f"systemctl status {arg} --no-pager", f"服务 {arg}")


def act_ops_logs(arg, cfg, dry=False):
    if not arg:
        # 没指定就列最近有日志的 unit
        return _read("ops_logs", arg, cfg,
                     "journalctl -n 60 --no-pager", "最近日志")
    if dry: return True, f"[dry] journalctl -u {arg} -n 60", {}
    return _read("ops_logs", arg, cfg,
                 f"journalctl -u {arg} -n 60 --no-pager", f"日志 {arg}")


def act_ops_k8s_pods(arg, cfg, dry=False):
    cmd = f"kubectl get pods {'-n '+arg if arg else '-A'} -o wide"
    if dry: return True, f"[dry] {cmd}", {}
    return _read("ops_k8s_pods", arg, cfg, cmd, "集群 Pod", src=SRC_K8S)


def act_ops_k8s_nodes(arg, cfg, dry=False):
    cmd = "kubectl get nodes -o wide"
    if dry: return True, f"[dry] {cmd}", {}
    return _read("ops_k8s_nodes", arg, cfg, cmd, "节点", src=SRC_K8S)


def act_ops_k8s_svc(arg, cfg, dry=False):
    cmd = "kubectl get svc,ingress -A"
    if dry: return True, f"[dry] {cmd}", {}
    return _read("ops_k8s_svc", arg, cfg, cmd, "Service / Ingress", src=SRC_K8S)


def act_ops_k8s_describe(arg, cfg, dry=False):
    """★ pod 挂了的首选 —— describe 的 Events 段直接给原因。"""
    if not arg:
        return False, "没说要查哪个 pod", {}
    cmd = f"kubectl describe pod {arg} -A | head -80"
    if dry: return True, f"[dry] {cmd}", {}
    # describe 输出长，直接拉窗口
    if window.available():
        window.show_pod_why(arg)
        ok, out = _run_wsl(cmd, 20)
        _LAST_OUT.update({"intent": "ops_k8s_describe", "cmd": cmd, "out": out})
        return ok, f"已开窗口显示 {arg} 的详情（含 Events）{_src_note(SRC_K8S)}", {}
    return _read("ops_k8s_describe", arg, cfg, cmd, f"pod {arg} 详情", 20, src=SRC_K8S)


def act_ops_k8s_logs(arg, cfg, dry=False):
    if not arg:
        return False, "没说要查哪个 pod 的日志", {}
    cmd = f"kubectl logs {arg} --tail=100 -p 2>&1 | tail -100"
    if dry: return True, f"[dry] {cmd}", {}
    return _read("ops_k8s_logs", arg, cfg, cmd, f"pod {arg} 日志", 20, hl="log", src=SRC_K8S)


# ── Prometheus 指标（⭐ TTFT / TPOT）──────────────────────────────────
QUERIES = {
    "ttft":      'histogram_quantile(0.50, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le)) * 1000',
    "ttft_p90":  'histogram_quantile(0.90, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le)) * 1000',
    "ttft_p99":  'histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[1m])) by (le)) * 1000',
    # ⚠️ vLLM 0.30 把 time_per_output_token_seconds 改名成 request_time_per_output_token_seconds
    "tpot":      'histogram_quantile(0.50, sum(rate(vllm:request_time_per_output_token_seconds_bucket[1m])) by (le))',
    "tpot_p99":  'histogram_quantile(0.99, sum(rate(vllm:request_time_per_output_token_seconds_bucket[1m])) by (le))',
    "kvcache":   'vllm:kv_cache_usage_perc * 100',
    "queue":     'vllm:num_requests_waiting',
    "running":   'vllm:num_requests_running',
    "throughput":'rate(vllm:generation_tokens_total[1m])',
    "qps":       'sum(rate(vllm:request_success_total[1m]))',
    "prefixcache": ('100 * sum(rate(vllm:prefix_cache_hits_total[1m])) '
                    '/ clamp_min(sum(rate(vllm:prefix_cache_queries_total[1m])), 0.001)'),
}

# 不给 :service 时默认查哪个 vLLM 服务
# ⚠️ 2026-10-17：Prometheus 加了 vllm-asr 的 target 之后，三个服务
#    （march7 336ms / embed 26ms / asr 15ms）差一个数量级，
#    不过滤就会混在一起算分位数 —— 算出来的数是假的。
_DEFAULT_SVC = "asr"

_LABEL = {"ttft": "TTFT p50 (ms)", "ttft_p90": "TTFT p90 (ms)", "ttft_p99": "TTFT p99 (ms)",
          "tpot": "TPOT p50 (s)", "tpot_p99": "TPOT p99 (s)",
          "kvcache": "KV cache 使用率 (%)", "queue": "排队请求数", "running": "在算请求数",
          "throughput": "生成吞吐 (tok/s)", "qps": "QPS", "prefixcache": "前缀缓存命中率 (%)"}

# GPU：**和 Grafana 面板同一个 exporter（nvidia_smi_exporter @ :9400）**
# ⚠️ 指标名是实测从 Prometheus 里查出来的，不是猜的：
#    是 `nvidia_smi_utilization_gpu_percent` 而【不是】..._ratio；
#    是 `nvidia_smi_memory_used_mib` 而【不是】..._bytes；
#    是 `nvidia_smi_temperature_celsius` 而【不是】..._gpu。
#    面板 60-grafana-dashboards.yaml 里用的就是这几个名字。
GPU_METRICS = [
    ("利用率",    "nvidia_smi_utilization_gpu_percent",    "%"),
    ("显存带宽",  "nvidia_smi_utilization_memory_percent", "%"),
    ("显存已用",  "nvidia_smi_memory_used_mib",            "MiB"),
    ("显存总量",  "nvidia_smi_memory_total_mib",           "MiB"),
    ("温度",      "nvidia_smi_temperature_celsius",        "°C"),
    ("功耗",      "nvidia_smi_power_draw_watts",           "W"),
    ("功耗上限",  "nvidia_smi_power_limit_watts",          "W"),
    ("SM 时钟",   "nvidia_smi_clock_sm_mhz",               "MHz"),
]


def _prom(query: str, timeout: int = 12):
    """查 Prometheus。

    ★★ 血债：**必须从 WSL 里 curl，不能在 Windows 上 urllib** ——
        Prometheus 是 NodePort，NodePort 走 iptables DNAT **不产生 listen socket**，
        而 WSL2 的 localhostForwarding 只转发**有 listen socket** 的端口
        → Windows 上访问 127.0.0.1:30090 必然 `WinError 10061 目标计算机积极拒绝`。
        （实测踩过：ops_metric 三个用例全 URLError，其余动作都正常。）
        从 WSL 内部访问 NodePort 是通的（curl 实测 HTTP 200）。
    """
    import shlex
    q = shlex.quote(query)
    cmd = f"curl -sG '{PROM}/api/v1/query' --data-urlencode 'query={query}'"
    ok, out = _run_wsl(cmd, timeout)
    if not ok:
        return None, out
    try:
        j = json.loads(out)
    except Exception:
        return None, f"返回不是 JSON：{out[:200]}"
    res = j.get("data", {}).get("result", [])
    if not res:
        # ★ 空结果不是错误 —— 很可能是指标改名了
        return None, "空结果（⚠️ 指标名可能已改名，去 Prometheus 自动补全确认）"
    return res, None


# 中文说法 → QUERIES 的 key（用户可能说「查首token延迟」而不是「查TTFT」）
_METRIC_ALIAS = {
    "首token": "ttft", "首字": "ttft", "首token延迟": "ttft", "首字延迟": "ttft",
    "每token": "tpot", "每字": "tpot", "每token延迟": "tpot",
    "kv缓存": "kvcache", "kv": "kvcache", "kv cache": "kvcache",
    "前缀缓存": "prefixcache", "前缀命中": "prefixcache",
    "排队": "queue", "等待": "queue", "积压": "queue",
    "并发": "running", "在跑": "running", "运行中": "running",
    "吞吐": "throughput", "生成速度": "throughput",
}


def _norm_metric(name: str) -> str:
    """把语音给过来的指标名归一成 QUERIES 的 key。

    ⚠️⚠️ 血债（2026-10-17 用户实测）：
       说「查TTFT」被 ASR 转成「TTFT。」，然后报
         不认识的指标「TTFT。」（可选：ttft, ttft_p90, ...）
       原因：这里原来是 `QUERIES.get(name)` —— **完全没做归一化**，
       而 ASR 输出几乎【总会补一个句号】（实测：转写结果里
       「打开知谱。」「精确查找测试报告。」都带句号）。

       ⚠️ 短语表那条路（「查TTFT。」）能对上是因为短语匹配做了前缀处理；
         但 kNN / 候选表那条路会把【原始文本】直接当 arg 传进来 → 带句号。
         所以必须在【动作层】兜住，不能只靠短语表。

    归一化做四件事：
      ① 去空白 + 中英文标点（句号/逗号/问号/引号/括号/省略号…）
      ② 统一小写（ASR 可能给 TTFT / Ttft / ttft）
      ③ 去掉动词前缀（查/看/问/一下/的）
      ④ 中文说法映射（首token延迟 → ttft）
    """
    s = str(name or "").strip()
    s = re.sub(r"[\s。，、！？；：\.\,\!\?\;\:'\"“”‘’（）\(\)【】\[\]《》<>·…~]+", "", s)
    s = s.lower()
    for p in ("查", "看", "问", "一下", "的"):
        if s.startswith(p) and len(s) > len(p):
            s = s[len(p):]
    if s in QUERIES:
        return s
    for k, v in _METRIC_ALIAS.items():
        if k in s:
            return v
    # 前缀匹配（「ttft_p90」这种带后缀的）
    for k in sorted(QUERIES, key=len, reverse=True):
        if s.startswith(k):
            return k
    return s


def act_ops_metric(arg, cfg, dry=False):
    """arg 形如 'ttft' 或 'ttft:march7'。"""
    name, _, svc = (arg or "").partition(":")
    raw = name
    name = _norm_metric(name)
    q = QUERIES.get(name)
    if not q:
        return False, (f"不认识的指标「{raw}」（归一化后「{name}」）\n"
                       f"可选：{', '.join(QUERIES)}\n"
                       f"也可以说中文：首token延迟 / 每token延迟 / KV缓存 / 排队 / 并发 / 吞吐"), {}
    if svc:
        svc = re.sub(r"[\s。，、！？\.\,\!\?]+$", "", str(svc)).strip()
    else:
        # ⚠️⚠️ 2026-10-17：不给 :service 时【必须】默认限定到 asr。
        #    原来不加过滤 —— 那时 Prometheus 只抓了 march7(336ms) 和 embed(26ms)，
        #    两个混在一起算分位数已经不准（embed.json 头部的注释就写了这件事）。
        #    现在又加了 asr(15ms) 进来，三者差一个数量级，混算出来的数是【假的】。
        #    ⭐ 默认选 asr 的理由：它是语音自己的转写服务，问「TTFT」最关心它。
        #      要看别的就说「查march7的TTFT」/「查embed的延迟」。
        svc = _DEFAULT_SVC
    if svc:
        # ★ 三个 vLLM 服务延迟差一个数量级（336ms / 26ms / 15ms），必须分组
        q = q.replace("_bucket[", f'_bucket{{service="{svc}"}}[')
        q = q.replace("vllm:kv_cache_usage_perc", f'vllm:kv_cache_usage_perc{{service="{svc}"}}')
        q = q.replace("vllm:num_requests_waiting", f'vllm:num_requests_waiting{{service="{svc}"}}')
        q = q.replace("vllm:num_requests_running", f'vllm:num_requests_running{{service="{svc}"}}')
    if dry:
        return True, f"[dry] PromQL: {q}", {}
    res, err = _prom(q)
    if err:
        return False, err, {"promql": q}
    rows = []
    for x in res[:8]:
        lbl = x.get("metric", {})
        tag = lbl.get("service") or lbl.get("instance") or lbl.get("model_name") or ""
        try:
            v = float(x["value"][1])
            # ⚠️ histogram_quantile 在【最近 1 分钟没有请求】时返回 nan ——
            #    直接显示「nan」用户看不懂（实测用户会以为是坏了）。
            #    翻译成人话，并提示这是「没流量」不是「出错」。
            if v != v:      # NaN 判断（不能用 == ）
                v = "最近 1 分钟没请求"
            else:
                v = f"{v:.4g}"
        except Exception:
            v = x["value"][1]
        rows.append((tag or "值", v))
    if not rows:
        rows = [("值", "最近 1 分钟没请求")]
    render.render_kv(f"{_LABEL.get(name, name)}{(' @'+svc) if svc else ''}", rows,
                     subtitle=f"PromQL: {q[:120]}")
    out = " | ".join(f"{k}={v}" for k, v in rows)
    out += _src_note(SRC_PROM, "vLLM推理" if (svc or "") != "embed" else "向量")
    _LAST_OUT.update({"intent": "ops_metric", "cmd": q, "out": out})
    return True, out, {"promql": q}


# ── 打开（面板 / 终端）──────────────────────────────────────────────
def _wsl_host_ip(cfg=None) -> str:
    """WSL 的 IP —— Windows 上的浏览器**必须**用它，不能用 127.0.0.1。

    ⚠️⚠️ 血债（2026-10-17 用户实测「127.0.0.1:30030 拒绝连接」）：
       Grafana / Prometheus 是 **NodePort** —— NodePort 走 iptables DNAT，
       **不产生 listen socket**（实测 `ss -tln` 里 30xxx 一个都没有）。
       而 WSL2 的 localhostForwarding **只转发有 listen socket 的端口**
       → Windows 上访问 `127.0.0.1:30030` 必然 ERR_CONNECTION_REFUSED。
       从 WSL 内部访问是通的（curl 实测 HTTP 200），所以要用 WSL 的 IP。

    ⚠️ 同一个坑我在 `_prom()` 里已经踩过一次（那次是 urllib 报 10061），
       当时的结论是「PromQL 必须从 WSL 里 curl」—— 但**漏了「打开浏览器」这条路**：
       `webbrowser.open()` 是在 **Windows** 上执行的，照样连不上。
       所以打开面板前必须先把 host 换掉。

    ⚠️ WSL 的 IP **重启会变**（实测现在是 172.20.71.101，也可能 172.x.x.x 别的），
       所以**必须运行时查**，不能写死在 profile.json 里。
    """
    global _WSL_IP_CACHE
    if _WSL_IP_CACHE:
        return _WSL_IP_CACHE
    # ① 先看 profile 有没有显式配（给「我知道我的 IP 是啥」的情况留口子）
    try:
        for u in ((cfg.raw.get("urls") or []) if cfg is not None and hasattr(cfg, "raw") else []):
            if u.get("id") in ("grafana", "prometheus") and u.get("host"):
                _WSL_IP_CACHE = str(u["host"]).strip()
                return _WSL_IP_CACHE
    except Exception:
        pass
    # ② 运行时查（hostname -I 的第一个就是 eth0 的）
    try:
        ok, out = _run_wsl("hostname -I", 8)
        if ok:
            for tok in str(out).split():
                tok = tok.strip()
                # 只要私网 IPv4，别把 10.42.x.x（k3s pod 网段）当宿主
                if re.match(r"^(172|192\.168|10)\.\d+\.\d+\.\d+$", tok) and not tok.startswith("10.42."):
                    _WSL_IP_CACHE = tok
                    return _WSL_IP_CACHE
    except Exception:
        pass
    return ""


def _fix_host_for_windows(url: str, cfg=None) -> str:
    """把 URL 里的 127.0.0.1 / localhost 换成 WSL 的 IP。

    ⚠️ 为什么：NodePort 没有 listen socket，Windows 上连 127.0.0.1 必然拒绝
       （见 _wsl_host_ip 的血债注释）。只换 host，端口和路径不动。
    """
    ip = _wsl_host_ip(cfg)
    if not ip:
        return url
    return re.sub(r"^(https?://)(127\.0\.0\.1|localhost)(:\d+)?", lambda m: f"{m.group(1)}{ip}{m.group(3) or ''}", url)


# Grafana 面板的 uid（= k8s/live/60-grafana-dashboards.yaml 里各 json 的 "uid"）
# ⚠️ 加面板时这里要同步 —— 否则「打开XX面板」会 404。
#    实测取法：curl -s -u admin:admin http://127.0.0.1:30030/api/search?type=dash-db
DASH_UIDS = {
    "主机资源": "host-mon", "主机": "host-mon", "host": "host-mon",
    "gpu": "gpu-mon", "显卡": "gpu-mon", "gpu监控": "gpu-mon",
    "vllm": "vllm-mon", "推理": "vllm-mon", "llm": "vllm-mon", "生成": "vllm-mon",
    "向量": "vllm-embed-mon", "embed": "vllm-embed-mon", "嵌入": "vllm-embed-mon",
    "llamacpp": "llamacpp-mon", "llama": "llamacpp-mon",
}


def act_ops_open(arg, cfg, dry=False):
    what, _, sub = (arg or "").partition(":")
    urls = {u["id"]: u.get("url") for u in (cfg.raw.get("urls") or [])} if hasattr(cfg, "raw") else {}
    if dry:
        return True, f"[dry] 打开 {what}{(':'+sub) if sub else ''}", {}
    if what in ("grafana", "panel", "面板"):
        base = urls.get("grafana", "http://127.0.0.1:30030")
        # 支持「打开主机资源面板」这种自然说法：把名字映射成 uid
        key = (sub or "").strip().lower()
        uid = DASH_UIDS.get(key) or DASH_UIDS.get((sub or "").strip())
        if uid:
            url = f"{base}/d/{uid}"
        elif sub:
            url = f"{base}/d/{sub}-mon"      # 兜底：直接把 sub 当 uid 前缀
        else:
            url = base
        # ⚠️⚠️ 必须换 host：Grafana 是 NodePort，没有 listen socket，
        #    Windows 上开 127.0.0.1:30030 会 ERR_CONNECTION_REFUSED（用户实测）。
        url = _fix_host_for_windows(url, cfg)
        import webbrowser
        webbrowser.open(url)
        hint = "" if uid or not sub else f"（没登记过的面板名「{sub}」，按 {sub}-mon 试的）"
        return True, f"已打开 Grafana：{url}{hint}", {"url": url}
    if what in ("prometheus", "prom"):
        url = _fix_host_for_windows(urls.get("prometheus", "http://127.0.0.1:30090/graph"), cfg)
        import webbrowser
        webbrowser.open(url)
        return True, f"已打开 Prometheus：{url}", {"url": url}
    if what in ("terminal", "wsl", "终端"):
        return window.open_terminal()
    if what in ("logdir", "日志目录"):
        return window.open_in_window("ls -la ~/dsh_logs 2>/dev/null || ls -la /tmp", "日志目录")
    return False, f"不认识要打开什么：{arg}", {}


# ══════════════════════════════════════════════════════════════════════
#  L3 · 变更（★ 两步确认，绝不一步执行）
# ══════════════════════════════════════════════════════════════════════
def _ask_confirm(intent: str, arg: str, what: str, cmd: str, dry: bool):
    """第一步：只登记 + 回问。"""
    if dry:
        return True, f"[dry] 会问「确认{what} {arg} 吗？」", {}
    _PENDING["intent"], _PENDING["arg"], _PENDING["cmd"] = intent, arg, cmd
    return True, f"⚠️ 确认{what} {arg} 吗？说「确认」执行，说「取消」放弃", {"pending": True}


# ══════════════════════════════════════════════════════════════════════
#  ★ 变更类为什么用 `sudo -n`（血债）
#     实测：WSL 里 `systemctl restart nginx` 直接报
#         Failed to restart nginx.service: Interactive authentication required.
#     而**交互式 sudo 会卡在密码提示上** —— 语音 agent 没有终端可以输密码，
#     进程会一直挂住，用户只看到"没反应"。
#     所以统一加 `-n`（non-interactive）：需要密码就**立刻失败**并给出可操作的提示，
#     绝不挂起。真要执行，先让用户在 WSL 里配好免密 sudo：
#         echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl" | sudo tee /etc/sudoers.d/voice-ops
# ══════════════════════════════════════════════════════════════════════
def act_ops_restart_service(arg, cfg, dry=False):
    if not arg:
        return False, "没说要重启哪个服务", {}
    return _ask_confirm("ops_restart_service", arg, "重启服务",
                        f"sudo -n systemctl restart {arg}", dry)


def act_ops_restart_pod(arg, cfg, dry=False):
    if not arg:
        return False, "没说要重启哪个 pod", {}
    return _ask_confirm("ops_restart_pod", arg, "滚动重启",
                        f"kubectl rollout restart deploy/{arg}", dry)


def act_ops_stop_service(arg, cfg, dry=False):
    if not arg:
        return False, "没说要停哪个服务", {}
    return _ask_confirm("ops_stop_service", arg, "停掉服务",
                        f"sudo -n systemctl stop {arg}", dry)


def act_ops_confirm(arg, cfg, dry=False):
    """第二步：只有存在待确认项时才真执行。"""
    if not _PENDING.get("intent"):
        return False, "没有待确认的操作", {}
    it, a, cmd = _PENDING["intent"], _PENDING["arg"], _PENDING.get("cmd") or ""
    _PENDING["intent"] = _PENDING["arg"] = _PENDING["cmd"] = None
    if dry:
        return True, f"[dry] {cmd}", {}
    ok, out = _run_wsl(cmd, 60)
    # ★ sudo -n 需要密码时的提示（别让用户以为"没反应"）
    if not ok and ("password is required" in out.lower() or "a password is required" in out.lower()
                   or "interactive authentication" in out.lower()):
        hint = ("需要 sudo 权限但当前不能免密执行。在 WSL 里配一次免密即可：\n"
                "  echo \"$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl\" | "
                "sudo tee /etc/sudoers.d/voice-ops")
        render.render_result(False, f"执行失败（sudo 需要密码）：{cmd}", hint, None)
        return False, hint, {"cmd": cmd, "need_sudo": True}
    render.render_result(ok, f"{'已执行' if ok else '执行失败'}：{cmd}", out, None)
    return ok, out or cmd, {"cmd": cmd}


def act_ops_cancel(arg, cfg, dry=False):
    if not _PENDING.get("intent"):
        return False, "没有待确认的操作", {}
    what = _PENDING["intent"]
    _PENDING["intent"] = _PENDING["arg"] = _PENDING["cmd"] = None
    return True, f"已取消（{what}）", {}


# ══════════════════════════════════════════════════════════════════════
#  ③ AI 升级（DSH 会话）
# ══════════════════════════════════════════════════════════════════════
def act_ops_ask_ai(arg, cfg, dry=False):
    """开放性排查交给 DSH 会话（后台跑，不阻塞语音）。

    ★★ 设计取舍（用户决定）：**不做指代消解**。
       最初写了一版「把『第3个error的pod』解析成具体名字再交给 AI」的引用消解层
       （ops/listref.py）。**实测后推翻**：
         · 要把「序数 + 状态过滤 + kubectl 表格解析 + 上一条输出」全串起来，
           **意图太复杂、传递的东西太多**
         · 而且解析出来的还是**可能已过期的缓存** —— 集群状态随时在变
       ✅ 改成：**直接把原话交给 AI，并明确告诉它「自己重新跑命令拿最新数据」**。
          数据更新、链路更短、意图更简单。

    ★ 三条实测踩过的坑（dsh_session.py 头部有完整记录）：
      ① 402 QUOTA：headless 的 cordis.patch.yml 是空的 []，会落到 bundle 默认的
         llm-deepseek(api.deepseek.com)，那个 key 余额 0 → 任务立刻 402。
         必须带 --patch dsh_model_patch.yml。
      ② cmd.exe 剥引号：路径含空格时多引号命令行会被劈开，
         只有 `cmd /c call "<路径>" args…` 稳。
      ③ 同一会话同时只能有一个写者 → 用【单个后台 worker + 队列】串行执行。
    """
    task = (arg or "").strip()
    if not task:
        return False, "没说要让 AI 看什么", {}
    if dry:
        return True, f"[dry] 会起 DSH 会话排查：「{task}」", {}

    b = getattr(cfg, "behavior", {}) or {}
    ws = str(b.get("dsh_workspace", r"C:\Users\user\Desktop\ds harness"))

    # ── ① 告诉 AI 它能自己跑命令（这是「不做指代消解」的前提）──
    head = (
        "我在做运维排查，请帮我分析。\n"
        f"工作区：{ws}\n"
        f"环境：Windows + WSL(Ubuntu-24.04)。命令都在 WSL 里跑，"
        f"kubectl / nvidia-smi / systemctl / journalctl / ss / df 都可用；"
        f"k3s 集群是本机的（单节点 desktop-a00thlv）。\n"
        "★ 如果需要集群或系统现状，**请自己重新执行命令拿最新数据**"
        "（例如 `kubectl get pods -A -o wide`、`kubectl describe pod <name>`、"
        "`kubectl logs <name> --tail=100 -p`），**不要依赖我下面贴的输出** —— "
        "那是我之前跑的，可能已经过期。\n"
        "★ Prometheus 在 http://127.0.0.1:30090（NodePort，**必须从 WSL 里 curl**，"
        "Windows 上访问 localhost 会连接被拒）。\n"
    )

    # ── ② 把最近一次只读输出作为【线索】附上（明确标注可能过期）──
    ctx = ""
    try:
        if b.get("dsh_context_inject", True) and _LAST_OUT.get("out"):
            mx = int(b.get("dsh_context_max_chars", 3000) or 3000)
            ctx = (f"\n【线索：我最近跑过的命令（可能已过期，请以你自己跑的结果为准）】\n"
                   f"意图：{_LAST_OUT.get('intent')}\n"
                   f"命令：{_LAST_OUT.get('cmd')}\n"
                   f"```\n{_LAST_OUT['out'][:mx]}\n```\n")
    except Exception:
        pass

    full = f"{head}\n我的诉求：{task}\n{ctx}"

    # ══════════════════════════════════════════════════════════════════
    #  ★★ 2026-10-17 用户指定：**用 opencode CLI，显式新建会话，开一个窗口**
    #
    #  原来是 dsh_session.run_task()（DSH headless，后台跑，结果去 dsh_logs/ 翻）。
    #  用户要求改成：
    #    ① 显式的新建会话 —— 每次都是【新会话】，不是接着上次
    #       （opencode 的 `--session <id>`：id 不存在就新建）
    #    ② 用 opencode CLI（不是 DSH）
    #    ③ 【启动一个窗口】—— 要能看着它跑，不是闷在后台
    #
    #  ⚠️ 为什么不用命令行直接传 prompt：prompt 是多行中文，
    #     经过 cmd.exe → powershell 两层引号会被劈开（DSH 那边踩过：
    #     「cmd.exe 剥引号，路径含空格时多引号命令行会被劈开」）。
    #     所以走【文件】：
    #       prompt  → <tmp>\voice_askai_<sid>.txt   （UTF-8，多行随便写）
    #       启动器  → <tmp>\voice_askai_<sid>.ps1   （纯 ASCII，只负责读文件+调 opencode）
    #     这样 .ps1 本身是 ASCII，不受 PowerShell 5.1 的编码限制。
    # ══════════════════════════════════════════════════════════════════
    import os
    import tempfile
    import time as _time

    oc = str(b.get("opencode_cli", r"C:\Users\user\.opencode\bin\opencode.exe"))
    if not os.path.exists(oc):
        return False, (f"找不到 opencode CLI：{oc}\n"
                       f"（在 profile 的 behavior.opencode_cli 里改路径）"), {}

    # ══════════════════════════════════════════════════════════════════
    #  ⚠️⚠️ opencode 的认证与免费档 —— 2026-10-17 全部实测，别重复踩
    #
    #  症状：`opencode run` 报
    #     Error: OpenCode's free tier can only be used from within OpenCode.
    #  两个独立原因，**两个都要解决**：
    #
    #  ① 没认证 → 建立 opencode-go 连接
    #     · opencode-go 的 integration 定义（opencode api integration.list 拿到的）：
    #         methods: [{type:"key"}, {type:"env", names:["OPENCODE_API_KEY"]}]
    #       ⚠️ 环境变量名是 OPENCODE_API_KEY，**不是** DSH 用的 OPENCODE_GO_API_KEY
    #     · `--method env` 不被接受（Available: key）
    #     · `--method key` 要求交互终端（API key input requires an interactive terminal）
    #     ✅ 正解：走 HTTP API（**从 bash 调，PowerShell 会把 JSON 引号剥掉**）
    #         opencode api POST /api/integration/opencode-go/connect/key \
    #           --data '{"key":"<OPENCODE_GO_API_KEY 的值>"}'
    #       验证：`opencode auth list` 出现 "OpenCode Go ... stored"
    #             `opencode api integration.list` 里 connections 有 cred_xxx
    #
    #  ② 默认模型是免费的 → 撞免费档
    #     · `opencode models` 里 opencode-go 有 beam-free（免费）和
    #       deepseek-v4.1-flash / glm-5.3 / gpt-6-luna 等（付费）
    #     ✅ 正解：指定付费模型。已在 ~/.config/opencode/opencode.json 里设了默认：
    #         {"model": "opencode-go/deepseek-v4.1-flash"}
    #       （实测不指定就报免费档；显式带 --model 或设默认后正常）
    #
    #  打通后的形态（这就是下面 launcher 用的）：
    #     opencode run --session ses_voice_<时间戳> --auto "<prompt>" "<workspace>"
    #     → 返回 "> build · deepseek-v4.1-flash" + 回答，会话也建了
    # ══════════════════════════════════════════════════════════════════
    sid = f"ses_voice_{_time.strftime('%Y%m%d-%H%M%S')}"     # ★ 新会话 ID
    # ⚠️ opencode 要求会话 ID 以 "ses" 开头（实测：
    #    `Error: Expected a string starting with "ses"  at ["sessionID"]`），
    #    所以不能叫 voice-xxx，必须是 ses_xxx。
    tmp = tempfile.gettempdir()
    pf = os.path.join(tmp, f"voice_askai_{sid}.txt")
    sf = os.path.join(tmp, f"voice_askai_{sid}.ps1")
    try:
        with open(pf, "w", encoding="utf-8") as fh:
            fh.write(full)
    except Exception as e:
        return False, f"写 prompt 文件失败：{type(e).__name__}: {e}", {}

    # ⚠️ 启动器必须【纯 ASCII】—— PowerShell 5.1 读 BOM-less UTF-8 会按 ANSI 解，
    #    中文会乱。prompt 放在单独的 UTF-8 文件里读，这里只写 ASCII。
    launcher = (
        "$ErrorActionPreference = 'Continue'\n"
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8\n"
        "$OutputEncoding = [Text.Encoding]::UTF8\n"
        f"$p = Get-Content -Raw -Encoding UTF8 '{pf}'\n"
        "Write-Host '=== opencode ops analysis ===' -ForegroundColor Cyan\n"
        f"Write-Host 'session : {sid}' -ForegroundColor DarkGray\n"
        f"Write-Host 'workdir : {ws}' -ForegroundColor DarkGray\n"
        "Write-Host 'note    : it runs commands itself; give it time.' -ForegroundColor DarkGray\n"
        "Write-Host ''\n"
        # ⭐ 用 run 子命令（非交互）—— 2026-10-17 实测打通：
        #    `opencode run --session ses_xxx --auto "<prompt>" <dir>`
        #    返回 "> build · deepseek-v4.1-flash" + 真正的回答，会话也建了。
        #
        #    ⚠️ 之前用 `--prompt $p`（交互式入口）是因为免费档报
        #       「free tier can only be used from within OpenCode」。
        #       根因找到并解决了：**默认模型是 -free 的那个**，
        #       指定付费模型 opencode-go/deepseek-v4.1-flash 就绕过了。
        #       （认证方式见下面的注释）
        f"& '{oc}' run --session '{sid}' --auto $p '{ws}'\n"
        "Write-Host ''\n"
        "Write-Host '[done] press any key to close' -ForegroundColor Green\n"
        "$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')\n"
    )
    try:
        with open(sf, "w", encoding="ascii", errors="replace") as fh:
            fh.write(launcher)
    except Exception as e:
        return False, f"写启动器失败：{type(e).__name__}: {e}", {}

    # 开窗口（start 起一个独立控制台；用 -File 调 ps1，绕开引号剥离）
    # ⚠️ `start` 的语法是 start ["标题"] 命令 —— 标题【必须带引号】，
    #    否则 start 会把标题当成要执行的程序（实测报
    #    「The system cannot find the file opencode-voice-...」）。
    #    subprocess 只在参数含空格时才加引号，所以这里显式给空标题 ""。
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", "powershell", "-NoProfile",
             "-ExecutionPolicy", "Bypass", "-File", sf],
            shell=False,
        )
    except Exception as e:
        return False, f"起窗口失败：{type(e).__name__}: {e}", {}

    return (True,
            (f"🤖 已开新窗口交给 opencode 排查\n"
             f"   会话：{sid}（新会话，不是接上次）\n"
             f"   工作区：{ws}\n"
             f"   ⚠️ 排查过程中它自己会跑命令拿最新数据，你不用管"),
            {"session": sid, "prompt_file": pf, "launcher": sf})


# ══════════════════════════════════════════════════════════════════════
#  注册表 —— 这一行就是"接进全局动作表"的接口
# ══════════════════════════════════════════════════════════════════════
GAME_ACTIONS = {
    # L1 只读
    "ops_disk":          act_ops_disk,
    "ops_mem":           act_ops_mem,
    "ops_cpu":           act_ops_cpu,
    "ops_gpu":           act_ops_gpu,
    "ops_ports":         act_ops_ports,
    "ops_proc":          act_ops_proc,
    "ops_service":       act_ops_service,
    "ops_logs":          act_ops_logs,
    "ops_k8s_pods":      act_ops_k8s_pods,
    "ops_k8s_nodes":     act_ops_k8s_nodes,
    "ops_k8s_svc":       act_ops_k8s_svc,
    "ops_k8s_describe":  act_ops_k8s_describe,
    "ops_k8s_logs":      act_ops_k8s_logs,
    "ops_metric":        act_ops_metric,
    "ops_open":          act_ops_open,
    # L3 变更（两步确认）
    "ops_restart_service": act_ops_restart_service,
    "ops_restart_pod":     act_ops_restart_pod,
    "ops_stop_service":    act_ops_stop_service,
    "ops_confirm":         act_ops_confirm,
    "ops_cancel":          act_ops_cancel,
    # AI 升级
    "ops_ask_ai":        act_ops_ask_ai,
}
