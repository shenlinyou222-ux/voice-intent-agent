import ctypes, time
from ctypes import wintypes
u = ctypes.WinDLL("user32", use_last_error=True)
k = ctypes.WinDLL("kernel32", use_last_error=True)
PSAPI = ctypes.WinDLL("psapi", use_last_error=True)

def integrity_of(pid):
    """看能不能拿到路径 / 能否以 PROCESS_TERMINATE 打开 → 判断是否更高权限"""
    PROCESS_QUERY_LIMITED = 0x1000
    PROCESS_TERMINATE = 0x0001
    h = k.OpenProcess(PROCESS_QUERY_LIMITED, False, pid)
    path_ok = bool(h)
    if h: k.CloseHandle(h)
    h2 = k.OpenProcess(PROCESS_TERMINATE, False, pid)
    term_ok = bool(h2)
    if h2: k.CloseHandle(h2)
    return path_ok, term_ok, ctypes.get_last_error()

import subprocess, re
out = subprocess.run(["tasklist","/FI","IMAGENAME eq clash-verge.exe","/FO","CSV","/NH"],
                     capture_output=True, text=True).stdout
m = re.search(r'"clash-verge\.exe","(\d+)"', out)
pid = int(m.group(1)) if m else None
print("clash-verge PID =", pid)
if pid:
    pq, pt, err = integrity_of(pid)
    print(f"  能查询(受限)={pq}  能以 TERMINATE 打开={pt}  last_error={err}")
    print("  → " + ("更高权限(管理员)运行：普通权限杀不掉，只能【请求它自己退出】"
                    if not pt else "权限够：应该能杀（那失败原因另有其它）"))
    # 找它的所有顶层窗口（包括隐藏的），发 WM_CLOSE
    WM_CLOSE = 0x0010
    found = []
    CB = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, lp):
        wpid = wintypes.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
        if wpid.value == pid:
            buf = ctypes.create_unicode_buffer(256)
            u.GetWindowTextW(hwnd, buf, 256)
            cls = ctypes.create_unicode_buffer(256)
            u.GetClassNameW(hwnd, cls, 256)
            found.append((hwnd, buf.value, cls.value, bool(u.IsWindowVisible(hwnd))))
        return True
    u.EnumWindows(CB(cb), 0)
    print(f"  它的顶层窗口 {len(found)} 个：")
    for hwnd, title, cls, vis in found[:8]:
        print(f"    hwnd={hex(hwnd)} vis={vis} cls={cls[:28]!r} title={title[:30]!r}")
    # 发 WM_CLOSE 给每个窗口
    for hwnd, title, cls, vis in found:
        u.PostMessageW(hwnd, WM_CLOSE, 0, 0)
    time.sleep(2.5)
    still = subprocess.run(["tasklist","/FI","IMAGENAME eq clash-verge.exe","/NH"],
                           capture_output=True, text=True).stdout
    alive = "clash-verge.exe" in still
    print("  发 WM_CLOSE 后仍在运行?" , alive, "→", "优雅关闭失败" if alive else "✅ 已优雅关闭")
