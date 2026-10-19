import numpy as np
try:
    from pycaw.pycaw import AudioUtilities
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import IAudioEndpointVolume
    from pycaw.constants import EDataFlow, ERole
    print("  === 所有录音设备（含音量/静音）===")
    devs = AudioUtilities.GetAllDevices()
    for d in devs:
        try:
            if d.dataflow != EDataFlow.eCapture.value:
                continue
            name = (d.FriendlyName or "")[:44]
            vol = None; mute = None
            try:
                itf = d._dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                v = cast(itf, POINTER(IAudioEndpointVolume))
                vol = v.GetMasterVolumeLevelScalar(); mute = bool(v.GetMute())
            except Exception as e:
                vol = f"读取失败({type(e).__name__})"
            flag = ""
            if isinstance(vol, float):
                if mute: flag = "  ← ⚠️ 已静音"
                elif vol < 0.35: flag = "  ← ⚠️ 音量极低"
            print("    %-46s vol=%s mute=%s%s" % (name, 
                  ("%.2f" % vol) if isinstance(vol, float) else vol, mute, flag))
        except Exception:
            continue
except Exception as e:
    print("  pycaw 不可用:", type(e).__name__, e)
