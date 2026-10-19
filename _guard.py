from pathlib import Path
v = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\voice_agent.py")
t = v.read_text(encoding="utf-8")
key = "麦克风音量过低"
if key in t:
    print("  已有，跳过")
else:
    anchor = "        floor = getattr(self, \"_noise_floor\", 0.0)"
    assert anchor in t, "噪声门锚点未匹配"
    blk = ("        # ⭐⭐ 低电平护栏（血债）：实测用户对着麦克风说话，语音峰值只有\n"
           "        #   0.0039 (-48 dBFS)、RMS 0.00047 -- 比设备自噪声 0.00058 还低，\n"
           "        #   信噪比 ~0 dB -> Silero 判\"非人声\"是【正确】的，任何算法都救不回。\n"
           "        #   根因在系统侧（录音音量/麦克风加强 ~0、或设备选错/静音）。\n"
           "        #   这里主动告警，免得再花几轮去怀疑代码。\n"
           "        try:\n"
           "            _pk = float(np.abs(wav).max())\n"
           "            if _pk < 0.01:\n"
           "                log(f\"   ⚠️ 麦克风音量过低：本段峰值 {_pk:.5f}（{20*np.log10(max(_pk,1e-9)):.0f} dBFS）。\"\n"
           "                    f\"请在 设置→系统→声音→输入 里把该设备的【级别/麦克风加强】拉高，\"\n"
           "                    f\"或换一个在动的输入设备。\")\n"
           "                self.stats[\"low_level\"] = self.stats.get(\"low_level\", 0) + 1\n"
           "        except Exception:\n"
           "            pass\n") + anchor
    t = t.replace(anchor, blk, 1)
    v.write_text(t, encoding="utf-8"); print("  ✅ 已加低电平护栏（峰值 < 0.01 就告警）")
