#!/usr/bin/env bash
set -e
cd /mnt/e/habitica-setup
SRC="/mnt/c/Users/user/Desktop/ds harness/voice_agent"

echo "════════ ① 同步 ════════"
for f in intent_config.json actions.py voice_agent.py vad.py model_runner.py \
         build_dataset.py train_intent.py package_model.py test_agent.py \
         testcases.json README.md build_exe.ps1 setup_buildenv.ps1; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" voice_agent/ && echo "  ✅ $f"
done

echo
echo "════════ ② CI 校验 ════════"
python3 scripts/ci/verify-voiceagent.py 2>&1 | tail -3

echo
echo "════════ ③ 回归：4 个 target ════════"
for t in march7 embed intent voiceagent; do
  printf '  %-12s ' "$t"
  CI_TARGET=$t bash scripts/ci/target.sh --print 2>&1 | grep -E '^DEPLOY|^VERIFY_PY' | tr '\n' ' '
  echo
done

echo
echo "════════ ④ 确认没碰 k8s ════════"
git status -s k8s/ | head -3 || true
echo "  k8s/ 改动行数: $(git status -s k8s/ | wc -l)"

echo
echo "════════ ⑤ 提交 ════════"
git add -A voice_agent scripts/ci/verify-voiceagent.py scripts/ci/lib.sh Jenkinsfile
git -c core.quotepath=false commit -q -F - <<'MSG'
fix(voiceagent): 修 wav/label 错位 + 参数 snap —— 端到端 81.28% → 98.03%

## 三个真 bug（都是实测抓出来的）

### ① wav 和 label 错位（最严重，CTC 直接崩）
数据集 ID 原来是 `{split}_{intent}_{序号}`，序号是【全局 shuffle 之后】的位置。
一扩充数据集 → shuffle 变了 → 序号对应的文本全变了，
但 tts_batch.ps1 会【跳过已存在的 wav】→ 复用了旧文件
→ audio 是 A 句、label 是 B 句。
现象：意图精度看着还行（类别编码在 ID 里），但 CTC 转写 0/609，
      模型输出「s放器」「我开」这种垃圾。
修：ID 改成【文本内容的 md5 前 10 位】，跨数据集变更稳定。

### ② CTC loss 权重
λ=1 时模型会牺牲 CTC 去换意图精度。加难负样本后 CTC 直接崩
（loss 停在 4.46，正常是 0.59）。改 λ=2 把 CTC 钉住。

### ③ 闭集参数匹配失败（端到端从 98.69% 掉到 81.28%）
CTC 会【吃掉英文单词的首字母】：
    Ollama → llama    Steam → team    Python → ython
    PowerShell → owerhell   Edge浏览器 → dge浏览器   Trae → ra
闭集意图的动作完全由意图决定，但参数匹配不上就不知道该启动哪个。
修：新增 actions.resolve() —— 按【模型判定的意图】去对应闭集里做
    编辑距离 snap（全串优先 + 前后缀兜底），并把礼貌前缀/时间副词/
    句尾语气词全剥掉。
    端到端 81.28% → 88.67% → 94.42% → 98.03%

## 最终实测（609 条 val）
  ① 规则分句       100.00%
  ② 模型意图识别    98.69%   （8 类；7 类 98.7~100%，unknown 78.12%）
  ③ 端到端执行      98.03%
  CTC 转写完全正确  77.18%

## exe 打包（Windows 侧）
  dist\voice_agent\  1.3 GB（exe 58 MB + 模型 768 MB + 依赖）
  实测单条 110~125 ms（CPU）
  踩过的坑（都写进注释了）：
    · .ps1 必须存成 UTF-8 with BOM（PS 5.1 对无 BOM 的按 ANSI 读 → 中文乱码 → 语法错）
    · $ErrorActionPreference 不能用 Stop（PyInstaller 的 INFO 走 stderr → 被当致命错误）
    · 不能 exclude torch.distributed（torch.utils.data.dataloader 依赖它）
    · PyInstaller 的 hook-webrtcvad.py 对 webrtcvad-wheels 不兼容
      （copy_metadata('webrtcvad') 找不到，发行版名是 webrtcvad_wheels）→ 已本地修补
    · 冻结后 __file__ 指向 _internal\ → 配置/模型要按 sys.executable 找
    · 冻结后 stdout 是 cp1252 → 打印中文崩 → 强制 UTF-8

## 不破坏现有部署
  k8s/ 一个字节没动。Jenkinsfile 只在新 stage ⓪ 和 4 个 when 守卫上加了
  `params.TARGET != 'voiceagent'`。回归实测 4 个 target 的 profile 都正常。
MSG
git --no-pager log --oneline -1
echo
echo "════════ ⑥ 推送 ════════"
git push origin HEAD 2>&1 | tail -3
