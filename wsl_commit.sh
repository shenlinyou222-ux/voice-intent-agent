#!/usr/bin/env bash
set -e
cd /mnt/e/habitica-setup

echo "════════ ① 同步最新代码 ════════"
SRC="/mnt/c/Users/user/Desktop/ds harness/voice_agent"
for f in intent_config.json actions.py voice_agent.py vad.py model_runner.py \
         build_dataset.py train_intent.py package_model.py test_agent.py \
         testcases.json README.md build_exe.ps1 setup_buildenv.ps1; do
  if [ -f "$SRC/$f" ]; then cp "$SRC/$f" voice_agent/; echo "  ✅ $f"; else echo "  ⚠️  缺 $f"; fi
done

echo
echo "════════ ② 校验 ════════"
python3 scripts/ci/verify-voiceagent.py 2>&1 | tail -4

echo
echo "════════ ③ git 状态 ════════"
git add -A voice_agent scripts/ci/verify-voiceagent.py scripts/ci/lib.sh Jenkinsfile
git status -s | head -30

echo
echo "════════ ④ 确认没碰现有清单 ════════"
git status -s k8s/ | head -5 || echo "  k8s/ 无改动 ✅"
echo "  现有模型锁文件:"
git status -s k8s/live/models.lock || echo "    models.lock 无改动 ✅"

echo
echo "════════ ⑤ 提交 ════════"
git -c core.quotepath=false commit -q -F - <<'MSG'
feat(voiceagent): 语音意图 Agent —— 常开流式监听 → 意图识别 → 自动执行

新增 voice_agent/：一个独立的语音命令 Agent，说「启动记事本」真的弹记事本，
说「打开B站」用默认浏览器打开，说「搜索文件季度报告」用文件管理器搜索。

## 意图集合（8 类）
  launch_app    启动<程序>      24 个闭集 → Popen(exe)
  open_url      打开<网址>      16 个闭集 → 默认浏览器
  open_file     打开<文件>       7 个闭集 → os.startfile
  open_web      打开<任意>      不在闭集 → 默认浏览器打开/搜索
  web_search    搜索<词>        默认浏览器搜索
  file_search   搜索文件<词>    文件管理器搜索（Windows Search 协议）
  exact_search  精确查找<词>    必须含「精确」→ 用转写原文，不纠错
  unknown       其它            不执行

## 架构（V5 集成）
  wav → A(Qwen3-ASR audio_tower 18层, 冻结) → ctc_head(可训 5.95M)
      ├─ V1: p=softmax(logits/τ) → z=pᵀE_pad → 压缩r=3 → [CLS,z] → B → CLS1
      └─ V3: CTC解码 → 字符串 → dmeta分词 → 精确词嵌入行 → B → CLS2
      → concat(CLS1,CLS2) 1536维 → intent_head → 8 类

  ⭐ V1 提供软容错，V3 提供与纯文本路径的对齐（cos 0.95，V1 只有 0.35）。
     两者拼起来最优：V1 97.87% / V3 95.51% / V5 集成 98.11%。

  ⭐⭐ 关键：转写字符串【必须过 dmeta 分词器】再查词嵌入行，不能逐字符查。
     实测 cos 从 0.33 → 0.95（逐字符 → 分词后），token 数 9.4 → 9.2
     （纯文本路径 9.3，对齐了）。

## 常开流式
  Qwen3-ASR 的 audio_tower 本来就是 chunk 化的（n_window=50 → chunk=1s，
  n_window_infer=800 → 注意力窗口 8s，卷积逐 chunk 独立），流式不用改架构。
  实测听到句子的 60%（1.71s）就已经满精度。

## CI（TARGET=voiceagent）
  ⚠️ Jenkins agent 是 Linux，PyInstaller 打不了 Windows exe。
     所以 CI 只做能在 Linux 上做且有意义的三件事：
       · intent_config.json 结构校验
       · 规则分句测试（纯 stdlib，与 Windows 端 test_agent.py 共用 testcases.json）
       · 源码语法检查
     exe 由 Windows 侧 voice_agent/build_exe.ps1 生成。

  ⭐ 本次 CI 校验抓到 6 个真 bug（规则分句 21/27 → 26/27）：
       「上知乎」「访问GitHub」「看看X文件」→ unknown（open 触发词缺 上/访问/看看）
       「帮我查一下X」→ 参数多带「一下」（「帮我查」抢在「查一下」前面）
       「帮我查找文件X」→ 被「帮我查」抢成 web_search
       「帮我精确查找X」→ 参数多带「帮我」（find 匹配后前缀没剥掉）
     修法：加 _norm() 先剥礼貌前缀（帮我/给我/请/麻烦…）+ 补触发词。

## 不破坏现有部署
  · k8s/live/ 一个字节没动（models.lock 也没动）
  · Jenkinsfile 只在【新增】的 stage ⓪ 和 4 个 when 守卫上加了
    `params.TARGET != 'voiceagent'`，旧 3 个 target 的 profile 逐字不变
    （已用 target.sh --print 对照验证）
  · 回归实测：merged-intent 1/1、vllm-asr 1/1、vllm-embed 1/1 都没动
MSG
git --no-pager log --oneline -1
echo
echo "════════ ⑥ 推送 ════════"
git push origin HEAD 2>&1 | tail -3
