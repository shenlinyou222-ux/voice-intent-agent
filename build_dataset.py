# -*- coding: utf-8 -*-
"""从 intent_config.json 生成训练数据 + TTS 任务表

8 个意图：
  launch_app    启动<程序>        24 个闭集
  open_url      打开<网址>        16 个闭集
  open_file     打开<文件>        7 个闭集
  open_web      打开<任意>        不在闭集 → 默认浏览器
  web_search    搜索<词>          默认浏览器搜索
  file_search   搜索文件<词>      文件管理器搜索
  exact_search  精确查找<词>      必须含"精确"
  unknown       其它              不执行

难点（模型必须学会的区分）：
  「打开B站」        → open_url（在闭集）
  「打开github.com」 → open_web（不在闭集）
  「搜索python」     → web_search
  「搜索文件python」 → file_search    ← 只差"文件"两个字
  「精确查找python」 → exact_search   ← 只差"精确"两个字
"""
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
CFG = json.loads((ROOT / "intent_config.json").read_text(encoding="utf-8"))
SEED = 20261011
rng = random.Random(SEED)


def sid(split, intent, text):
    """⭐ ID 必须由【文本内容】决定，不能用序号。

    踩过的坑：原来用 `{split}_{intent}_{序号}`，序号是全局 shuffle 之后的位置。
    一扩充数据集 → shuffle 变了 → 序号对应的文本全变了，
    但 tts_batch.ps1 会【跳过已存在的 wav】→ 复用了旧文件
    → audio 和 label 错位 → 意图精度看着还行（类别编码在 ID 里），
      但 CTC 转写彻底崩（0/609，模型输出「s放器」这种垃圾）。
    """
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return f"{split}_{intent}_{h}"

T_LAUNCH = ["启动{名}", "打开{名}", "运行{名}", "开一下{名}", "开个{名}", "给我开{名}",
            "帮我打开{名}", "帮我启动{名}", "来一下{名}", "我要用{名}", "把{名}打开",
            "把{名}启动起来", "{名}打开一下", "现在打开{名}", "帮我运行{名}", "开一下{名}吧"]
T_OPEN_URL = ["打开{名}", "开{名}", "帮我打开{名}", "打开{名}网页", "打开{名}网站",
              "上{名}", "我要上{名}", "访问{名}", "去{名}", "看看{名}", "帮我开{名}",
              "把{名}打开", "打开{名}页面", "{名}打开", "上个{名}", "我要看{名}"]
T_OPEN_FILE = ["打开{名}文件", "打开{名}文档", "打开{名}", "帮我打开{名}文件",
               "打开{名}这个文件", "把{名}文件打开", "我要看{名}文件", "打开{名}文件看看",
               "打开{名}这个文档", "帮我打开{名}", "看看{名}文件", "打开{名}文档看一下"]
T_OPEN_WEB = ["打开{名}", "帮我打开{名}", "打开{名}看看", "开一下{名}", "我要打开{名}",
              "打开这个{名}", "帮我开{名}", "去{名}", "打开{名}吧", "访问{名}"]
T_WEB_SEARCH = ["搜索{名}", "搜一下{名}", "查一下{名}", "帮我搜{名}", "帮我查{名}",
                "搜个{名}", "搜索一下{名}", "找一下{名}", "帮我搜索{名}", "我要搜{名}",
                "搜索{名}看看", "查查{名}", "网上搜一下{名}", "帮我找{名}"]
T_FILE_SEARCH = ["搜索文件{名}", "查找文件{名}", "搜文件{名}", "找文件{名}",
                 "帮我搜索文件{名}", "帮我找文件{名}", "搜索{名}这个文件", "查找{名}文件",
                 "在文件里搜{名}", "帮我搜一下文件{名}", "搜一下文件{名}", "找一下文件{名}"]
T_EXACT = ["精确查找{名}", "精确搜索{名}", "精确搜{名}", "帮我精确查找{名}",
           "精确查找一下{名}", "帮我精确搜索{名}", "精确搜一下{名}", "精确查找{名}这个",
           "精确搜索一下{名}", "我要精确查找{名}", "精确查找{名}试试"]

# 自由文本参数池（open_web / web_search / file_search / exact_search 共用）
FREE = [
    # 网址类（open_web 用）
    "github.com", "bilibili.com", "zhihu.com", "pypi.org", "arxiv.org",
    "huggingface.co", "modelscope.cn", "stackoverflow.com", "csdn.net",
    # 搜索词类
    "python教程", "resnet50", "bert模型", "量化方法", "语音识别",
    "cuda编程", "矩阵乘法优化", "注意力机制", "transformer架构", "模型压缩",
    "numpy用法", "pytorch安装", "docker教程", "git命令", "linux内核",
    "知识蒸馏", "lora微调", "onnx导出", "推理加速", "显存优化",
    # 文件名类（file_search 用）
    "季度报告", "会议纪要", "实验数据", "日志文件", "配置文件",
    "训练脚本", "测试报告", "论文草稿", "预算表", "客户名单",
]
WEB_URLS = FREE[:9]
FREE_WORDS = FREE[9:]

# ══════════ unknown：闲聊（易负样本）+ 硬负样本 ══════════
# ⚠️ 上一版把硬负样本堆到 101 条 → CTC 头崩了（转写退化成几个字），
#    而且 open_url→unknown 误拒 3 条。教训：
#      · 硬负样本要【精选】不要堆量，尤其是裸触发词「打开」「搜索」这类
#        本身是合法前缀的说法，堆多了会让模型对合法短命令也拒识
#      · unknown 主要靠【闲聊】撑量（易负样本，不干扰 CTC）
CHITCHAT = [
    "今天天气怎么样", "现在几点了", "你是谁", "讲个笑话", "帮我订一张高铁票",
    "明天会下雨吗", "一加一等于几", "我饿了", "唱首歌给我听", "把电脑关了",
    "帮我把这个文件删掉", "今天星期几", "你叫什么名字", "给我推荐一部电影",
    "我心情不好", "外面冷吗", "帮我写一封邮件", "重启一下电脑", "把音量调到最大",
    "今年是几几年", "北京到上海多远", "给我讲个故事", "你会做饭吗", "帮我订个外卖",
    "现在股市怎么样", "帮我算一下房贷", "明天要开会吗", "这个多少钱",
    "我想睡觉了", "帮我倒杯水", "我的快递到哪了", "附近有什么好吃的",
    "这个周末有安排吗", "帮我记一下", "我忘了带钥匙", "地铁几点停运",
    "帮我查一下话费", "今天有什么新闻", "帮我关一下灯", "我手机没电了",
    "这个怎么做才对", "你懂我的意思吗", "刚才说到哪了", "我有点累了",
    "帮我看看这个", "算了不用了", "等一下再说", "我先走了", "好的谢谢",
    "帮我把空调打开", "现在外面热不热", "我明天几点起床", "这个字怎么读",
    "帮我翻译一句话", "我的快递什么时候到", "今天适合出门吗", "我该穿什么衣服",
    "晚饭吃什么好", "帮我算算这个月花了多少", "这个电影好看吗", "附近有停车场吗",
    "帮我把闹钟关掉", "我想听点音乐", "现在路况怎么样", "帮我查一下航班",
    "我最近有点失眠", "这个多少钱能买到", "帮我记一下这个号码", "明天要带伞吗",
    "这个东西怎么用", "帮我把这个发给他", "我什么时候有空",
]
# 硬负样本：精选 30 条【明确不是命令】的说法
HARD_NEG = [
    # 半截话（说了但没说要干什么）
    "我想搜索一下", "能不能搜索", "帮我看看", "这个怎么打开", "文件在哪",
    "刚才那个搜索", "把搜索关了", "搜索不到", "没有精确", "打开过了",
    # 疑问 / 元讨论
    "怎么打开一个文件", "搜索功能怎么用", "为什么打不开", "打开是什么意思",
    "你刚才说什么", "再说一遍", "我没说打开", "不是这个意思",
    "打开和启动有什么区别", "这句话怎么说", "帮我把这个翻译一下",
    # 含触发词但说的是状态/故障
    "我的浏览器打不开了", "这个网站打不开", "帮我看看为什么打不开",
    "刚才那个程序卡住了", "电脑有点卡", "网络好像断了",
    "现在开着吗", "打开了吗", "好了没有", "还在吗", "可以了吗",
    "刚才那个执行了吗", "有没有打开成功", "是不是已经开了",
]

rows = []


def add(text, intent, arg=None, kind="closed"):
    rows.append({"text": text, "intent": intent, "arg": arg, "kind": kind})


for a in CFG["apps"]:
    for t in T_LAUNCH:
        add(t.format(名=a["name"]), "launch_app", a["name"])
for u in CFG["urls"]:
    for t in T_OPEN_URL:
        add(t.format(名=u["name"]), "open_url", u["name"])
for f in CFG["files"]:
    for t in T_OPEN_FILE:
        add(t.format(名=f["name"]), "open_file", f["name"])
# open_web：网址类自由文本
for w in WEB_URLS:
    for t in T_OPEN_WEB:
        add(t.format(名=w), "open_web", w, "freetext")
# web_search
for w in FREE_WORDS:
    for t in T_WEB_SEARCH:
        add(t.format(名=w), "web_search", w, "freetext")
# file_search
for w in FREE_WORDS:
    for t in T_FILE_SEARCH:
        add(t.format(名=w), "file_search", w, "freetext")
# exact_search
for w in FREE_WORDS:
    for t in T_EXACT:
        add(t.format(名=w), "exact_search", w, "freetext")
# unknown
for c in CHITCHAT:
    add(c, "unknown", None, "ood")
for c in HARD_NEG:
    add(c, "unknown", None, "ood_hard")

# 去重 + 分层
seen = set(); ded = []
for r in rows:
    if r["text"] in seen:
        continue
    seen.add(r["text"]); ded.append(r)
rng.shuffle(ded)

by_i = {}
for r in ded:
    by_i.setdefault(r["intent"], []).append(r)
jobs = []
for it in sorted(by_i):
    pool = by_i[it][:]
    rng.shuffle(pool)
    k = max(1, int(round(len(pool) * 0.7)))
    for j, r in enumerate(pool):
        sp = "train" if j < k else "val"
        jobs.append({"id": sid(sp, it, r["text"]), "text": r["text"], "intent": it,
                     "kind": r["kind"], "split": sp, "arg": r.get("arg")})

c = Counter(j["intent"] for j in jobs)
cv = Counter(j["intent"] for j in jobs if j["split"] == "val")
print("=" * 78)
print(f"  意图数 {len(c)}   总样本 {len(jobs)}   "
      f"train {sum(1 for j in jobs if j['split']=='train')}   val {sum(1 for j in jobs if j['split']=='val')}")
print("=" * 78)
for it in sorted(c):
    print(f"  {it:14s} 总 {c[it]:>4}   val {cv[it]:>4}")

io.open(ROOT / "intent_dataset.jsonl", "w", encoding="utf-8").write(
    "\n".join(json.dumps(j, ensure_ascii=False) for j in jobs))
io.open(ROOT / "intent_tts.tsv", "w", encoding="utf-8").write(
    "\n".join(f"{j['id']}\t{j['text']}" for j in jobs))
print(f"\n-> intent_dataset.jsonl  intent_tts.tsv  ({len(jobs)} 条)")
