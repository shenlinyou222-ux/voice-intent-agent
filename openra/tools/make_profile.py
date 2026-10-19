# -*- coding: utf-8 -*-
"""生成【版本化】的 OpenRA 语音控制 profile。

为什么要生成而不是手写：
  · profile 里的三样东西必须能独立版本化 —— 热词(hotwords) / 意图示例+embedding 索引(index) / 指令映射
  · 每样都算出内容哈希，写进 profile 的 *-version 字段；运行时装的是哪一版一目了然
  · 加一条口令 = 改这个脚本 + 重跑（毫秒级），不碰模型权重

产出：
  openra/profiles/<id>/profile.json     —— 运行时读的总配置（含版本戳）
  openra/profiles/<id>/hotwords.txt     —— 给 ASR 的热词（版本化，一行一个）
  openra/profiles/<id>/intents.jsonl    —— 意图示例语料（给 build_profile_index.py 建 embedding 索引）

用法：
  python openra/tools/make_profile.py                 # 默认 openra-v1
  python openra/tools/make_profile.py --id openra-v2  # 新版本（改内容后）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent          # openra/tools
OPENRA_DIR = HERE.parent                        # openra/
PROFILES = OPENRA_DIR / "profiles"

# ══════════════════════════════════════════════════════════════
#  1. 单位/建筑表（actor 名与队列名取自 E:\OpenRA\mods\ra\rules 实测）
# ══════════════════════════════════════════════════════════════
ITEMS = [
    # (actor, 中文名, 英文名, 队列, 别名…)
    ("HARV", "采矿车", "Ore Truck", "Vehicle", ["矿车", "采矿车", "矿卡", "采矿卡车", "挖矿车"]),
    ("MCV", "基地车", "Mobile Construction Vehicle", "Vehicle", ["基地车", "mcv", "建造车"]),
    ("2TNK", "中型坦克", "Medium Tank", "Vehicle", ["坦克", "中坦", "中型坦克", "主战坦克"]),
    ("3TNK", "重型坦克", "Heavy Tank", "Vehicle", ["重坦", "重型坦克", "重坦克"]),
    ("4TNK", "猛犸坦克", "Mammoth Tank", "Vehicle", ["猛犸", "猛犸坦克", "天启", "天启坦克"]),
    ("1TNK", "轻型坦克", "Light Tank", "Vehicle", ["轻坦", "轻型坦克", "轻坦克"]),
    ("TTNK", "特斯拉坦克", "Tesla Tank", "Vehicle", ["特斯拉坦克", "磁暴坦克", "电击坦克"]),
    ("CTNK", "时空坦克", "Chrono Tank", "Vehicle", ["时空坦克", "超时空坦克"]),
    ("V2RL", "V2 火箭车", "V2 Rocket Launcher", "Vehicle", ["v2", "火箭车", "导弹车", "v2火箭"]),
    ("ARTY", "火炮", "Artillery", "Vehicle", ["火炮", "大炮", "榴弹炮"]),
    ("APC", "装甲运兵车", "Armored Personnel Carrier", "Vehicle", ["运兵车", "装甲车", "运输车"]),
    ("JEEP", "吉普车", "Ranger", "Vehicle", ["吉普", "吉普车", "侦察车"]),
    ("MNLY", "布雷车", "Minelayer", "Vehicle", ["布雷车", "地雷车"]),
    ("FTRK", "防空车", "Mobile Flak", "Vehicle", ["防空车", "移动防空"]),
    ("DTRK", "自爆卡车", "Demolition Truck", "Vehicle", ["自爆卡车", "炸弹车"]),
    ("E1", "步枪兵", "Rifle Infantry", "Infantry", ["步兵", "步枪兵", "机枪兵", "动员兵"]),
    ("E3", "火箭兵", "Rocket Soldier", "Infantry", ["火箭兵", "导弹兵", "反坦克兵"]),
    ("E2", "手雷兵", "Grenadier", "Infantry", ["手雷兵", "掷弹兵"]),
    ("E4", "喷火兵", "Flame Infantry", "Infantry", ["喷火兵", "火焰兵"]),
    ("E6", "工程师", "Engineer", "Infantry", ["工程师", "工程兵"]),
    ("E7", "谭雅", "Tanya", "Infantry", ["谭雅", "谭娅"]),
    ("DOG", "警犬", "Attack Dog", "Infantry", ["狗", "警犬", "军犬"]),
    ("SHOK", "震击兵", "Shock Trooper", "Infantry", ["震击兵", "电击兵"]),
    ("MEDI", "医疗兵", "Medic", "Infantry", ["医疗兵", "医生"]),
    ("MECH", "机械师", "Mechanic", "Infantry", ["机械师", "修理工"]),
    ("THF", "小偷", "Thief", "Infantry", ["小偷", "盗贼"]),
    ("POWR", "发电厂", "Power Plant", "Building", ["电厂", "发电厂", "电站"]),
    ("APWR", "高级发电厂", "Advanced Power Plant", "Building", ["高级电厂", "高级发电厂"]),
    ("TENT", "盟军兵营", "Allied Barracks", "Building", ["兵营", "盟军兵营"]),
    ("BARR", "苏军兵营", "Soviet Barracks", "Building", ["苏军兵营", "苏联兵营"]),
    ("WEAP", "战车工厂", "War Factory", "Building", ["战车工厂", "坦克工厂", "重工厂"]),
    ("PROC", "矿石精炼厂", "Ore Refinery", "Building", ["精炼厂", "矿厂", "矿石精炼厂"]),
    ("DOME", "雷达", "Radar Dome", "Building", ["雷达", "雷达站"]),
    ("FIX", "维修厂", "Service Depot", "Building", ["维修厂", "修理厂", "维修工厂", "修理工厂",
                                                  "维修中心", "修理中心", "维修车间", "修理车间"]),
    ("AFLD", "机场", "Airfield", "Building", ["机场", "飞机场"]),
    ("HPAD", "直升机坪", "Helipad", "Building", ["直升机坪", "停机坪"]),
    ("SYRD", "船厂", "Naval Yard", "Building", ["船厂", "造船厂"]),
    ("KENN", "狗舍", "Kennel", "Building", ["狗舍", "犬舍"]),
    ("ATEK", "盟军科技中心", "Allied Tech Center", "Building", ["盟军科技中心", "科技中心"]),
    ("STEK", "苏军科技中心", "Soviet Tech Center", "Building", ["苏军科技中心"]),
    ("GUN", "炮台", "Turret", "Defense", ["炮台", "机枪塔"]),
    ("AGUN", "防空炮", "AA Gun", "Defense", ["防空炮", "防空"]),
    ("TSLA", "特斯拉线圈", "Tesla Coil", "Defense", ["特斯拉", "特斯拉线圈", "电塔", "磁暴线圈"]),
    ("SAM", "防空导弹", "SAM Site", "Defense", ["防空导弹", "导弹塔"]),
    ("IRON", "铁幕装置", "Iron Curtain", "Defense", ["铁幕", "铁幕装置"]),
    ("PDOX", "超时空传送仪", "Chronosphere", "Defense", ["超时空传送仪", "传送仪"]),
    ("GAP", "干扰器", "Gap Generator", "Defense", ["干扰器", "幻影发生器"]),
    ("SILO", "矿石储存罐", "Silo", "Defense", ["矿仓", "储存罐"]),
    ("SS", "潜艇", "Submarine", "Ship", ["潜艇"]),
    ("DD", "驱逐舰", "Destroyer", "Ship", ["驱逐舰"]),
    ("CA", "巡洋舰", "Cruiser", "Ship", ["巡洋舰"]),
    ("PT", "炮艇", "Gunboat", "Ship", ["炮艇", "快艇"]),
    ("LST", "运输舰", "Transport", "Ship", ["运输舰", "登陆艇"]),
    ("MSUB", "导弹潜艇", "Missile Submarine", "Ship", ["导弹潜艇", "核潜艇"]),
    ("HELI", "长弓直升机", "Longbow", "Aircraft", ["长弓", "直升机"]),
    ("HIND", "雌鹿直升机", "Hind", "Aircraft", ["雌鹿"]),
    ("MH60", "黑鹰直升机", "Black Hawk", "Aircraft", ["黑鹰"]),
    ("MIG", "米格战机", "MiG Attack Plane", "Aircraft", ["米格", "米格战机"]),
    ("YAK", "雅克攻击机", "Yak Attack Plane", "Aircraft", ["雅克", "攻击机"]),
]

# 队列表：说「所有坦克」→ 选中整条生产线的东西
QUEUES = {
    "Vehicle": ["所有车辆", "全部车辆", "所有载具", "坦克部队", "所有坦克", "装甲部队"],
    "Infantry": ["所有步兵", "全部步兵", "步兵部队"],
    "Building": ["所有建筑", "全部建筑", "我的建筑"],
    "Defense": ["所有防御", "全部防御", "防御建筑"],
    "Ship": ["所有船", "全部海军", "海军"],
    "Aircraft": ["所有飞机", "全部空军", "空军"],
}

STANCES = {
    "HoldFire": ["停火", "不开火", "禁止开火"],
    "ReturnFire": ["还击", "反击"],
    "Defend": ["防御", "防守"],
    "AttackAnything": ["主动攻击", "自由开火", "见谁打谁"],
}

# ══════════════════════════════════════════════════════════════
#  2. 意图（游戏专用；用户明确说原有桌面意图可以完全丢掉）
#     每个意图给一批说法，用来建 embedding 索引（kNN）
# ══════════════════════════════════════════════════════════════
INTENTS = [
    {
        "id": "game_select", "desc": "选中单位/建筑", "needs_arg": "filter",
        "triggers": ["选中", "选择", "选", "切换到", "全选"],
        "examples": [
            "选中所有部队", "选中全部坦克", "选中所有步兵", "选中所有采矿车", "选中所有建筑",
            "全选部队", "选择所有单位", "选中所有防御建筑", "选中所有飞机", "选中所有船",
            "选中空闲的单位", "选中附近的部队", "把坦克都选上", "选择全部部队", "选中我的采矿车",
            "选中所有车辆", "把步兵选上",
        ],
    },
    {
        "id": "game_move", "desc": "把选中的单位移动到坐标", "needs_arg": "x,y",
        "triggers": ["移动到", "移动", "去", "走到", "开往", "前进到", "到"],
        "examples": ["移动到 60 40", "去 30 20", "部队开到 50 50", "移动到坐标 20 30",
                     "所有坦克去 80 60", "把部队移动到 10 10", "前进到 40 40",
                     "部队往 60 40 走", "把所有单位调到 20 20", "让坦克去 30 40",
                     "移动到指定坐标 10 20", "部队挪到 50 50", "移动到这个位置 64 64"],
    },
    {
        "id": "game_attack", "desc": "攻击目标（坐标或最近的敌人）", "needs_arg": "x,y|nearest",
        "triggers": ["攻击", "打", "揍", "集火", "消灭", "干掉"],
        "examples": ["攻击这个目标", "攻击最近的敌人", "打他", "集火那个建筑", "消灭敌人",
                     "攻击 60 40", "干掉那个炮台", "所有部队攻击",
                     "打掉那个建筑", "对那个目标开火", "集火攻击 60 40 的敌人",
                     "攻击他的基地", "把那个采矿车打掉"],
    },
    {
        "id": "game_attackmove", "desc": "攻击前进", "needs_arg": "x,y",
        "triggers": ["攻击前进", "边打边进", "推进到", "进攻到"],
        "examples": ["攻击前进到 60 40", "部队攻击前进", "边打边进到 50 50", "推进到 80 80",
                     "攻击前进到敌方基地", "边打边推进 70 70", "带兵攻击前进 40 40",
                     "进攻推进到 60 60"],
    },
    {
        "id": "game_stop", "desc": "停止当前动作", "needs_arg": None,
        "triggers": ["停止", "停下", "停", "别动", "取消动作"],
        "examples": ["停止", "全部停下", "别动了", "取消当前动作", "停下来",
                     "取消所有动作", "停止当前命令", "所有单位停止"],
    },
    {
        "id": "game_scatter", "desc": "散开", "needs_arg": None,
        "triggers": ["散开", "分散", "散"],
        "examples": ["散开", "全部散开", "部队分散开"],
    },
    {
        "id": "game_guard", "desc": "守卫某点", "needs_arg": "x,y",
        "triggers": ["守卫", "防守", "守住", "保护"],
        "examples": ["守卫这个位置", "守住 40 40", "部队防守这里", "保护基地",
                     "守卫这个区域", "派兵守住 30 30", "保护这个位置", "在原地防守"],
    },
    {
        "id": "game_stance", "desc": "切换单位姿态", "needs_arg": "stance",
        "triggers": ["姿态", "切换姿态", "开火模式"],
        "examples": ["姿态防御", "切换成停火", "单位还击", "自由开火", "姿态主动攻击",
                     "把姿态切成防御", "单位改为自由开火", "切换姿态为还击"],
    },
    {
        "id": "game_harvest", "desc": "采矿车去采矿", "needs_arg": None,
        "triggers": ["采矿", "挖矿", "去采矿", "采资源", "拉矿"],
        "examples": ["矿车去采矿", "开始采矿", "去挖矿", "所有矿车采资源",
                     "采矿车去挖矿", "让矿车拉矿", "开始采矿石", "矿车去采资源"],
    },
    {
        "id": "game_deploy", "desc": "基地车展开", "needs_arg": None,
        "triggers": ["展开", "部署", "建立基地", "展开基地"],
        "examples": ["基地车展开", "展开", "部署基地", "把基地车展开",
                     "基地车就地展开", "把基地车部署成基地"],
    },
    {
        "id": "game_build", "desc": "生产单位/建筑", "needs_arg": "item",
        "triggers": ["建造", "造", "生产", "造一个", "造两辆", "来一个", "出"],
        "examples": ["建造坦克", "造两个矿车", "生产步兵", "造一个电厂", "来一辆基地车",
                     "出五个火箭兵", "建造战车工厂", "造三辆重型坦克", "生产谭雅",
                     "造两辆重型坦克", "生产三个火箭兵", "建造一个雷达", "来两辆矿车",
                     "出四个步兵", "造个战车工厂"],
    },
    {
        "id": "game_place", "desc": "把造好的建筑放到地图上", "needs_arg": "x,y|auto",
        "triggers": ["放置", "放在", "放下", "放到", "摆下"],
        "examples": ["放置电厂", "把电厂放在这里", "放下建筑", "放到基地旁边",
                     "把造好的建筑放下来", "放置在 40 40"],
    },
    {
        "id": "game_group", "desc": "语音编队（控制组 1~9）", "needs_arg": "n,parts",
        "triggers": ["编队", "编为一队", "编为", "编成", "组队", "编入"],
        "examples": ["三辆坦克二步兵编为一队", "全部坦克编为一队", "所有矿车编为二队",
                     "选中所有步兵编成三队", "把坦克编入一队", "编为一队",
                     "两个步兵编为四队", "全部坦克编成二队"],
    },
    {
        "id": "game_explore", "desc": "自动探索（侦察兵沿路点探图）", "needs_arg": None,
        "triggers": ["探索", "侦察", "探路", "自动探索", "开图"],
        "examples": ["自动探索", "侦察兵去探索", "派个单位去探路", "把地图探开",
                     "让最快的单位去侦察", "开局去侦察", "去探图"],
    },
    {
        "id": "game_cancel", "desc": "取消生产", "needs_arg": "item",
        "triggers": ["取消", "取消建造", "不造了"],
        "examples": ["取消建造坦克", "取消生产步兵", "不造坦克了", "取消这个",
                     "取消建造中的坦克", "把生产队列取消", "停止生产步兵"],
    },
    {
        "id": "game_repair", "desc": "修理建筑", "needs_arg": None,
        "triggers": ["修理", "修复", "维修"],
        "examples": ["修理建筑", "修复这个建筑", "修理基地",
                     "修理受损的建筑", "把电厂修一下", "修复我的建筑"],
    },
    {
        "id": "game_sell", "desc": "出售建筑", "needs_arg": None,
        "triggers": ["出售", "卖掉", "卖了"],
        "examples": ["卖掉这个建筑", "出售电厂", "把这个炮台卖了",
                     "把电厂卖掉", "出售选中的建筑", "卖掉这个炮台"],
    },
    {
        "id": "game_capture", "desc": "占领建筑", "needs_arg": None,
        "triggers": ["占领", "夺取", "占"],
        "examples": ["占领这个建筑", "工程师占领", "夺取敌方基地",
                     "工程师去占领", "占领敌方的建筑", "夺取这个基地"],
    },
    {
        "id": "game_rally", "desc": "设置集结点", "needs_arg": "x,y",
        "triggers": ["集结点", "集合点", "设置集结点"],
        "examples": ["设置集结点 40 40", "集结点设在这里", "兵营集结点 30 30",
                     "把兵营集结点设在 40 40", "设置新的集结点"],
    },
    {
        "id": "game_state", "desc": "查询战况", "needs_arg": None,
        "triggers": ["战况", "情况", "有多少钱", "报告"],
        "examples": ["现在什么情况", "我有多少钱", "报告战况", "我还有多少单位", "电力够吗",
                     "报告当前战况", "我现在的资金", "还剩多少电力"],
    },
    {
        "id": "game_ping", "desc": "在地图上标记", "needs_arg": "x,y",
        "triggers": ["标记", "标个点", "ping"],
        "examples": ["标记这里", "在地图上标个点", "标记 50 50",
                     "在这里打个标记", "标记这个位置"],
    },
    {
        "id": "game_chat", "desc": "在游戏里喊话", "needs_arg": "text",
        "triggers": ["喊话", "说", "发消息", "告诉队友"],
        "examples": ["喊话说进攻", "告诉队友我来了", "发消息撤退",
                     "喊话告诉队友进攻", "在聊天里说撤退"],
    },
]


def sha(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def build(profile_id: str, version: int, intents=None) -> tuple[dict, list[str], list[tuple[str, str]]]:
    """intents=None 表示全量；给了就只保留这些意图（用于"收缩意图、一点点加"）。"""
    intents = intents or INTENTS
    active_ids = {i["id"] for i in intents}
    items_active = ITEMS
    """热词按【优先级】排（ASR 提示词会按 max_chars 截尾，重要的必须在前）：
       ① 中文单位/建筑名 + 别名（最容易被听错：「精炼厂」→「金炼厂」）
       ② 队列表/姿态词
       ③ 命令触发词（移动/攻击/停止… 这些常见词 ASR 本来就稳）
       ④ 英文单位名（中英混说时被听错的那一类）
       ⑤ 杂项
    ⚠️ 血债：原来把触发词放最前，结果 asr_hotword_chars=160 一截，
       单位名（矿车/电厂/精炼厂）全被截掉 → 「精炼厂」被转成「金炼厂」。
    """
    prio: list[list[str]] = [[], [], [], [], []]

    for it in ITEMS:
        _actor, cn, en, _queue, aliases = it
        prio[0] += [cn] + list(aliases)
    for words in list(QUEUES.values()) + list(STANCES.values()):
        prio[1] += words
    for intent in intents:
        prio[2] += intent["triggers"]
    for it in ITEMS:
        prio[3].append(it[2])
    prio[4] += ["坐标", "敌人", "目标", "基地", "部队", "单位", "附近", "最近", "所有", "全部"]

    hotwords, seen = [], set()
    for group in prio:
        for w in group:
            w = (w or "").strip()
            if w and w.lower() not in seen:
                seen.add(w.lower())
                hotwords.append(w)

    items = [
        {"id": actor, "name": cn, "en": en, "queue": queue, "aliases": sorted({cn, en, actor, *aliases})}
        for actor, cn, en, queue, aliases in ITEMS
    ]

    examples = []
    for intent in intents:
        for ex in intent["examples"]:
            examples.append((ex, intent["id"]))

    hotword_blob = "\n".join(hotwords)
    intent_blob = json.dumps(
        [{"id": i["id"], "examples": i["examples"], "triggers": i["triggers"]} for i in intents],
        ensure_ascii=False, sort_keys=True)

    profile = {
        "profile_id": profile_id,
        "version": version,
        "domain": "openra",
        "note": "游戏专用 profile：原有桌面意图（打开程序/网址/搜索）不在本文件里 —— 用户明确要求丢掉。",
        # ── 版本戳：三样东西各自独立版本化 ──
        "hotwords_version": f"{profile_id}-hw-{sha(hotword_blob)}",
        "intent_version": f"{profile_id}-intent-{sha(intent_blob)}",
        "index_version": None,          # build_profile_index.py 建成后回填
        "index_file": "index.json",
        # ── 后台推理服务（embedding 重训练用的就是它）──
        "embed": {
            "url": "http://172.20.71.101:30802/v1/embeddings",
            "model": "dmeta-small",
            "thr": 0.30,
            "margin": 0.0,
        },
        # ── 游戏内桥 ──
        # timeout 给足：游戏窗口失焦/最小化时 OpenRA 主循环会掉帧，
        # 逻辑 tick 变稀，3s 会偶发超时（实测踩过），10s 稳。
        "bridge": {"host": "127.0.0.1", "port": 38900, "token": None, "timeout": 10.0},
        "game": {"mod": "ra", "coord_max": 128},
        # ── 规则直通（不依赖模型，先命中先执行）──
        "triggers": {i["id"]: i["triggers"] for i in intents},
        "intents": [{"id": i["id"], "desc": i["desc"], "needs_arg": i["needs_arg"]} for i in intents],
        "items": items,
        "queues": QUEUES,
        "stances": STANCES,
        "hotwords_file": "hotwords.txt",
        "hotwords_count": len(hotwords),
    }
    return profile, hotwords, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="openra-v1")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--intents", default=None,
                    help="只保留这些意图（逗号分隔）——「收缩意图、一点点加」用；默认全量。"
                         "例：--intents game_build,game_cancel,game_group,game_explore")
    args = ap.parse_args()

    intents = None
    if args.intents:
        want = [x.strip() for x in args.intents.split(",") if x.strip()]
        by_id = {i["id"]: i for i in INTENTS}
        unknown = [w for w in want if w not in by_id]
        if unknown:
            print(f"未知意图: {unknown}")
            return 1
        intents = [by_id[w] for w in want]

    profile, hotwords, examples = build(args.id, args.version, intents)
    out = PROFILES / args.id
    out.mkdir(parents=True, exist_ok=True)

    (out / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "hotwords.txt").write_text("\n".join(hotwords) + "\n", encoding="utf-8")
    with (out / "intents.jsonl").open("w", encoding="utf-8") as f:
        for text, intent in examples:
            f.write(json.dumps({"text": text, "intent": intent}, ensure_ascii=False) + "\n")

    print(f"profile   : {out / 'profile.json'}")
    print(f"hotwords  : {out / 'hotwords.txt'}  ({len(hotwords)} 条, {profile['hotwords_version']})")
    print(f"intents   : {len(profile['intents'])} 个意图 / {len(examples)} 条示例")
    print(f"items     : {len(profile['items'])} 个可生产条目")
    print(f"intent_ver: {profile['intent_version']}")


if __name__ == "__main__":
    main()
