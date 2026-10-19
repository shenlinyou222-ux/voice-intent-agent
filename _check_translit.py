# -*- coding: utf-8 -*-
"""拉丁白名单 ↔ 中文转写 双向核对(2026-10-18 统一标注标定)。

对 _ENG_EXCEPT 每个词,检查其中文转写(按 pypinyin 单字读法)是否在
latin_variants() 的变体集合里 —— 读法不一致(多音字/白名单手写错)会死在
模板子序列匹配上(克拉什 什(sh,en) vs 手写 (sh,i) 的教训)。
"""
import sys

sys.path.insert(0, r"C:\Users\user\Desktop\ds harness\voice_agent")

from phon_match import _ENG_EXCEPT, latin_variants, syllables  # noqa: E402

CHECK = {
    "deepseek": "迪普西克", "clash": "克拉什", "trae": "特雷", "ollama": "欧拉玛",
    "harness": "哈尼斯", "github": "吉特哈布", "llama": "拉玛", "python": "派森",
    "shell": "谢尔", "dvc": "迪威西", "api": "诶派艾", "gpt": "吉皮提",
    "blender": "布伦德", "steam": "斯提姆", "edge": "艾吉", "cmd": "西艾姆迪",
    "doubao": "豆包", "qwen": "千问", "jiqizhixin": "机器之心", "wiki": "维基",
    "wikipedia": "维基百科", "arxiv": "阿克赛唯", "anthropic": "安思罗匹克",
    "claude": "克劳德", "chatgpt": "恰特吉皮提", "huggingface": "哈京费斯",
    "docker": "道克", "gitee": "吉提", "modelscope": "莫德勒斯扣",
    "modelcope": "莫德勒口坡",
    "openai": "欧盆艾", "oschina": "开源中国",
    "paperswithcode": "陪珀斯危思口德",
    "powershell": "泡沃谢尔", "pypi": "派派", "youtube": "油管", "work": "沃克",
    "buddy": "巴迪", "stack": "斯泰克", "papers": "陪珀斯", "with": "危思",
    "code": "口德", "googlescholar": "谷歌学术", "hackernews": "哈克新闻",
    "infoq": "因弗求", "news": "牛斯", "hacker": "哈克",
    "dockerhub": "道克哈布", "hub": "哈布", "cnblogs": "博客园",
    "kaggle": "卡盖尔", "stackoverflow": "斯泰克奥沃弗洛", "workbuddy": "沃克巴迪",
}


def _sig(seq):
    return " ".join(f"{a}/{b}" for a, b in seq)


def main():
    bad = 0
    for w, zh in CHECK.items():
        vs = latin_variants(w)
        want = syllables(zh)
        if any(list(v) == list(want) for v in vs):
            continue
        bad += 1
        print(f"MISS {w!r} (中文={zh}): 期望 {_sig(want)}")
        have = [" ".join(f"{a}/{b}" for a, b in v) for v in vs[:3]]
        print(f"      现有: {have}")
    print()
    print(f"共 {len(CHECK)} 条,缺失 {bad} 条")


if __name__ == "__main__":
    main()
