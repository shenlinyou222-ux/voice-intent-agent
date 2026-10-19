# -*- coding: utf-8 -*-
"""从 desktop profile 抽出【模型层】，生成 voice_agent 用的 --patch 覆盖文件。

为什么需要：
  headless profile 的 cordis.patch.yml 是空的 []，它会落到 bundle 默认的
  llm-deepseek(api.deepseek.com)；而那个 key 余额为 0 → 任务立刻 402 QUOTA。
  我们这个会话（desktop profile）实际用的是 provider=opencode-go /
  model=deepseek-v4.1-flash，密钥从 ~/.dsh/.credentials.yaml 的 refs 里按
  apiKeyEnv 名字解析。
所以把 desktop 的模型层原样抽出来，语音启动时用 --patch 套上。
"""
from pathlib import Path

DESKTOP = Path(r"C:\Users\user\.dsh\profiles\desktop\cordis.patch.yml")
OUT = Path(__file__).parent / "dsh_model_patch.yml"

HDR = [
    "# 语音启动 DSH 会话时套用的【模型层】—— 与 desktop profile 一致。",
    "# 为什么需要：headless 的 cordis.patch.yml 是空的 []，会落到 bundle 默认的",
    "# llm-deepseek(api.deepseek.com)，而那个 key 余额为 0 → 任务立刻 402 QUOTA。",
    "# 用 `dsh headless --patch dsh_model_patch.yml ...` 覆盖 provider/model；",
    "# 密钥仍从 ~/.dsh/.credentials.yaml 的 refs 按 apiKeyEnv 名字解析。",
    "",
]


def main():
    lines = DESKTOP.read_text(encoding="utf-8").splitlines()
    # 找 llm-deepseek 那段起点，到 "- id: permission" 之前
    start = next(i for i, l in enumerate(lines)
                 if l.strip().startswith("- id: llm-deepseek"))
    end = next(i for i, l in enumerate(lines)
               if l.strip().startswith("- id: permission"))
    layer = lines[start:end]
    OUT.write_text("\n".join(HDR + layer) + "\n", encoding="utf-8")
    print(f"desktop patch 行 {start+1}..{end} → {OUT.name}"
          f"（{OUT.stat().st_size} 字节 / {len(HDR)+len(layer)} 行）")
    for l in layer[:6]:
        print("   ", l)
    print("    ...")
    for l in layer[-3:]:
        print("   ", l)
    # 校验 YAML 能解析（yaml 不一定装了，装了才校验）
    try:
        import yaml
        data = yaml.safe_load(OUT.read_text(encoding="utf-8"))
        ids = [d.get("id") for d in data]
        print(f"\nYAML 解析通过：{len(data)} 个条目 id={ids}")
    except ImportError:
        print("\n（没装 pyyaml，跳过解析校验）")
    except Exception as e:
        print(f"\n❌ YAML 解析失败: {type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
