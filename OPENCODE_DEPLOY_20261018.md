# OPENCODE 部署标记（2026-10-18/19）

本文件由 **OPENCODE** 生成，标记一次**新增 + 5 处带备份补丁 + k8s realtime 启用**的部署。

| 项 | 值 |
|---|---|
| 生成者 | opencode（`ses_eb3637506ffeoeQ5UIVdyl40q4`） |
| 时间 | 2026-10-18 / 2026-10-19 |
| 新增 | `opencode_fastpath/`、`run_opencode_shadow.cmd`、`run_opencode_live.cmd`、本文件 |
| 修改 | `voice_agent.py`（**6 处 `[OPENCODE-*]`**：A 路段转写后 / B 部分文本 / C realtime 启动 / D 音频块推送 / E 段边界 / F PTT 启动崩溃修复） |
| 备份 | `voice_agent.py.bak-opencode-20261018` |
| k8s | `default/vllm-asr` args 追加 `--hf-overrides={"architectures":["Qwen3ASRRealtimeGeneration"]}`（启用 `/v1/realtime`；回滚数组在 `k8s/rollback_args.json`） |
| 自检 | `python -m opencode_fastpath.selfcheck` → **16/16 PASS** |
| 回滚 | `python -m opencode_fastpath.patch --rollback` + k8s rollback + 删新增路径，见 `opencode_fastpath/ROLLBACK.md` |

## 这一层补的是什么

`voice_agent` 已有 `--dry-run`、规则直通、闭集解析、`phon_match` 近似打分 —— 本层**不替代**它们，
只补"规则判 unknown 之后"的**精确学习表**：

1. 拼音掩码（同音自动同键）→ 一次整数查表 → `(intent, arg)`；
2. **A 路接入**：段转写完成即查表（live 命中 → 提前执行并跳过本段 `handle`，`_a_executed` 保证 B 路不重复）；
3. **真实流式早火**：`AgentStream`（WS `/v1/realtime`）→ 部分文本 → 多子串前缀 → 命中即执行；
   实测「现在打开远程桌面」在 **「现在打开远程」** 就已触发（dry，恰好一次）；
4. 影子模式（默认）只记录、不改写；破坏性意图默认拦截；
5. JSONL 遥测（decisions/rescue/pending/a_path）+ 有界存储（20B/行，可裁前缀）。

详见 `opencode_fastpath/README.md`；文件哈希与 k8s 变更见 `opencode_fastpath/OPENCODE_MANIFEST.json`。
