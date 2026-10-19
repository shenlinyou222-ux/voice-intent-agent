# 启用 vLLM realtime 路由（opencode 部署的一部分）

> **OPENCODE** ｜ 2026-10-19 ｜ 依据：本仓库 `STREAMING_ENGINE_NOTES.md`
> —— vLLM 若不把架构指到 `Qwen3ASRRealtimeGeneration`，`/v1/realtime` 路由**不会注册**，
> WS 握手表现为 **HTTP 403**（Starlette 对未匹配 WS 路由的默认响应）。

## 做了什么

给 `default/vllm-asr` 的 args 末尾加一项：

```
--hf-overrides={"architectures": ["Qwen3ASRRealtimeGeneration"]}
```

同一个引擎继续提供 `/v1/audio/transcriptions`（离线）与 `/v1/realtime`（WS 流式）。

## 应用

```bash
# 在 WSL 里
kubectl patch deploy vllm-asr --type=json \
  --patch-file "/mnt/c/Users/user/Desktop/ds harness/voice_agent/opencode_fastpath/k8s/patch_args.json"
kubectl rollout status deploy/vllm-asr --timeout=300s
```

## 回滚

```bash
kubectl patch deploy vllm-asr --type=json \
  --patch-file "/mnt/c/Users/user/Desktop/ds harness/voice_agent/opencode_fastpath/k8s/rollback_args.json"
kubectl rollout status deploy/vllm-asr --timeout=300s
```

`rollback_args.json` 里是**打补丁前的原始 args 全量数组**（不含 `--hf-overrides`）。

## 验证（2026-10-19 实测）

| 项 | 结果 |
|---|---|
| `GET /v1/models` | `qwen3-asr` 正常 |
| WS `ws://172.20.71.101:30805/v1/realtime` | 握手成功、`session.created`、6 个 `transcription.delta` |
| 离线 `POST /v1/audio/transcriptions` | `{"text":"现在打开远程桌面。",...}` —— 未被破坏 |
| 端到端早火 | 部分文本「现在打开远程」即触发 `launch_app(远程桌面)`（dry） |
