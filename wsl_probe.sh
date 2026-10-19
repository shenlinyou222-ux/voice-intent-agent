#!/bin/bash
OUT=/mnt/c/Users/user/Desktop/ds\ harness/voice_agent/wslinfo.txt
exec > "$OUT" 2>&1
echo "=== whoami/host ==="; whoami; hostname; uname -r
echo "=== k3s present? ==="; which k3s kubectl 2>/dev/null; ls -la /etc/rancher/k3s/ 2>/dev/null | head
echo "=== systemd? ==="; ps -p 1 -o comm= 2>/dev/null; systemctl is-active k3s 2>&1 | head -2
echo "=== k3s kubectl pods ==="; k3s kubectl get pods -A 2>&1 | head -25
echo "=== k3s kubectl svc ==="; k3s kubectl get svc -A 2>&1 | head -20
echo "=== k3s kubectl deploy ==="; k3s kubectl get deploy -A 2>&1 | head -20
echo "=== manifests ==="; ls -la /var/lib/rancher/k3s/server/manifests/ 2>&1 | head -15
echo "=== images (containerd) ==="; k3s ctr images ls -q 2>/dev/null | head -30
echo "=== model files (qwen/asr/gguf/mmproj/safetensors) ==="
find / -xdev -maxdepth 7 \( -iname "*qwen*asr*" -o -iname "*.gguf" -o -iname "mmproj*" \) -printf "%s\t%p\n" 2>/dev/null | sort -rn | head -25
echo "=== candidate model dirs (size) ==="
du -sh /models /data /opt/models /root/models /home/*/models /mnt/e/models /mnt/c/models 2>/dev/null | head
echo "=== servers installed ==="
for b in vllm llama-server llama-cli python3 docker nerdctl; do printf "%-14s " "$b"; which $b 2>/dev/null || echo "-"; done
echo "=== listening ports ==="; ss -ltnp 2>/dev/null | head -20
echo "=== GPU in wsl? ==="; ls -la /dev/dxg 2>/dev/null; nvidia-smi -L 2>&1 | head -3
echo "=== DONE ==="
