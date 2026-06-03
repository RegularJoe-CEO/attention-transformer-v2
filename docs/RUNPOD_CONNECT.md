# RunPod Connect — what to copy (you cannot edit SSH on RunPod)

RunPod only lets you **copy** connection strings. Use this map:

| RunPod button | Use for `scripts/pod`? | Notes |
|---------------|------------------------|--------|
| **SSH over exposed TCP** | **YES — use this** | `root@IP -p PORT`. Best for rsync + long builds. |
| **SSH** (proxy) | Sometimes | `user@ssh.runpod.io`. Often needs RunPod account SSH key. |
| **Direct TCP ports** | No | For custom servers you expose (not shell). |
| **Port 8888 / Jupyter** | No | Web notebook — not for `cargo`/CUDA tests. |

## One-time setup (paste only)

```bash
cd ~/Desktop/eRock/_audit_repos/attention-transformer-v2
scripts/pod-setup
```

When prompted, paste the **entire** line from **SSH over exposed TCP**, e.g.:

```text
ssh root@203.0.113.50 -p 22154 -i ~/.ssh/id_ed25519
```

If RunPod shows a **different** `-i` path (pod-generated key), paste that exactly.

## Daily commands (after pod-setup)

```bash
scripts/pod status
scripts/pod          # full test
scripts/pod quick
scripts/pod shell    # same as manual SSH, uses saved settings
```

Show the exact SSH command we will run:

```bash
source scripts/runpod_ssh.sh && runpod_ssh_display
```

## If TCP line is not available

1. Copy the **SSH** proxy line instead into `pod-setup`.
2. Or add your Mac public key to RunPod **account** Settings → SSH Public Keys:

```bash
cat ~/.ssh/id_ed25519.pub
```

## 8888 / Jupyter

Open in browser for notebooks only. For the transformer engine, use **SSH over exposed TCP** + `scripts/pod`.