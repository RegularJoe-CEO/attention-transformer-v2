# RunPod workflow (fast iterations)

Repo on GitHub: **https://github.com/RegularJoe-CEO/attention-transformer-v2** (private — use a PAT or SSH key on the pod).

## CPU vs GPU on the same repo

| Command | Lane | Needs GPU |
|---------|------|-----------|
| `cargo test --release` | CPU AUDIT | No |
| `cargo run --example production_demo` | CPU AUDIT | No |
| `cargo build --features cuda` | TRADE build | No (needs `nvcc` to compile kernels) |
| `cargo run --features cuda --example cuda_verify` | TRADE run | **Yes** |

Your Mac proved AUDIT. The pod proves TRADE.

---

## First time on a new pod (web terminal — paste once)

```bash
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_ARCH=90
export INSTALL_DIR=/workspace/attention-transformer-v2

# Private repo: use a GitHub PAT (Settings → Developer settings → PAT, repo scope)
# export GITHUB_TOKEN=ghp_xxxx
git clone https://${GITHUB_TOKEN:+${GITHUB_TOKEN}@}github.com/RegularJoe-CEO/attention-transformer-v2.git "$INSTALL_DIR"
cd "$INSTALL_DIR"
bash scripts/runpod_v2_test.sh
```

If clone fails (private repo), use **SSH clone** after adding your pod’s public key to GitHub, or use **rsync from Mac** below.

---

## Your RunPod SSH (preconfigured)

Edit `scripts/runpod.env` if the pod changes. Default:

- Host: `wcjtnrd5obyie7-64411782@ssh.runpod.io`
- Key: `~/.ssh/id_ed25519`

**Commands (from repo root) — no exports needed:**

```bash
cd ~/Desktop/eRock/_audit_repos/attention-transformer-v2
scripts/pod status    # test SSH + GPU name
scripts/pod           # sync + full GPU test
scripts/pod quick     # sync + cuda_verify only
scripts/pod sync      # rsync only
scripts/pod save "fix cuda bench"
scripts/pod shell     # SSH into pod
```

If `scripts/pod status` says `Permission denied (publickey)`, add your Mac pubkey in RunPod:
**Settings → SSH Public Keys** → paste output of `cat ~/.ssh/id_ed25519.pub`

---

## Fastest loop for many iterations (recommended)

### A) Git push / pull (simple, good with commits)

**Mac (after edits):**
```bash
cd ~/Desktop/eRock/_audit_repos/attention-transformer-v2
git add -A && git commit -m "wip" && git push
```

**Pod:**
```bash
cd /workspace/attention-transformer-v2
git pull
export PATH="/usr/local/cuda/bin:$PATH" CUDA_ARCH=90
bash scripts/runpod_v2_test.sh
```

Only re-run the examples you care about for speed:
```bash
cargo run --release --features cuda --example cuda_verify
cargo run --release --features cuda --example cuda_bench -- 200 1024 1024 16
```

### B) rsync from Mac (fastest without committing)

1. RunPod → your pod → **Connect** → copy **SSH over exposed TCP**, e.g. `root@203.0.113.50 -p 22154`

2. **Mac:**
```bash
export RUNPOD_SSH='root@YOUR_IP -p YOUR_PORT'
cd ~/Desktop/eRock/_audit_repos/attention-transformer-v2
bash scripts/sync_and_test_runpod.sh
```

This syncs `~/Desktop/eRock/_audit_repos/attention-transformer-v2` → pod and runs the full test script. Repeat the same command after every local edit (no `git push` needed).

---

## One-shot fresh deploy (wipe + clone)

```bash
export REPO_URL=https://github.com/RegularJoe-CEO/attention-transformer-v2.git
export INSTALL_DIR=/workspace/attention-transformer-v2
export BRANCH=main
bash -c "$(curl -fsSL https://raw.githubusercontent.com/RegularJoe-CEO/attention-transformer-v2/main/scripts/runpod_fresh_deploy_v2.sh)"
```

(Use after `runpod_fresh_deploy_v2.sh` is on `main` — or clone locally on pod as above.)

---

## What to copy back

From the pod log `runpod_v2_test_*.log`:

- `production_demo` receipt lines
- `cuda_verify` max diff (should be ~0)
- `cuda_bench` ms/layer and TFLOPS
- Any `runpod_quant_gate` PASS/FAIL