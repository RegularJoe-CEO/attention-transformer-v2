# Mac vs RunPod — which commands?

| You see this prompt | Where you are | Repo path |
|---------------------|---------------|-----------|
| `ericwaller@...` Mac | Your Mac | `~/Desktop/eRock/_audit_repos/attention-transformer-v2` |
| `root@bff7ccc1fa32` or `root@...:/#` | **RunPod pod** | `/workspace/attention-transformer-v2` |

## On the RunPod terminal (you are here if prompt is `root@bff7...`)

`scripts/pod-setup` and `~/Desktop/...` **do not exist** on the pod.

**Paste this block:**

```bash
cd /workspace
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_ARCH=90
# Create token at https://github.com/settings/tokens (repo scope), then:
export GITHUB_TOKEN='paste_ghp_token_here'
curl -fsSL https://raw.githubusercontent.com/RegularJoe-CEO/attention-transformer-v2/main/scripts/bootstrap_pod_terminal.sh -o /tmp/bootstrap.sh
bash /tmp/bootstrap.sh
```

If raw GitHub fails (private repo), clone by hand:

```bash
cd /workspace
export GITHUB_TOKEN='ghp_xxxx'
git clone https://x-access-token:${GITHUB_TOKEN}@github.com/RegularJoe-CEO/attention-transformer-v2.git
cd attention-transformer-v2
bash scripts/runpod_v2_test.sh
```

## On your Mac (sync Desktop code to pod)

```bash
cd ~/Desktop/eRock/_audit_repos/attention-transformer-v2
scripts/pod-setup    # paste RunPod SSH line once
scripts/pod          # rsync + GPU test
```

Mac pushes your edits; pod pulls or receives rsync.