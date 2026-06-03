# RunPod — one paste block

**New terminal?** `cargo` is not on PATH until you **source** (do not run `pod_env.sh` as a command).

```bash
cd /workspace/attention-transformer-v2
git pull
source scripts/pod_env.sh
bash scripts/gpu_quick.sh
```

Attention-only bench:

```bash
cargo run --release --features cuda --example cuda_bench -- 200 1024 1024 16
```

Note the space: `--example cuda_bench` (not `--examplecuda_bench`).