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

Energy + power (auto `cd`, `source`, dmon):

```bash
bash /workspace/attention-transformer-v2/scripts/energy_bench_pod.sh
```

Or manually (must **not** run from `/`):

```bash
cd /workspace/attention-transformer-v2
source scripts/pod_env.sh
nvidia-smi dmon -s p -d 1 > /tmp/dmon.log 2>&1 &
DMON=$!
cargo run --release --features cuda --example cuda_bench -- 200 1024 1024 16
kill $DMON 2>/dev/null || true
cargo run --release --example energy_sweep > energy.csv
```