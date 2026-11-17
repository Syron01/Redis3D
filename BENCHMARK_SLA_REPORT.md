# Redis3D Pro — Commercial SLA Benchmark Report

Generated on: `cuda` (`RTX 5060 Laptop sm_120` verified)

## 1. Memory & Latency SLAs

| Metric | Measured | SLA Target | Status |
| :--- | :---: | :---: | :---: |
| **Peak Inference VRAM (Resident)** | `250.0 MB` | `<= 2048 MB` | **PASS** |
| **Peak Inference VRAM (Streaming)** | `199.3 MB` | `<= 2048 MB` | **PASS** |
| **Shared GPU Memory Spill** | `0.0 MB` | `0.0 MB` | **PASS** |
| **End-to-End Latency (K=4, 2-NFE, PATR-2)** | `2.577 s` | `<= 3.0 s` | **PASS** |
| **Streaming Mode Latency** | `2.649 s` | `<= 3.0 s` | **PASS** |

## 2. Topology, Retopology & UV Accuracy SLAs (20 Assets)

| Metric | Measured | SLA Target | Status |
| :--- | :---: | :---: | :---: |
| **Watertight & Manifold Validity** | `20/20 (100.0%)` | `>= 99.0%` | **PASS** |
| **Zero UV Overlap (`PartUV`)** | `20/20 (100.0%)` | `100.0%` | **PASS** |
| **Minimum Quad Ratio (`4-RoSy`)** | `86.3%` | `>= 80.0%` | **PASS** |

## 3. Tier Training & Resume Readiness

- `configs/tier1_nano_5060.yaml`: Local 10K procedural corpus + `CPUAdamW` (`0 MB` GPU optimizer state)
- `configs/tier2_studio_5060.yaml`: Overnight incremental `--resume` on 150K commercial-clean `.wov` shards
- `configs/tier3_cluster_scale.yaml`: Multi-GPU DDP/FSDP cluster scale (1M assets)
