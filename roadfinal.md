# AXION-3D PRO: 99%-Accuracy Ultra-Light Commercial 3D Foundation Engine (`roadfinal.md`)

> **CONSTITUTION FOR QWEN4-MAX AUTONOMOUS AGENT:**
> You are founding and engineering the core intellectual property of a commercial 3D Generative AI company (**AXION-3D PRO**) inside this directory.
> 1. **ZERO EXTERNAL 3D APIS:** Never call Tripo, Meshy, Rodin, Hunyuan API, Replicate, fal.ai, or any cloud 3D service. 100% local execution, 100% original clean-room IP.
> 2. **ZERO STUBS / ZERO MOCKS:** Every mathematical operator, CUDA/Triton kernel, neural layer, post-processor, and verification test MUST be genuinely implemented and numerically validated.
> 3. **STRICT SLA ENFORCEMENT:**
>    - **Peak Inference VRAM:** `1.0 GB – 2.0 GB` (Hard Fail Assert: `<= 2048 MB`, `0 MB` Shared GPU Memory spill).
>    - **End-to-End Generation Latency:** `1.0 – 3.0 seconds` on target RTX 5060 Laptop GPU.
>    - **Topological Reliability:** `>= 99.9%` manifold/watertight validity on final exported assets.
>    - **Geometric & Alignment Accuracy:** `>= 99%` F-Score@0.01 reconstruction accuracy on in-domain benchmarks + measurable Best-of-4 `Consilience-3D` + `PATR-2` pixel-alignment locking.

---

## 1. Exact Target Hardware & System Contract (From Telemetry)

Code and runtime auto-tuning MUST match this physical machine:
- **OS:** Windows 11 x64 (Turkish/Unicode locale-safe file paths via `pathlib.Path` + `utf-8` everywhere!). Support native Windows CUDA + optional WSL2 Linux Triton build.
- **CPU:** Intel Core i7-14650HX (16 Cores / 24 Logical Threads, up to 4.86 GHz, 30 MB L3, Virtualization Enabled).
  - *Policy:* Cap procedural worker pools at `16` threads to keep Windows + CUDA feeding threads responsive.
- **RAM:** 32.0 GB DDR5-5600 MT/s (`31.7 GB` usable, `~18.2 GB` free under active desktop).
  - *Policy:* Training uses pinned DDR5 CPU AdamW Offload so GPU optimizer states take `0 MB` VRAM.
- **Storage:** Crucial P310 1 TB NVMe SSD (`CT1000P310SSD8`, 932 GB formatted `C:`).
  - *Policy:* Compress all dataset shards using Zstandard-compressed `.wov` binary format; batch sequential writes to protect SSD endurance.
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (**Blackwell `GB206`, `sm_120`**):
  - 3328 CUDA Cores, 104 5th-Gen Tensor Cores, 8.0 GB GDDR7 (384 GB/s), 32 MiB L2 Cache.
  - **Idle Desktop Reality:** Windows DWM + Browser already consume `~1.4 GB` VRAM (`~6.6 GB` free).
  - **CRITICAL `sm_120` BLACKWELL KERNEL DIRECTIVE:** Unlike datacenter `sm_100` (B200 `tcgen05`), `sm_120` uses warp-level `mma.sync` and **exhibits known CUTLASS grouped/MoE FP4 kernel failures** unless strictly dense-compiled (`compute_120f`/`sm_120a`). Therefore, AXION-3D **strictly uses Dense `BiDelta-SSA` (NO MoE Routing!)** and Dense Block-16 `NVFP4` GEMM + Triton/PyTorch verified fallback!
  - **AC Power Watchdog:** Phase 0 inspects NVML `power_management_limit` and warns if running unplugged (`< 70W` battery throttle vs `100-115W` AC performance).

---

## 2. Competitive Gap Audit & How AXION-3D PRO Defeats 2026 Incumbents

| Pain Point in 2026 Market (Tripo / Meshy / Rodin / TRELLIS.2) | AXION-3D PRO Architectural Solution | Commercial Startup Advantage |
| :--- | :--- | :--- |
| **1. Baked Shadows/Highlights in Albedo** | `PIXL-Intrinsic` De-Lighting Pre-Filter + Disentangled `MIKU` PBR Latent Space | Assets relight physically inside Unreal Engine 5, Unity, Blender |
| **2. Alignment Drift & Extra Limbs** | **Best-of-4 `Consilience-3D` Consensus** + **`PATR-2` Pixel-Locked QEF** | Eliminates single-pass hallucinations & locks silhouettes to pixels in `< 0.6s` |
| **3. Melted CAD Edges / Missing Thin Parts** | $\mathcal{W}\text{-OVoxel}^2$ + **Rank-Analyzed `CAD-Snap`** + Adaptive $\ell=10$ ($1024^3$) | Razor-sharp $90^\circ$ mechanical bevels + thin blades/spokes in the *same* model as organics |
| **4. Messy Non-Manifold Topology & Bad UVs** | Native Dual-Contour Quads + **`4-RoSy` QuadRetopo** + **`MeshRepair`** + **`PartUV`** | **99.9%+** Watertight/Manifold SLA + clean animation loops + low-stretch UVs |
| **5. 12–29 GB VRAM / Cloud-Only Lock-in** | **Dense `NVFP4 + BiLoCo-SVDQuant`** + **Morton `BiDelta-SSA`** + **`OT-MeanFlow`** | **1.1–1.8 GB VRAM**, **1–3s latency**, 100% air-gapped enterprise privacy, `~$0.0001`/asset |
| **6. Legal / Regional License Trap** | Clean-room codebase + Automated **`license_audit.py`** provenance filter | 100% indemnifiable commercial IP for B2B studios & manufacturers |

---

## 3. Complete Mathematical Architecture

### 3.1 Representation: $\mathcal{W}\text{-OVoxel}^2\text{ + CAD-Snap}$
Domain $\Omega = [-1, 1]^3$, sparse multi-resolution octree depths $\ell \in \{5, 6, \dots, 10\}$ ($32^3$ to $1024^3$ virtual resolution).

1. **Field-Free Dual-Grid + Dual-Sheet Part Storage ($K \le 2$ sheets/cell):**
   Inside active leaf cell $c = (i,j,k,\ell)$, for each surface sheet $s \in \{0, 1\}$ with Hermite samples $\{(\mathbf{p}_m, \mathbf{n}_m)\}_{m \in \mathcal{S}_{c,s}}$:
   $$\mathbf{A}_{c,s} = \sum_{m} \mathbf{n}_m \mathbf{n}_m^\top + \lambda_{\text{reg}} \mathbf{I}_3, \qquad \mathbf{b}_{c,s} = \sum_{m} \mathbf{n}_m (\mathbf{n}_m^\top \mathbf{p}_m) + \lambda_{\text{reg}} \bar{\mathbf{p}}_{c,s}$$
   $$\mathbf{v}_{c,s}^* = \text{clamp}_{[0,1]^3}\!\left(\mathbf{A}_{c,s}^{-1}\mathbf{b}_{c,s}\right)$$
   - **`CAD-Snap` Rank Eigen-Analysis:** Compute eigenvalues $\sigma_1 \ge \sigma_2 \ge \sigma_3$ of unregularized $\sum \mathbf{n}_m\mathbf{n}_m^\top$:
     - $\sigma_2 / \sigma_1 < \epsilon_{\text{plane}}$ -> **Planar Sheet** (prunable up octree!)
     - $\sigma_3 / \sigma_1 < \epsilon_{\text{edge}} \le \sigma_2 / \sigma_1$ -> **Sharp Ridge/Crease Edge** (lock to $\ell \ge 9$, zero smoothing!)
     - $\sigma_3 / \sigma_1 \ge \epsilon_{\text{edge}}$ -> **Sharp Corner Vertex** (exact 3-plane intersection pinned to $\ell = 10$!)
   - **Topology & Material Bitfield per Sheet:**
     - `edge_mask` (3-bit primal crossing `(ex, ey, ez)` + 3-bit orientation/winding + 1-bit `open_sheet` + 1-bit `dual_sheet_present` + 6-bit `part_id` $\in [0, 63]$).
     - **PBR Tuple:** $\mathbf{m}_{c,s} = (\mathbf{a}_{\text{albedo}} \in [0,1]^3,\; r_{\text{rough}} \in [0,1],\; \mu_{\text{metal}} \in [0,1],\; \alpha_{\text{opacity}} \in [0,1],\; \varepsilon_{\text{emissive}} \in [0,1],\; \mathbf{o}_{\text{oct}} \in [-1,1]^2)$ where $\mathbf{o}_{\text{oct}}$ is lossless octahedral normal encoding $\mathbb{S}^2 \leftrightarrow [-1,1]^2$.

2. **Occupancy-Normalized 3D Haar-Lifting Wavelet Compression:**
   For each $2\times 2\times 2$ sibling group with occupancy mask $\omega \in \{0,1\}^8$ ($|\omega| \ge 1$):
   $$\mathbf{s}_{\ell-1} = \frac{1}{|\omega|} \sum_{q:\omega_q=1} \mathbf{f}_q, \qquad \mathbf{d}_{\ell-1, q} = \mathbf{f}_q - \mathbf{s}_{\ell-1}$$
   Prune children into parent super-voxel ONLY IF:
   $$\max_{q:\omega_q=1} \|\mathbf{d}_{\ell-1, q}^{\text{geom}}\|_\infty < \tau_{\text{geom}}(\ell) \;\;\wedge\;\; \max_{q:\omega_q=1} \|\mathbf{d}_{\ell-1, q}^{\text{PBR}}\|_\infty < \tau_{\text{pbr}}(\ell) \;\;\wedge\;\; \text{rank}(\mathbf{A}_{\text{parent}}) = 1$$
   *(Rank-2 sharp creases and Rank-3 corners are NEVER pruned away — guaranteeing zero rounded CAD bevels while cutting smooth-wall tokens by **5×–12×**!)*

3. **64-bit Morton Serialization:**
   $\text{key}(x,y,z,\ell) = (\ell \ll 60) \mid \text{dilate3}(x) \mid (\text{dilate3}(y)\ll 1) \mid (\text{dilate3}(z)\ll 2)$. All sparse lookups execute via vectorized CUDA/PyTorch `torch.searchsorted` on sorted Morton buffers — zero Python dicts, zero external `spconv`/`torch_scatter` build failures.

---

### 3.2 Hyperspherical Sparse Wavelet VAE (`SW-VAE`)
- **Fixes 1-Step Equator Bias (*Sphere Encoder 2* insight):** Apply **RMS-Norm Hyperspherical Shell Projection** to bottleneck latents $\mathbf{Z} \in \mathbb{R}^{L \times 24}$ ($L \in [512, 2048]$) so latent norms concentrate around $\sqrt{24}$ matching $\mathcal{N}(0, I_d)$ shells, preventing 1-step generative collapse!
- **Fixes Blurry Averaging:** Replace naive L2 averaging with **Hybrid QEF-Normal-Gradient + Multi-Scale Edge Loss**:
  $$\mathcal{L}_{\text{VAE}} = \lambda_v \|\hat{\mathbf{v}}-\mathbf{v}\|_1 + \lambda_{\text{QEF}} \sum_m (\mathbf{n}_m^\top(\hat{\mathbf{v}}-\mathbf{p}_m))^2 + \lambda_{\nabla} \|\nabla_{\text{oct}}\hat{\mathbf{v}} - \nabla_{\text{oct}}\mathbf{v}\|_1 + \lambda_e \text{FocalBCE}(\hat{e}, e) + \lambda_{\text{oct}} \|\hat{\mathbf{o}}_{\text{oct}} - \mathbf{o}_{\text{oct}}\|_2^2 + \lambda_{\text{pbr}} \|\hat{\mathbf{m}} - \mathbf{m}\|_{\text{Huber}} + \lambda_{\text{KL}} D_{\text{KL}}$$

---

### 3.3 `BiDelta-SSA` Dense Backbone + `OT-MeanFlow-3D`
- **Why Dense `BiDelta-SSA` (Zero MoE):** Avoids `sm_120` grouped GEMM FP4 bugs and stabilizes quantization recovery.
- **Block Layout (`3:1` Interleaving × 4 = 16–20 Blocks):**
  - 75% **Bidirectional Gated-DeltaNet** (`O(L)` linear state-space scan across forward Morton, reverse Morton, and axis-permuted Morton curves).
  - 25% **Spatial Sparse Window Attention (`SSA`)** over local 3D Morton windows (`W = 64` + `16` global register tokens) with **symmetric low-precision QAT awareness (`Attn-QAT` rule)**.
- **Sinkhorn-OT + JVP-Free Interval-Splitting MeanFlow + Baked CFG:**
  Forward interpolation $\mathbf{Z}_t = (1-t)\mathbf{Z}_0 + t\mathbf{Z}_1$ ($t=0$ noise, $t=1$ data) paired via 15-step mini-batch Sinkhorn-Knopp Optimal Transport $\pi^*$:
  - **Boundary Loss ($r=t$):** $\|\mathbf{u}_\theta(\mathbf{Z}_t, t, t, \mathbf{c}, w) - \mathbf{v}_{\text{cfg}}\|_2^2$
  - **Self-Consistent Split ($0 \le r < m < t \le 1$, EMA target $\bar{\theta}$):**
    $$\mathbf{u}_1 = \mathbf{u}_{\bar{\theta}}(\mathbf{Z}_r, r, m, \mathbf{c}, w), \qquad \hat{\mathbf{Z}}_m = \mathbf{Z}_r + (m - r)\mathbf{u}_1, \qquad \mathbf{u}_2 = \mathbf{u}_{\bar{\theta}}(\hat{\mathbf{Z}}_m, m, t, \mathbf{c}, w)$$
    $$\mathcal{L}_{\text{MF}} = \left\|\mathbf{u}_\theta(\mathbf{Z}_r, r, t, \mathbf{c}, w) - \text{stopgrad}\!\left(\frac{m-r}{t-r}\mathbf{u}_1 + \frac{t-m}{t-r}\mathbf{u}_2\right)\right\|_2^2$$
  - **Inference Speed:** 1-NFE (`~85 ms/candidate`), 2-NFE (`~170 ms/candidate`).

---

### 3.4 The 99%-Accuracy Closed-Loop Pipeline (`Consilience-3D` + `PIXL-Intrinsic` + `PATR-2` + `MeshRepair`)

To guarantee **consistent, verifiable accuracy** instead of single-sample neural gambling:

1. **`PIXL-Intrinsic` De-Lighting Filter (~18M params, `< 25 ms`):**
   Decomposes input image $I_{\text{raw}} \to (I_{\text{albedo}}, I_{\text{shading}}, I_{\text{normal}}, I_{\text{edges}})$. Prevents baked lighting artifacts inside 3D textures.
2. **Parallel $K=4$ Candidate Rollout (`~240 ms` batched 1-NFE!):**
   Samples 4 orthogonal/Sobol noise seeds $\{\mathbf{Z}_0^{(k)}\}_{k=1}^4$ in one micro-batch `B=4` (using `< 1.4 GB` VRAM!) and solves $\hat{\mathbf{Z}}_1^{(k)} = \mathbf{Z}_0^{(k)} + \mathbf{u}_\theta(\mathbf{Z}_0^{(k)}, 0, 1, \mathbf{c}, w)$.
3. **`Consilience-3D` Verifier-Free Consensus & Physical Ranking (`~110 ms`):**
   Coarsely decodes the 4 candidates into sparse point/dual-grid clouds and evaluates:
   $$S(k) = \alpha_1 \underbrace{\text{ConsensusChamfer}^{-1}(k, \{1..4\}\setminus k)}_{\text{Eliminates hallucinations/extra limbs}} + \alpha_2 \underbrace{\text{SilhouetteIoU}(k, I_{\text{ref}})}_{\text{Pixel outline lock}} + \alpha_3 \underbrace{\text{EdgeCosine}(k, I_{\text{edges}})}_{\text{Engraved detail match}} + \alpha_4 \underbrace{\text{TopoHealth}(k)}_{\text{Component/Euler validity}}$$
   Selects winning candidate $k^*$ (and optionally runs 1 fast polish step $0.5 \to 1.0$ on $k^*$).
4. **`PATR-2` Differentiable QEF & Intrinsic Texture Lock (`~350 ms`):**
   Runs 6 Adam steps on the active dual vertices $\mathbf{v}_{c,s}$ and leaf PBR tokens of $k^*$ using differentiable soft rasterization against $(I_{\text{albedo}}, I_{\text{normal}}, I_{\text{edges}}, \text{mask})$, clamping vertices inside $[0,1]^3$ and preserving `CAD-Snap` rank constraints!
5. **Deterministic Neuro-Symbolic Topology Repair (`~80 ms`):**
   Half-edge manifold check -> non-manifold pinch splitting -> small floating island pruning -> hole bridging -> consistent outward winding -> optional `Sym-Lock` bilateral snap. Guarantees **99.9%+ watertight/manifold validity**.

---

### 3.5 Hybrid `NVFP4 + BiLoCo-SVDQuant` + Sensitive-Layer Shielding (`sm_120` Safe)
Every dense Linear weight matrix in `Conditioner`, `Stage 1`, and `Stage 2` (excluding shielded layers) uses **Hybrid SVDQuant + BiLoCo**:
$$\mathbf{W} \approx \mathcal{Q}_{\text{NVFP4}}(\mathbf{W}_{\text{res}}) + \mathbf{U}_{16}\mathbf{V}_{16}^\top + \mathbf{B}_{\pm 1, 256} \mathbf{D}_{\text{scale}} \mathbf{C}_{\pm 1, 256}^\top$$
- **E2M1 NVFP4 + Group-16 FP8 E4M3 micro-scales:** `4.5` bits/param.
- **FP16 Top-16 SVD Adapter ($\mathbf{U}_{16}\mathbf{V}_{16}^\top$):** Absorbs extreme channel outliers.
- **Binary $\pm 1$ Rank-256 `BiLoCo` Spectral Tail ($\mathbf{B}_{\pm 1}, \mathbf{C}_{\pm 1}$):** Packs 256 spectral rank-1 correction bits inside `uint64` words using negligible RAM (`~2x` lower quantization RMSE than FP16 rank-32 alone!).
- **Sensitive-Layer Shielding (Mandatory for 99% Geometry Accuracy!):** `SW-VAE coordinate/QEF heads`, `PATR-2`, `positional embeddings`, `time/CFG embeddings`, and `LayerNorms` remain in **FP16/BF16** (costs `< 30 MB` total VRAM, eliminates surface quantization ripple!).

---

## 4. Exact Memory Budget Breakdown (Enforced by Automated Watchdogs)

### 4.1 Inference Budget (Strict Cap: `<= 2048 MB`, Target `1.20 – 1.75 GB`)

| Module | Precision | Params | Weights VRAM | Peak Act / Buffer |
| :--- | :--- | :---: | :---: | :---: |
| `PIXL-Intrinsic` De-Lighter | FP16 / FP8 | 18M | 24 MB | 55 MB |
| `Conditioner` (`NativeZero` or `LocalSigLIP`) | NVFP4 + BiLoCo | 65M–86M | 48 MB | 95 MB |
| `Stage 1` (`Octree-Coarse-Flow`) | NVFP4 + BiLoCo | 95M | 56 MB | 160 MB |
| `Stage 2` (`BiDelta-SSA` `OT-MeanFlow-3D`) | NVFP4 + BiLoCo | 310M | 190 MB | 395 MB (at `B=4` 1-NFE!) |
| `SW-VAE Decoder` (Coords shielded FP16) | Mixed FP8/FP16 | 55M | 96 MB | 280 MB |
| `Consilience-3D` + `PATR-2` + CUDA QEF | PyTorch/CUDA | — | — | 190 MB (reuses Stage 2 arena) |
| Post-Process (`QuadRetopo`/`PartUV`/`Axion-Rig`) | On-Demand Stream | 78M | 44 MB | 120 MB |
| PyTorch CUDA Context & Guard Margin | — | — | — | 210 MB |
| **ALL-RESIDENT PEAK (No Disk/RAM Swap)** | — | **~621M** | **~458 MB** | **~1.34 – 1.74 GB** |
| **ULTRA-LOW STREAMING MODE (DDR5 Pinned Pool)** | — | **~621M** | **~190 MB max** | **~860 – 990 MB** |

### 4.2 Local Training Budget (`RTX 5060 8 GB`, Free ~6.5 GB)
- **GPU Activations + Weights:** `BF16` + Full Block Activation Checkpointing (`<= 4.8 GB` VRAM).
- **Optimizer States:** Pinned `32 GB DDR5` CPU AdamW (`0 MB` GPU VRAM).
- **Zero Spill Guard:** Automatic `pynvml` monitor aborts & logs diagnostic warning if `shared_gpu_spill_mb > 0`.

---

## 5. Self-Audit: Known Limitations & Mandatory Mitigations (Startup Readiness)

1. **Open-World Zero-Shot Generalization vs. Compute Physics:**
   - *Gap:* A foundation model trained purely on 1 laptop GPU cannot ingest 5M+ assets in a weekend.
   - *Solution:* Three clean config tiers share the *exact same architecture*:
     - `configs/tier1_nano_5060.yaml` -> 10K–50K procedural CAD/organic/rigged assets generated locally on the 24-thread i7-14650HX; validates **99% F-Score reconstruction, 1.3 GB VRAM, 1-3s inference, and end-to-end product features** in hours.
     - `configs/tier2_studio_5060.yaml` -> Incremental overnight `--resume` training on 100K–300K locally compiled MIT/CC-BY meshes (`Objaverse-LVIS`, `ABC-1M`, `ABO`, `GSO`, `Rig-XL`) stored as ~15–45 GB `.wov` shards on the 1 TB NVMe.
     - `configs/tier3_cluster_scale.yaml` -> Identical training code scaled via PyTorch DDP/FSDP across rented multi-GPU nodes (500K–2M commercial-clean assets) to produce the flagship ~458 MB downloadable startup weights!
2. **Data License Contamination Risk:**
   - *Gap:* Commercial B2B clients reject models trained on `CC-BY-NC` or restricted open weights.
   - *Solution:* `axion3d/data/license_audit.py` enforces a strict manifest allowlist (`MIT`, `Apache-2.0`, `CC0`, `CC-BY-4.0`, `Procedural-Owned`) and drops any flagged asset prior to `.wov` shard compilation.
3. **Sub-Voxel Thin Geometry (< 1 voxel):**
   - *Gap:* Swords, glasses frames, spokes vanish in naive SDFs.
   - *Solution:* $\mathcal{W}\text{-OVoxel}^2$ uses field-free edge-crossing intersection flags + `open_sheet` bit + $\ell=10$ ($1024^3$) localized adaptive splitting on thin Rank-2/3 regions.

---

## 6. Complete Project Directory Tree

```text
axion3d_pro/
├── roadfinal.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── hw_rtx5060_laptop.yaml
│   ├── tier1_nano_5060.yaml
│   ├── tier2_studio_5060.yaml
│   └── tier3_cluster_scale.yaml
├── axion3d/
│   ├── __init__.py
│   ├── core/
│   │   ├── hardware_guard.py        # sm_120 check, AC power warning, VRAM <= 2048 MB assert
│   │   ├── morton.py                # 64-bit Morton encode/decode, radix sort, searchsorted LUT
│   │   └── telemetry.py             # Latency breakdown, NVML VRAM + shared spill audit
│   ├── math/
│   │   ├── qef.py                   # Tikhonov 3x3 analytical QEF + CAD-Snap eigen-rank solver
│   │   ├── octahedral.py            # Lossless S^2 <-> [-1,1]^2 octahedral normal codec
│   │   ├── wavelet_octree.py        # Mask-normalized 3D Haar-lifting + CAD-Snap pruning
│   │   ├── dual_contouring.py       # Field-free Dual-Grid + Dual-Sheet mesh extractor
│   │   ├── topology_repair.py       # Deterministic half-edge manifold & watertight repair
│   │   └── optimal_transport.py     # Sinkhorn-Knopp mini-batch OT pairing
│   ├── kernels/
│   │   ├── __init__.py              # Runtime dispatcher: sm_120 dense Triton/CUDA -> PyTorch
│   │   ├── nvfp4_biloco.py          # NVFP4 E2M1 block-16 + FP16 r16 + Binary ±1 r256 BiLoCo
│   │   ├── triton_nvfp4_linear.py   # Fused NVFP4+BiLoCo kernel (NO grouped MoE!)
│   │   ├── triton_sparse_attn.py    # Block-sparse spatial attention with Attn-QAT safety
│   │   └── cuda_qef.py              # Accelerated QEF & soft-rasterization operators
│   ├── data/
│   │   ├── license_audit.py         # Commercial-safe license allowlist validator
│   │   ├── wov_format.py            # Zstd-compressed .wov v2 binary shard codec
│   │   ├── procedural_gen.py        # 16-worker CAD/Organic/Cloth/Rigged synthetic factory
│   │   ├── mesh_to_wov.py           # Exact mesh+PBR -> W-OVoxel^2 compiler (< 0.8s/asset)
│   │   ├── dataset_ingestor.py      # Local .glb/.obj/.step/.ply -> verified .wov compiler
│   │   └── dataloader.py            # Pinned-DDR5 sparse collator
│   ├── models/
│   │   ├── intrinsic_delight.py     # PIXL-Intrinsic de-lighting + edge/normal estimator
│   │   ├── conditioner.py           # NativeZero ViT+BPE & optional local open-weights loader
│   │   ├── sw_vae.py                # RMS-Norm Hyperspherical Sparse Wavelet VAE
│   │   ├── bidelta_ssa.py           # Morton Bidirectional Gated-DeltaNet + Sparse Attention
│   │   ├── coarse_flow.py           # Stage 1: Depth-6 occupancy OT-MeanFlow
│   │   ├── latent_meanflow.py       # Stage 2: W-OVoxel^2 OT-MeanFlow + Baked CFG + Sym-Lock
│   │   ├── consilience_verifier.py  # K=4 Consensus + Silhouette/Edge/Topo scorer
│   │   └── pipeline.py              # Full AXION3DProPipeline (Resident & Streaming modes)
│   ├── refinement/
│   │   └── patr2.py                 # Differentiable silhouette+normal+albedo QEF optimizer
│   ├── training/
│   │   ├── cpu_adamw.py             # Pinned 32GB DDR5 CPU AdamW offload
│   │   ├── train_intrinsic.py       # Trains PIXL-Intrinsic on procedural multi-light renders
│   │   ├── train_vae.py             # Trains Hyperspherical SW-VAE
│   │   ├── train_coarse.py          # Trains Stage 1
│   │   ├── train_meanflow.py        # Trains Stage 2 (Sinkhorn-OT + JVP-Free Split MeanFlow)
│   │   ├── quantize_nvfp4_biloco.py # AWQ-weighted SVDQuant + BiLoCo calibration & QAT
│   │   └── train_rig.py             # Trains Axion-Rig skeleton & geodesic skinning
│   ├── postprocess/
│   │   ├── quad_retopo.py           # 4-RoSy cross-field + adaptive quad-dominant remesher
│   │   ├── part_uv.py               # Semantic part seams + LSCM conformal UV + skyline pack
│   │   ├── pbr_upscaler.py          # Tiled 4K/8K UV PBR super-resolution (< 1.4GB VRAM)
│   │   ├── auto_rig.py              # Skeleton Tree Transformer + Geodesic Skinning Weights
│   │   ├── print_doctor.py          # Watertight, wall-thickness >= 0.8mm check, STL/3MF export
│   │   └── exporters.py             # Validated GLB (KHR PBR + Skins), OBJ, USDZ, STL, 3MF
│   ├── ui/
│   │   ├── server.py                # FastAPI + WebSocket multi-tenant API (Auth + Queue + SLA)
│   │   └── static/                  # Interactive Three.js PBR + Wire/UV/Bone/Print Studio UI
│   └── bridges/
│       ├── blender_addon.py         # Blender 4.2+ live WebSocket bridge
│       └── mcp_server.py            # Local MCP tool server for AI agents
├── scripts/
│   ├── generate_startup_corpus.py   # Builds local .wov corpus
│   ├── benchmark_sla.py             # Measures 99%-accuracy SLAs vs baselines
│   └── demo_cli.py                  # Production CLI entry point
└── tests/
    ├── test_phase0_hardware.py
    ├── test_phase1_math_wov.py
    ├── test_phase2_topology_repair.py
    ├── test_phase3_data_license.py
    ├── test_phase4_intrinsic_vae.py
    ├── test_phase5_bidelta_meanflow.py
    ├── test_phase6_nvfp4_biloco.py
    ├── test_phase7_consilience_patr2.py
    ├── test_phase8_postprocess_99sla.py
    ├── test_phase9_ui_bridges.py
    └── test_phase10_end_to_end_sla.py
```

---

## 7. Phased Implementation Plan & Mandatory Acceptance Gates

> **Execution Rule:** Build strictly phase by phase. Run `pytest tests/test_phaseX_*.py -v -s` after each phase. Do NOT advance if any assertion fails.

### Phase 0: Hardware Diagnostics, Blackwell `sm_120` Probe & Guardrails
- Implement `hardware_guard.py` and `telemetry.py`.
- Detect `sm_120`, check AC power limit via `pynvml`, implement `VRAMBudgetGuard(max_mb=2048)` that tracks PyTorch peak VRAM AND NVML process VRAM to catch shared-RAM spill (`shared_spill == 0`).
- **Gate (`test_phase0_hardware.py`):** Verify BF16/FP8 ops on `sm_120`, confirm guard throws on `2.1 GB` and passes on `1.5 GB`.

### Phase 1: $\mathcal{W}\text{-OVoxel}^2\text{ + CAD-Snap}$ Mathematical Core
- Implement `morton.py`, `octahedral.py`, `qef.py` (with Eigen-rank Planar/Edge/Corner CAD-Snap), `wavelet_octree.py` (Mask-Normalized 3D Haar Lifting preserving Rank-2/3 sharp features), and `dual_contouring.py` (Dual-Grid + Dual-Sheet touching parts).
- **Gate (`test_phase1_math_wov.py`):**
  - Test on sharp boolean CAD block, open wavy cloth sheet, double-walled hollow shell, and two touching semantic parts (`part_id=0,1`).
  - **Pass:** Lossless mode `Chamfer < 1e-5`, sharp $90^\circ$ normal cosine `> 0.999`; Pruned mode achieves **`>= 5.0x` token compression** with `F-Score@0.01 >= 0.995` and extraction `< 120 ms`.

### Phase 2: Deterministic Topology Repair (`MeshRepair`)
- Implement `topology_repair.py`: half-edge structure construction, non-manifold vertex/edge splitting, tiny debris removal (`< 0.1%` volume), minimal hole bridging, consistent outward normal orientation, and optional bilateral `Sym-Lock`.
- **Gate (`test_phase2_topology_repair.py`):** Inject synthetic defects (flipped normals, open seams, non-manifold pinch vertices, floating dust triangles) into 50 meshes -> repair pipeline restores **100% manifold + watertight (`50/50`)** in `< 100 ms/mesh`.

### Phase 3: License-Audited 16-Worker Procedural & Local `.wov` Data Factory
- Implement `license_audit.py`, `wov_format.py`, `procedural_gen.py` (6 rich categories + ground-truth multi-light vs intrinsic albedo/normal renders + ground-truth rig skeletons!), `mesh_to_wov.py`, `dataset_ingestor.py`.
- **Gate (`test_phase3_data_license.py`):** Verify `license_audit` rejects mocked `CC-BY-NC` manifests, generate 256 multi-part PBR/rigged procedural assets across 16 CPU workers in `< 60s`, verify `.wov` shard size (`< 160 KB/asset`) and lossless decode.

### Phase 4: `PIXL-Intrinsic` De-Lighter & Hyperspherical `SW-VAE`
- Implement `intrinsic_delight.py` (decomposes shaded renders into lighting-free albedo, normal, edge map) and `sw_vae.py` (RMS-Norm hyperspherical bottleneck + FP16 shielded coordinate/QEF heads + Hybrid QEF-Normal-Gradient loss).
- Implement `cpu_adamw.py`, `train_intrinsic.py`, `train_vae.py`.
- **Gate (`test_phase4_intrinsic_vae.py`):**
  - `PIXL-Intrinsic` separates synthetic shaded renders to albedo `PSNR > 27 dB` in `< 25 ms`.
  - `SW-VAE` overfits 32 diverse `.wov` assets on RTX 5060 (`< 5.2 GB` training VRAM, `0 MB` shared spill) achieving **`F-Score@0.01 >= 0.990`** reconstruction and latent norm variance within `5%` of $\sqrt{24}$ hypersphere!

### Phase 5: `BiDelta-SSA` Backbone & `OT-MeanFlow-3D`
- Implement `conditioner.py` (`NativeZero` + optional `LocalSigLIP/DINO` interface), `bidelta_ssa.py` (strictly dense `BiDelta-SSA`, zero MoE), `optimal_transport.py`, `coarse_flow.py`, `latent_meanflow.py`, `train_coarse.py`, `train_meanflow.py`.
- **Gate (`test_phase5_bidelta_meanflow.py`):**
  - Linear $O(L)$ memory scaling verified up to $L=4096$.
  - JVP-Free Interval-Splitting MeanFlow converges so **1-NFE / 2-NFE sampling matches 32-step ODE (`< 4%` latent error)** and `Sym-Lock` guarantees `|x + mirror(x)| < 1e-5`.

### Phase 6: Blackwell `sm_120` `NVFP4 + BiLoCo-SVDQuant` Kernel Suite
- Implement `nvfp4_biloco.py`, `triton_nvfp4_linear.py`, `triton_sparse_attn.py`, `quantize_nvfp4_biloco.py`.
- Shield sensitive coordinate/QEF/norm/embed layers in FP16 while compressing 94% of backbone Linear layers to `NVFP4 E2M1 + FP16 r16 + Binary ±1 r256 BiLoCo`.
- **Gate (`test_phase6_nvfp4_biloco.py`):**
  - Quantized layer cosine similarity `>= 0.994` vs FP16 reference (`BiLoCo + SVDQuant` strictly beats plain `FP4` and plain `SVDQuant r16`).
  - Forward speed benchmark confirms `sm_120` execution without illegal instruction / grouped GEMM NaNs.

### Phase 7: `Consilience-3D` Best-of-4 Verifier & `PATR-2` Pixel-Lock Refiner
- Implement `consilience_verifier.py` (batched `B=4` 1-NFE candidate generation + Consensus/Silhouette/Edge/Topo scorer) and `patr2.py` (6-step differentiable rasterizer locking active dual vertices & PBR tokens while preserving `CAD-Snap` sharp ranks).
- **Gate (`test_phase7_consilience_patr2.py`):**
  - On controlled occluded/noisy test inputs, `Consilience-3D + PATR-2` improves Silhouette IoU by `>= +6%` over single-seed pass, rejects injected faulty candidates, executes in `<= 700 ms`, and stays strictly inside `<= 2.0 GB` VRAM.

### Phase 8: Commercial Post-Processing Suite (`99.9%` Topology & Game/Print Ready)
- Implement `quad_retopo.py` (`4-RoSy` quad extraction targeting user polycount), `part_uv.py` (`PartUV` seam cuts + LSCM + skyline pack), `pbr_upscaler.py` (tiled 4K/8K PBR maps), `auto_rig.py` (Skeleton Tree Transformer + Geodesic Skinning), `print_doctor.py` (wall thickness `>= 0.8 mm`, base flatten, `.stl/.3mf`), `exporters.py`.
- **Gate (`test_phase8_postprocess_99sla.py`):**
  - Across 20 generated assets: **100% (`20/20`) pass `is_watertight == True` & `non_manifold_edges == 0`**, **Quad ratio `>= 80%`**, **UV overlap `== 0`**, **GLB schema + skinning weights valid (`sum == 1.0`)**, **4K texture bake `< 1.5 GB` VRAM**.

### Phase 9: Enterprise Web Studio, Multi-Tenant API, Blender Bridge & MCP Server
- Implement `ui/server.py` (FastAPI + WebSocket + API key & concurrent job queue + real-time NVML VRAM gauge), `ui/static/` (Three.js PBR/Wireframe/UV/Bone/Print-Heatmap inspector), `bridges/blender_addon.py`, `bridges/mcp_server.py`.
- **Gate (`test_phase9_ui_bridges.py`):** End-to-end API/WebSocket test receives a valid textured/retopologized/rigged `.glb` in `< 3.5s` total HTTP turnaround with peak VRAM `<= 2.0 GB`.

### Phase 10: End-to-End SLA Proof, `Tier-1` Training & Commercial Benchmark Report
- Implement `scripts/generate_startup_corpus.py`, `scripts/benchmark_sla.py`, and `scripts/demo_cli.py`.
- Execute `tier1_nano_5060.yaml` full pipeline locally on this laptop (generate 10,000 procedural PBR+rigged `.wov` assets -> train `PIXL-Intrinsic`, `SW-VAE`, `Stage 1`, `Stage 2`, `Axion-Rig` -> calibrate `NVFP4+BiLoCo` -> run `benchmark_sla.py`).
- Generate `BENCHMARK_SLA_REPORT.md` proving:
  - Peak Inference VRAM (Resident & Streaming modes): **`<= 2048 MB`** (`0 MB` shared spill)
  - End-to-End Latency (`K=4` Consensus + 2-NFE + `PATR-2` + `MeshRepair`): **`<= 3.0 seconds`**
  - In-Domain Reconstruction & Silhouette Alignment (`F-Score@0.01`, `Silhouette IoU`, `Topology Validity`): **`>= 99%` SLA compliance**
  - Clean overnight resume scripts ready for `tier2_studio_5060.yaml` and `tier3_cluster_scale.yaml`!

---

## 8. Final Commercial CLI Contract

```bash
# Verify entire 10-phase system & 99% SLA thresholds on RTX 5060
pytest tests/ -v

# Generate a verified, de-lit, pixel-locked, quad-retopo, UV-packed, rigged GLB in <= 2GB VRAM & 1-3s
python scripts/demo_cli.py \
  --image concept.png \
  --prompt "precision armored mech helmet with sharp beveled visor" \
  --candidates 4 --steps 2 --patr-steps 6 --cad-snap --symmetry \
  --retopo-quads 16000 --auto-rig --tex-res 4096 --print-check \
  --max-vram-mb 1800 --out mech_helmet.glb

# Launch Commercial Studio Web App + Local Blender/MCP API
python -m axion3d.ui.server --host 127.0.0.1 --port 7860 --vram-mode resident
```