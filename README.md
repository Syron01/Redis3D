# Redis3D

RTX 5060 Laptop (sm_120) üzerinde 2 GB VRAM tavanıyla çalışan yerel 3D üretim motoru. Hedef: bulut 3D servislerine (Tripo, Meshy) cihazda çalışan bir alternatif. Dış API yok, model ağırlığı indirilmiyor.

Durum: Faz 2/10 tamam. W-OVoxel² + CAD-Snap ve Deterministik Topoloji Onarımı gerçek GPU'da kanıtlı; 30 testin 30'u yeşil.

## Çalıştırma

```bash
pip install -r requirements.txt
python -m pytest tests/ -v -s
```

Gerçek çıktı (17 Kasım 2026, bu makine, RTX 5060 Laptop):

```text
CAD lossless: chamfer=4.55e-08, cos_x=1.000000, cos_y=1.000000, edge_verts=231
Cloth lossless: chamfer=4.69e-08, cells=725636, tris=1344756
Shell lossless: chamfer=1.68e-06, outer/inner samples=18283/11717
Touching parts lossless: chamfer=6.60e-08, part0/part1 faces=83842/84666
Twin spheres: chamfer=2.41e-06, dual-sheet cells=60
Pruned sphere: tokens 151400->512 (295.70x), extract=30.7 ms, F=0.99997
CAD prune: ratio=21.12x, sharp tokens preserved=1918
Phase 2 gate: 50/50 manifold+watertight, mean=2.5 ms/mesh, worst=6.8 ms, peak VRAM=9.2 MB
============================= 30 passed in 34.56s =============================
```

## Şu an çalışanlar

- Alan-free Dual-Grid + Dual-Sheet çıkarıcı: SDF işaret değişimlerinden Hermite örnekleri, kesişimler 16 adım bisection ile bulunur.
- CAD-Snap: normal kovaryansı eigen-rank analizi; rank-2 keskin kenar ve rank-3 köşeler 3-düzlem kesişimine kilitlenir.
- QEF fp64 Cholesky: rank-1 sayfalarda cond ~1e6; kapalı-form Cholesky ile sayısal kararlılık.
- Haar-lifting wavelet budama: mask-normalize ortalama; küre 295.7x, CAD 21.1x, F-Score@0.01 = 0.99997.
- Deterministik Topoloji Onarımı: yarım-kenar analizi, yapışık tepe noktası ayırma, delik köprüleme, ters normal düzeltme, Sym-Lock ikili ayna hizalama (50/50 kusursuz onarım, ortalama 2.5 ms).
- `VRAMBudgetGuard(max_mb=2048)`: 1545.1 MB geçer, 2109 MB bloklar, shared-RAM taşması 0 MB.

## Gereksinimler

Windows 11, RTX 5060 Laptop (sm_120), Python 3.12, torch 2.11.0+cu128, pynvml, scipy, numpy, matplotlib.

## Sınırlar

- Yalnızca sm_120 doğrulandı; başka GPU'da probe hata verir.
- Delik kapatma düz merkez fanı kullanır; geniş deliklerde eğrilik kopyalanmaz.
- Mesh çıkarımı ve onarım hazır; üretici model eğitimi (Faz 4-5) sonraki adımlarda.

## Yol haritası

Faz kabul kapıları `roadfinal.md` içinde tanımlı. Her faz, gerçek GPU'da `%100 pytest` ve `peak VRAM <= 2048 MB` kanıtıyla ilerler.

<details><summary>Faz listesi</summary>

0. Donanım tanılama + VRAM watchdog — tamam
1. W-OVoxel² + CAD-Snap matematik çekirdeği — tamam
2. Deterministik topoloji onarımı — tamam
3. Lisans denetimi + .wov veri fabrikası
4. PIXL-Intrinsic + SW-VAE
5. BiDelta-SSA + OT-MeanFlow-3D
6. NVFP4 + BiLoCo-SVDQuant
7. Consilience-3D + PATR-2
8. Retopo / UV / rig / print / export
9. Web stüdyo + API + Blender + MCP
10. Tier-1 eğitim + SLA benchmark raporu

</details>

## Lisans

Tescilli ticari yazılım; tüm hakları saklıdır. Temiz oda (clean-room) geliştirme, üçüncü taraf model ağırlığı yok.
