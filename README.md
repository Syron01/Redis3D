# Redis3D

RTX 5060 Laptop (sm_120) üzerinde 2 GB VRAM tavanıyla çalışan yerel 3D üretim motoru. Hedef: bulut 3D servislerine (Tripo, Meshy) cihazda çalışan bir alternatif. Dış API yok, model ağırlığı indirilmiyor.

Durum: Faz 3/10 tamam. W-OVoxel², Topoloji Onarımı ve Lisans Denetimli .wov Veri Fabrikası kanıtlı; 41 testin 41'i yeşil.

## Çalıştırma

```bash
pip install -r requirements.txt
python -m pytest tests/ -v -s
```

Gerçek çıktı (17 Kasım 2026, bu makine, RTX 5060 Laptop):

```text
CAD lossless: chamfer=4.55e-08, cos_x=1.000000, cos_y=1.000000, edge_verts=231
Cloth lossless: chamfer=4.68e-08, cells=725636, tris=1344756
Shell lossless: chamfer=1.68e-06, outer/inner samples=18281/11719
Touching parts lossless: chamfer=6.60e-08, part0/part1 faces=83842/84666
Pruned sphere: tokens 151400->512 (295.70x), extract=49.8 ms, F=0.99997
Phase 2 gate: 50/50 manifold+watertight, mean=2.8 ms/mesh, worst=6.8 ms
WOV SLA: 2048-token asset compressed size: 125.05 KB (SLA: < 160.0 KB)
Phase 3 gate: 256 assets in 3.56s (72.0 assets/s, 16 CPU workers), shard avg 32.9 KB
============================= 41 passed in 53.56s =============================
```

## Şu an çalışanlar

- Alan-free Dual-Grid + Dual-Sheet çıkarıcı: SDF işaret değişimlerinden Hermite örnekleri, kesişimler 16 adım bisection ile bulunur.
- CAD-Snap: normal kovaryansı eigen-rank analizi; rank-2 keskin kenar ve rank-3 köşeler 3-düzlem kesişimine kilitlenir.
- Deterministik Topoloji Onarımı: yarım-kenar analizi, yapışık köşe ayırma, delik köprüleme, ters normal düzeltme (50/50 onarım, 2.8 ms).
- Lisans Denetimi: ticari allowlist (`MIT`, `Apache-2.0`, `CC0`, `Procedural-Owned`), kontaminasyon filtresi (`CC-BY-NC`, `GPL` %100 ret).
- 16-Worker Prosedürel Veri Motoru: 6 zengin kategori (CAD, Organik, Kumaş, İskelet, Kabuk, Montaj), 256 varlık / 3.56s.
- Zstd `.wov` v2 Codec: 64-bit Morton, PBR, iskelet ve intrinsics içeren binary format (125 KB, SLA < 160 KB).

## Gereksinimler

Windows 11, RTX 5060 Laptop (sm_120), Python 3.12, torch 2.11.0+cu128, pynvml, scipy, numpy, zstandard.

## Sınırlar

- Yalnızca sm_120 doğrulandı; başka GPU'da probe hata verir.
- Veri hattı ve derleyici hazır; üretici sinir ağları (Faz 4-5) sonraki adımlarda.

## Yol haritası

Faz kabul kapıları `roadfinal.md` içinde tanımlı. Her faz, gerçek GPU'da `%100 pytest` ve `peak VRAM <= 2048 MB` kanıtıyla ilerler.

<details><summary>Faz listesi</summary>

0. Donanım tanılama + VRAM watchdog — tamam
1. W-OVoxel² + CAD-Snap matematik çekirdeği — tamam
2. Deterministik topoloji onarımı — tamam
3. Lisans denetimi + .wov veri fabrikası — tamam
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
