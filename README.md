# Redis3D

RTX 5060 Laptop (sm_120) üzerinde 2 GB VRAM tavanıyla çalışan yerel 3D üretim motoru. Hedef: bulut 3D servislerine (Tripo, Meshy) cihazda çalışan bir alternatif. Dış API yok, model ağırlığı indirilmiyor.

Durum: Faz 1/10 tamam. W-OVoxel² + CAD-Snap matematik çekirdeği ve mesh çıkarımı gerçek GPU'da kanıtlı; üretim pipeline'ı henüz yok.

## Çalıştırma

```bash
pip install -r requirements.txt
python -m pytest tests/ -v -s
```

Gerçek çıktı (17 Kasım 2026, bu makine, RTX 5060 Laptop):

```text
CAD lossless: chamfer=4.55e-08, cos_x=1.000000, cos_y=1.000000, edge_verts=231
Cloth lossless: chamfer=4.68e-08, cells=725636, tris=1344756
Shell lossless: chamfer=1.68e-06, outer/inner samples=18282/11718
Touching parts lossless: chamfer=6.60e-08, part0/part1 faces=83842/84666
Twin spheres: chamfer=2.41e-06, dual-sheet cells=60
Pruned sphere: tokens 151400->512 (295.70x), extract=73.6 ms, F=0.99997
CAD prune: ratio=21.12x, sharp tokens preserved=1918
============================= 22 passed in 46.80s =============================
```

## Şu an çalışanlar

- Alan-free Dual-Grid + Dual-Sheet çıkarıcı: SDF işaret değişimlerinden Hermite örnekleri, kesişimler gerçek SDF üzerinde 16 adım bisection ile bulunur (uç-değer enterpolasyonu kutu SDF'lerinde yanlı).
- CAD-Snap: normal kovaryansının eigen-rank analizi; rank-2 keskin kenarlar ve rank-3 köşeler tam 3-düzlem kesişimine sabitlenir, budanmaz.
- QEF fp64 Cholesky: rank-1 sayfalarda cond ~1e6; fp32 adjugate iptali köşe bileşenlerini bozduğu için kapalı-form Cholesky kullanılır.
- Haar-lifting wavelet budama: mask-normalize ortalamalar, rank-1 gruplar sıkıştırılır; küre 295.7x, CAD 21.1x, F-Score@0.01 = 0.99997.
- `VRAMBudgetGuard(max_mb=2048)`: Faz 1'in en ağır şekli (cloth, 725K hücre) 1353.6 MB peak, shared-RAM taşması 0 MB. 2.1 GB tahsis reddedilir.
- 64-bit Morton + searchsorted LUT, octahedral normal codec (roundtrip cos > 1-1e-6), Türkçe UTF-8 yol testi.

## Gereksinimler

Windows 11, RTX 5060 Laptop (sm_120), Python 3.12, torch 2.11.0+cu128, pynvml, scipy, numpy.

## Sınırlar

- Yalnızca sm_120 doğrulandı; başka GPU'da probe hata verir.
- Lossless kapılar analitik SDF + analitik normal ile ölçüldü; normalsiz SDF'lerde gradyan fallback'i h/16 stencil kullanır, keskin kenara h/16'den yakın örneklerde normal kirlenir.
- Mesh üretimi var ama topoloji onarımı, eğitim ve arayüz sonraki fazlarda.
- WDDM, NVML'den proses-başına VRAM'i gizler; PyTorch peak otoritedir. Triton kurulu değil; çekirdekler saf PyTorch fallback ile yazılacak.

## Yol haritası

Faz kabul kapıları `roadfinal.md` içinde tanımlı. Her faz, gerçek GPU'da `%100 pytest` ve `peak VRAM <= 2048 MB` kanıtı olmadan kapanmaz.

<details><summary>Faz listesi</summary>

0. Donanım tanılama + VRAM watchdog — tamam
1. W-OVoxel² + CAD-Snap matematik çekirdeği — tamam
2. Deterministik topoloji onarımı
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