# Redis3D

RTX 5060 Laptop (sm_120) üzerinde 2 GB VRAM tavanıyla çalışan yerel 3D üretim motoru. Hedef: bulut 3D servislerine (Tripo, Meshy) cihazda çalışan bir alternatif. Dış API yok, model ağırlıkları indirilmiyor.

Durum: Faz 0/10 tamam. Donanım korumaları ve VRAM watchdog çalışıyor; üretim pipeline'ı henüz yok.

## Çalıştırma

```bash
pip install -r requirements.txt
python -m pytest tests/test_phase0_hardware.py -v -s
```

Gerçek çıktı (17 Kasım 2026, bu makine):

```text
GPU           : NVIDIA GeForce RTX 5060 Laptop GPU (sm_120)
Driver        : 617.14 | torch 2.11.0+cu128 | CUDA 12.8
AC adapter    : True (battery 48%)
Power limits  : enforced 115 W
BF16 matmul   : OK
FP8 scaled_mm : OK (rel_err=0.0375)
1.5 GB pass: peak=1545.1 MB, shared_spill=0.0 MB
2.1 GB correctly rejected: 2109.1 MB > 2048.0 MB hard cap
============================= 11 passed in 10.58s =============================
```

## Şu an çalışanlar

- `VRAMBudgetGuard(max_mb=2048)`: tahsis öncesi projection + PyTorch peak denetimi. 1.5 GB gerçek tahsis geçer, 2.1 GB reddedilir.
- Güç watchdog'u: NVML enforced limit + Windows AC adaptör durumu. Pilde veya 70 W altında uyarı üretir.
- Shared-RAM taşma denetimi: Windows `GPU Process Memory\Shared Usage` sayacından okunur; sözleşme 0 MB.
- UTF-8 + pathlib her yerde; Türkçe yerel ayar (`cp1254`) dosya adları testli.

## Gereksinimler

Windows 11, RTX 5060 Laptop (sm_120), Python 3.12, torch 2.11.0+cu128, pynvml.

## Sınırlar

- Yalnızca sm_120 doğrulandı; başka GPU'da probe hata verir.
- Mesh üretimi, eğitim ve arayüz sonraki fazlarda; bugün çalışan şey koruma katmanı.
- WDDM, NVML'den proses-başına VRAM'i gizler; PyTorch peak otoritedir.
- Triton kurulu değil; tüm çekirdekler saf PyTorch fallback ile yazılacak.

## Yol haritası

Faz kabul kapıları `roadfinal.md` içinde tanımlı. Her faz, gerçek GPU'da `%100 pytest` ve `peak VRAM <= 2048 MB` kanıtı olmadan kapanmaz.

<details><summary>Faz listesi</summary>

0. Donanım tanılama + VRAM watchdog — tamam
1. W-OVoxel² + CAD-Snap matematik çekirdeği
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