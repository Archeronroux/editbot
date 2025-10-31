# WhatsApp Group Member Count Editor Bot

Bot Telegram untuk mengedit angka "anggota" pada screenshot Info Grup WhatsApp secara otomatis, cepat, dan mulus menggunakan template digit per-platform.

## Fitur
- Mode: `/android`, `/iphone`, `/all` (auto-detect).
- Input: kirim screenshot dengan caption angka target (misal: `1234`).
- Output: gambar diedit dikirim sebagai dokumen (metadata & resolusi terjaga).
- Deteksi tema (light/dark) otomatis.
- Antrian per user: maks 5 pekerjaan, proses paralel, aman dari race condition.
- Pipeline: anchor-matching -> ROI -> inpaint -> render template -> blend -> simpan EXIF.

## Cara Jalankan
1. Python 3.11+ direkomendasikan.
2. Install deps:
   ```
   pip install -r req.txt
   ```
3. Buat file `.env`:
   ```
   BOT_TOKEN=123456:ABC-DEF...
   ```
4. Jalankan:
   ```
   python app.py
   ```

## Struktur Ringkas
```
.
├─ app.py
├─ cfg.py
├─ core/
│  ├─ process.py
│  ├─ detect.py
│  ├─ render.py
│  └─ utils.py
├─ tpl/
│  ├─ android/
│  │  ├─ light/anchors/anggota.png
│  │  ├─ light/anchors/anchor_mode.png
│  │  ├─ light/digits/{0..9,dot}.png
│  │  └─ dark/(anchors,digits)...
│  └─ iphone/
│     └─ light|dark/(anchors,digits)...
├─ out/
├─ .work/
├─ req.txt
└─ README.md
```

## Menyiapkan Template
- Letakkan digit `0-9` dan `dot.png` (RGBA) di `tpl/<mode>/<theme>/digits/`.
- Letakkan anchor `anggota.png` (cuplikan kata "anggota" sesuai tampilan) di `tpl/<mode>/<theme>/anchors/`.
- Opsi: `anchor_mode.png` untuk bantu auto-detect platform.
- Gunakan template resolusi menengah (tinggi digit ±120–160 px) agar scaling halus.

## Catatan Kinerja (VPS 1GB)
- Dependensi ringan (tanpa OCR berat). Hanya OpenCV + Pillow.
- Batas worker global `2` (dapat dinaikkan jika CPU memadai).
- Target waktu: 0.5–5 detik per gambar (bergantung resolusi & template scan), jauh di bawah 1 menit.

## Akurasi & Kemulusan
- Inpainting (TELEA) menghapus angka lama tanpa tepi keras.
- Digit dari template RGBA dirender dengan alpha blending & trimming kerning ringan.
- Posisi angka diratakan ke kanan dan diskalakan ke tinggi ROI agar konsisten.

## Roadmap opsional
- Tambah fallback OCR ringan untuk verifikasi.
- Tambah manifest per-template untuk ROI spesifik.
- Dukungan kirim sebagai foto (kompres Telegram) via command opsional.