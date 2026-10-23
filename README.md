# Hermes Agent + Qwen3.8-27B di molab

Notebook [marimo](https://marimo.io) untuk menjalankan [Hermes Agent](https://hermes-agent.nousresearch.com) dengan Qwen3.8-27B (default versi abliterated FP8) di GPU gratis [molab](https://molab.marimo.io) (RTX Pro 6000 Blackwell, 96 GB).

**Buka di molab:** https://molab.marimo.io/github/Hanzamas/hermes-qwen-molab/blob/main/hermes_qwen_molab.py

## Isi

- vLLM 0.32.0 di venv terpisah, melayani model di `127.0.0.1:8000` (tidak dibuka ke luar)
- Pilihan model: `munekazu/Huihui-Qwen3.8-27B-abliterated-FP8` (default), `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (BF16), `orcarouter/Qwen3.8-27B-Uncensored-FP8` (gated), `Qwen/Qwen3.8-27B-FP8`
- Konteks 262K native, atau 512K / 1M lewat YaRN
- Reasoning dikunci `xhigh`, level tertinggi yang diterima chat template Qwen3.8
- Chat Hermes di dalam notebook (tool call dan output tampil langsung), log event Hermes, grafik GPU/KV/token per detik

## Cara pakai

1. Buka link molab di atas, lalu nyalakan GPU lewat tombol *notebook specs*.
2. Jalankan sel **Cek mesin**.
3. Klik **Install**, lalu **Start vLLM**, tunggu panel Server `READY`.
4. Klik **Sambungkan Hermes**, lalu ngobrol di panel Chat.

## Hermes TUI di terminal

Ada terminal xterm.js langsung di notebook (bagian *Hermes TUI di terminal*). Ia jalan di mesin molab lewat widget marimo (anywidget), jadi tanpa port, proxy, atau tunnel. `colab-xterm` tidak bisa dipakai di molab karena memakai `google.colab.kernel.proxyPort`.

1. Pastikan server `READY` dan **Sambungkan Hermes** sudah diklik (itu yang mengisi `~/.hermes/config.yaml`).
2. Disarankan: klik **Pasang tmux**. Sesi terminal tetap hidup walau tab ditutup atau sel dijalankan ulang. Lepas dengan `Ctrl-b` lalu `d`.
3. Klik tombol Hermes di bawah terminal, atau ketik:

   ```bash
   source ~/.bashrc && hermes chat --reasoning xhigh --yolo
   ```

4. Lanjutkan sesi dari chat notebook dengan `--resume <session_id>` (ada di log Hermes) atau `--resume latest`.

Ketikan diproses kernel yang sama dengan sel notebook, jadi saat sel panjang (Install) jalan, terminal ikut menunggu. xterm.js dimuat browser dari CDN (esm.sh, cadangan jsdelivr).

Isi workspace molab dihapus tiap notebook mati, jadi install dan download model diulang tiap sesi.
Jangan membuka port server ke luar (tunnel/proxy): aturan molab melarangnya.

Belum dites end-to-end di GPU molab. Widget terminalnya juga belum pernah jalan di molab.
