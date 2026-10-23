# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "altair",
#     "anywidget",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Hermes + Qwen3.8 di molab")


@app.cell
def _():
    import marimo as mo
    import json
    import os
    import re
    import secrets
    import signal
    import subprocess
    import time
    import urllib.request
    from pathlib import Path

    return Path, json, mo, os, re, secrets, signal, subprocess, time, urllib


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hermes Agent + Qwen3.8-27B (abliterated) di molab

    Semua jalan **di dalam notebook ini**: vLLM melayani Qwen3.8-27B di `127.0.0.1:8000`,
    Hermes Agent memakainya sebagai otak, dan obrolan, log, serta grafik GPU tampil di sel-sel bawah.
    Default-nya versi abliterated (uncensored) FP8 `munekazu/Huihui-Qwen3.8-27B-abliterated-FP8`,
    hasil kuantisasi FP8 per blok dari `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (vision dan MTP utuh).
    Reasoning dikunci di `xhigh`: chat template Qwen3.8 cuma menerima `xhigh`, `medium`, dan `low`.

    **Urutan pakai**

    1. Nyalakan GPU lewat tombol *notebook specs* di header molab.
    2. Jalankan sel **Cek mesin**.
    3. Pilih pengaturan, klik **Install**, tunggu sampai kedua log selesai.
    4. Klik **Start vLLM**, pantau panel **Server** sampai `READY`.
    5. Klik **Sambungkan Hermes**, lalu ngobrol di panel **Chat**.

    **Catatan**

    - Konteks 512K dan 1M memakai YaRN statis (`factor` 2.0 dan 4.0). Kata Qwen, ini bisa sedikit menurunkan kualitas
      di teks pendek. Di tes MRCR v2 komunitas (15 sampel per sel, jadi berisik), skor turun dari 0.833 di 256K native
      ke sekitar 0.74 di 512K maupun 1M. Pilih 262K kalau nggak butuh konteks panjang.
    - KV cache: 262K pakai BF16 (paling akurat di konteks native), 512K/1M pakai FP8 (di tes yang sama skornya setara BF16).
    - Memori: bobot FP8 sekitar 31 GB, BF16 sekitar 56 GB. KV FP8 sekitar 32 KiB per token
      (16 layer full-attention × 4 KV head × 256 dim × K+V), jadi 1M token butuh sekitar 31 GiB.
      FP8 + 1M muat di 96 GB. BF16 maksimal 512K.
    - Model `orcarouter/...-Uncensored-FP8` itu gated: terima syaratnya di Hugging Face, lalu simpan `HF_TOKEN` di panel Secrets.
    - Abliteration menghapus penolakan model. Kualitas reasoning dan tool calling bisa sedikit turun dibanding model asli.
    - Isi workspace dihapus saat notebook mati. vLLM, Hermes, dan bobot model harus diinstall ulang tiap sesi.
    - Jangan membuka port ini ke luar (tunnel, proxy). Aturan molab melarangnya.
    - Terminal xterm.js ada di bagian **Hermes TUI di terminal**, langsung di notebook ini dan jalan di mesin molab.
      Dari situ kamu bisa menjalankan Hermes versi TUI.
    """)
    return


@app.cell
def _(Path, secrets):
    WORK = Path("/tmp/hermes-qwen")
    WORK.mkdir(parents=True, exist_ok=True)
    VENV = Path.home() / "vllm-env"
    VLLM_BIN = VENV / "bin" / "vllm"
    VLLM_OK = VENV / ".install-ok"
    HERMES_BIN = Path.home() / ".local" / "bin" / "hermes"
    VLLM_LOG = WORK / "vllm.log"
    VLLM_PID = WORK / "vllm.pid"
    SERVER_INFO = WORK / "server.json"
    HERMES_EVENTS = WORK / "hermes_events.log"
    HERMES_STDERR = WORK / "hermes_stderr.log"
    # label -> (Hugging Face repo, download size in GB, gated, max context that fits 96 GB)
    MODELS = {
        "Huihui abliterated FP8 (publik)": ("munekazu/Huihui-Qwen3.8-27B-abliterated-FP8", 31, False, 1000000),
        "Huihui abliterated BF16 (publik)": ("huihui-ai/Huihui-Qwen3.8-27B-abliterated", 56, False, 524288),
        "orcarouter Uncensored FP8 (gated)": ("orcarouter/Qwen3.8-27B-Uncensored-FP8", 31, True, 1000000),
        "Qwen asli FP8": ("Qwen/Qwen3.8-27B-FP8", 31, False, 1000000),
    }
    # xhigh is the highest level Qwen3.8's chat template defines.
    REASONING = "xhigh"
    SERVED_NAME = "qwen3.8-27b"
    PORT = 8000
    BASE_URL = f"http://127.0.0.1:{PORT}/v1"

    # The key stays inside the sandbox; it is kept in /tmp so reruns reuse it.
    _key_file = WORK / "api_key"
    if not _key_file.exists():
        _key_file.write_text(secrets.token_hex(16))
    API_KEY = _key_file.read_text().strip()

    # Popen handles started from this kernel; poll() reaps exited children so they never linger as zombies.
    PROCS = {}
    return (
        PROCS,
        API_KEY,
        BASE_URL,
        HERMES_BIN,
        HERMES_EVENTS,
        HERMES_STDERR,
        MODELS,
        PORT,
        REASONING,
        SERVED_NAME,
        SERVER_INFO,
        VENV,
        VLLM_BIN,
        VLLM_LOG,
        VLLM_OK,
        VLLM_PID,
    )


@app.cell
def _(PROCS, VLLM_PID, mo, os, re, signal, subprocess, time, urllib):
    def run_live(cmd, title, env=None):
        """Run a shell command and stream its last lines into the cell output."""
        _env = {**os.environ, **(env or {})}
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_env,
            start_new_session=True,
        )
        lines = []
        last = 0.0
        try:
            for line in proc.stdout:
                lines.append(line.rstrip("\n"))
                if time.time() - last > 0.5:
                    mo.output.replace(
                        mo.vstack([mo.md(f"**{title}** (jalan...)"), mo.plain_text("\n".join(lines[-40:]))])
                    )
                    last = time.time()
            code = proc.wait()
        finally:
            # An interrupted cell must not leave the command running in the background.
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        status = "selesai" if code == 0 else f"GAGAL (exit {code})"
        mo.output.replace(
            mo.vstack([mo.md(f"**{title}**: {status}"), mo.plain_text("\n".join(lines[-40:]))])
        )
        return code

    def http_get(url, headers=None, timeout=2):
        """Return (status, body); status is None when the request fails."""
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except Exception as e:
            return None, str(e)

    def tail(path, n=30, max_bytes=65536):
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - max_bytes))
                return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-n:])
        except FileNotFoundError:
            return "(belum ada log)"

    _error_line = re.compile(r"\b\w*(Error|Exception)\b:|\bERROR\b|out of memory|CUDA error")
    # vLLM's own wrapper messages only say "see root cause above".
    _wrapper = ("Engine core initialization failed", "Frontend process failed", "EngineCore failed to start")

    def log_errors(path, n=15):
        """Last distinct error lines of a log, without vLLM's generic wrapper messages."""
        found = []
        for line in tail(path, 5000, max_bytes=1 << 20).splitlines():
            if _error_line.search(line) and not any(w in line for w in _wrapper) and line not in found:
                found.append(line)
        return "\n".join(found[-n:])

    def vllm_alive():
        """True while the vLLM API server is running."""
        proc = PROCS.get("vllm")
        if proc is not None:
            return proc.poll() is None
        # Started before a kernel restart: the group is no longer our child, so check it directly.
        try:
            pid = int(VLLM_PID.read_text())
        except (FileNotFoundError, ValueError):
            return False
        try:
            os.killpg(pid, 0)
            return True
        except ProcessLookupError:
            return False

    def vllm_kill(sig=signal.SIGTERM, wait=60):
        """Signal the whole vLLM process group (API server + EngineCore) and wait until it is gone."""
        try:
            pid = int(VLLM_PID.read_text())
        except (FileNotFoundError, ValueError):
            return None
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass
        for _ in range(wait):
            if "vllm" in PROCS:
                PROCS["vllm"].poll()
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(1)
        else:
            os.killpg(pid, signal.SIGKILL)
        VLLM_PID.unlink(missing_ok=True)
        PROCS.pop("vllm", None)
        return pid

    return http_get, log_errors, run_live, tail, vllm_alive, vllm_kill


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Cek mesin
    """)
    return


@app.cell
def _(mo, subprocess):
    def _q(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return (r.stdout or r.stderr).strip()

    _gpu = _q("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    _cc = _q("command -v cc gcc nvcc | tr '\\n' ' '")
    try:
        DRIVER_MAJOR = int(_gpu.split(",")[1].strip().split(".")[0])
    except (IndexError, ValueError):
        DRIVER_MAJOR = 0

    if DRIVER_MAJOR == 0:
        _note = mo.callout(
            mo.md("GPU belum terdeteksi. Nyalakan GPU di *notebook specs*, lalu jalankan ulang sel ini sebelum Install."),
            kind="danger",
        )
    elif DRIVER_MAJOR < 580:
        _note = mo.callout(mo.md(f"Driver R{DRIVER_MAJOR}: vLLM dipasang dari wheel CUDA 12.9."), kind="info")
    else:
        _note = mo.callout(mo.md(f"Driver R{DRIVER_MAJOR}: vLLM dipasang dari wheel default CUDA 13."), kind="info")

    mo.vstack(
        [
            mo.md(f"**GPU:** `{_gpu or 'tidak terdeteksi'}`"),
            mo.md(f"**User id:** `{_q('id -u')}`  ·  **uv:** `{_q('uv --version')}`"),
            mo.md(f"**Compiler (Triton butuh `cc`):** `{_cc or 'tidak ada'}`"),
            mo.plain_text(_q("df -h / /tmp $HOME 2>/dev/null")),
            _note,
        ]
    )
    return (DRIVER_MAJOR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Pengaturan dan tombol
    """)
    return


@app.cell
def _(MODELS, mo):
    model_choice = mo.ui.dropdown(
        options=list(MODELS),
        value="Huihui abliterated FP8 (publik)",
        label="Model",
    )
    ctx_len = mo.ui.dropdown(
        options={"1M (YaRN factor 4.0)": 1000000, "512K (YaRN factor 2.0)": 524288, "262K (native)": 262144},
        value="1M (YaRN factor 4.0)",
        label="Konteks",
    )
    use_mtp = mo.ui.switch(value=False, label="MTP speculative decoding (lebih cepat, eksperimental)")
    install_btn = mo.ui.run_button(label="Install vLLM + Hermes")
    start_btn = mo.ui.run_button(label="Start vLLM", kind="success")
    stop_btn = mo.ui.run_button(label="Stop vLLM", kind="danger")
    connect_btn = mo.ui.run_button(label="Sambungkan Hermes")

    mo.vstack(
        [
            mo.hstack([model_choice, ctx_len], justify="start"),
            use_mtp,
            mo.md("_Reasoning dikunci di `xhigh`. Pengaturan dipakai saat **Start vLLM**. Untuk mengganti, Stop lalu Start lagi, lalu Sambungkan Hermes lagi._"),
            mo.hstack([install_btn, start_btn, stop_btn, connect_btn], justify="start"),
        ]
    )
    return connect_btn, ctx_len, install_btn, model_choice, start_btn, stop_btn, use_mtp


@app.cell
def _(DRIVER_MAJOR, VENV, VLLM_BIN, VLLM_OK, install_btn, mo, run_live):
    mo.stop(not install_btn.value, mo.md("_Klik **Install** untuk memasang vLLM._"))
    mo.stop(
        DRIVER_MAJOR == 0,
        mo.callout(mo.md("GPU belum terdeteksi. Nyalakan GPU, jalankan ulang **Cek mesin**, baru Install."), kind="danger"),
    )

    # Triton needs cc for its C launcher; FlashInfer JIT needs ninja (molab ships neither by default).
    run_live(
        "(command -v cc && command -v ninja) >/dev/null && echo 'cc + ninja ada' "
        "|| (apt-get -qq update && apt-get -qq install -y gcc g++ ninja-build >/dev/null && echo 'gcc + ninja terpasang') "
        "|| echo 'gcc/ninja gagal dipasang'; "
        # The pip CUDA runtime ships only libcudart.so.13, but FlashInfer links with -lcudart from lib64.
        "for D in /usr/local/lib/python3.*/site-packages/nvidia/cu13 "
        f"{VENV}/lib/python3.*/site-packages/nvidia/cu13; do "
        "[ -d \"$D\" ] || continue; L=$(find \"$D\" -name 'libcudart.so.1*' | head -1); "
        "[ -n \"$L\" ] && mkdir -p \"$D/lib64\" && ln -sf \"$L\" \"$D/lib64/libcudart.so\" && echo \"libcudart.so -> $L\"; done",
        "Cek compiler C + ninja + libcudart",
    )
    if VLLM_OK.exists():
        mo.output.replace(mo.md(f"vLLM sudah terpasang di `{VENV}`."))
    else:
        # A separate venv keeps vLLM's torch away from the notebook kernel's packages.
        if DRIVER_MAJOR < 580:
            _pkg = (
                '"https://github.com/vllm-project/vllm/releases/download/v0.32.0/'
                'vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" --torch-backend=cu129'
            )
        else:
            _pkg = "vllm==0.32.0 --torch-backend=cu130"
        _rc = run_live(
            f"set -e; uv venv --allow-existing {VENV}; "
            f"uv pip install --python {VENV}/bin/python {_pkg}; "
            f"{VLLM_BIN} download-kernels || echo 'download-kernels dilewati'",
            "Install vLLM 0.32.0",
        )
        if _rc == 0:
            VLLM_OK.touch()
    return


@app.cell
def _(HERMES_BIN, install_btn, mo, run_live):
    mo.stop(not install_btn.value, mo.md("_Klik **Install** untuk memasang Hermes Agent._"))

    if HERMES_BIN.exists():
        mo.output.replace(mo.md(f"Hermes sudah terpasang di `{HERMES_BIN}`."))
    else:
        run_live(
            "curl -fsSL https://hermes-agent.nousresearch.com/install.sh"
            " | bash -s -- --non-interactive --skip-browser --skip-computer-use",
            "Install Hermes Agent",
        )
    return


@app.cell
def _(
    API_KEY,
    MODELS,
    PORT,
    PROCS,
    REASONING,
    SERVED_NAME,
    SERVER_INFO,
    VLLM_BIN,
    VLLM_LOG,
    VLLM_OK,
    VLLM_PID,
    ctx_len,
    http_get,
    json,
    mo,
    model_choice,
    os,
    signal,
    start_btn,
    subprocess,
    use_mtp,
    vllm_alive,
    vllm_kill,
):
    mo.stop(
        not start_btn.value,
        mo.md("_Server jalan dengan pengaturan lama. Stop lalu Start untuk menerapkan perubahan._")
        if vllm_alive()
        else mo.md("_Klik **Start vLLM** untuk menyalakan server model._"),
    )
    mo.stop(not VLLM_OK.exists(), mo.callout(mo.md("vLLM belum terpasang. Klik **Install** dulu."), kind="danger"))
    _repo, _size_gb, _gated, _max_ctx = MODELS[model_choice.value]
    mo.stop(
        _gated and not os.environ.get("HF_TOKEN"),
        mo.callout(
            mo.md(f"`{_repo}` itu gated. Terima syaratnya di Hugging Face, simpan `HF_TOKEN` di panel Secrets, lalu Start lagi."),
            kind="danger",
        ),
    )
    mo.stop(
        ctx_len.value > _max_ctx,
        mo.callout(mo.md(f"`{_repo}` maksimal {_max_ctx:,} token di 96 GB. Pilih konteks lebih kecil atau model FP8."), kind="danger"),
    )

    if vllm_alive() or http_get(f"http://127.0.0.1:{PORT}/health")[0] == 200:
        mo.output.replace(mo.md("Server sudah jalan atau masih loading. Klik **Stop vLLM** dulu kalau mau ganti pengaturan."))
    else:
        # A crashed run can leave EngineCore children holding VRAM; clear them before starting again.
        vllm_kill(signal.SIGKILL, wait=15)
        _cmd = [
            str(VLLM_BIN), "serve", _repo,
            "--served-model-name", SERVED_NAME,
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "--api-key", API_KEY,
            "--max-model-len", str(ctx_len.value),
            # BF16 KV scored best at native context; with YaRN, FP8 KV scored the same (MRCR v2, n=15).
            "--kv-cache-dtype", "fp8" if ctx_len.value > 262144 else "auto",
            "--gpu-memory-utilization", "0.92",
            "--max-num-seqs", "4",
            "--reasoning-parser", "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--default-chat-template-kwargs", json.dumps({"reasoning_effort": REASONING}),
        ]
        if ctx_len.value > 262144:
            # YaRN override from the Qwen3.8-27B model card: factor 4.0 for 1M, 2.0 for ~524K.
            _rope = {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "rope_type": "yarn",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "factor": 4.0 if ctx_len.value > 524288 else 2.0,
                "original_max_position_embeddings": 262144,
            }
            _cmd += ["--hf-overrides", json.dumps({"text_config": {"rope_parameters": _rope}})]
        if use_mtp.value:
            _cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3})]

        _log = open(VLLM_LOG, "w")
        # start_new_session keeps the server alive when a cell is interrupted and makes pgid == pid.
        _proc = subprocess.Popen(
            _cmd,
            stdin=subprocess.DEVNULL,
            stdout=_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                # FlashInfer JIT looks for ninja and nvcc on PATH; molab does not export the CUDA bin dir.
                "PATH": f"{VLLM_BIN.parent}:/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
                "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda"),
            },
        )
        VLLM_PID.write_text(str(_proc.pid))
        PROCS["vllm"] = _proc
        # The Connect cell reads this instead of the widgets, so changing a widget never resets the chat.
        SERVER_INFO.write_text(json.dumps({"model": _repo, "context": ctx_len.value}))
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(
                        f"vLLM dimulai (pid `{_proc.pid}`) dengan `{_repo}`. "
                        f"Download sekitar {_size_gb} GB dulu, lalu load. Pantau panel **Server**."
                    ),
                    mo.plain_text(" ".join(_cmd).replace(API_KEY, "<api-key>")),
                ]
            )
        )
    return


@app.cell
def _(mo, stop_btn, vllm_kill):
    mo.stop(not stop_btn.value)

    # Waits for the whole group to free the GPU before a new Start is allowed.
    _pid = vllm_kill()
    mo.output.replace(
        mo.md(f"vLLM (pid `{_pid}`) dihentikan.") if _pid else mo.md("Tidak ada server vLLM yang tercatat jalan.")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Server dan grafik
    """)
    return


@app.cell
def _(PORT, VLLM_LOG, VLLM_PID, http_get, log_errors, mo, tail, time, vllm_alive):
    # A mo.Thread keeps updating while a Hermes chat turn occupies the kernel.
    def _watch_server():
        me = mo.current_thread()
        while not me.should_exit:
            if http_get(f"http://127.0.0.1:{PORT}/health")[0] == 200:
                badge = mo.callout(mo.md("**READY**: server siap di `127.0.0.1:8000`."), kind="success")
            elif vllm_alive():
                badge = mo.callout(mo.md("**Loading**: lagi download atau load model."), kind="neutral")
            elif VLLM_PID.exists():
                badge = mo.callout(mo.md("**vLLM mati.** Lihat baris error di bawah, lalu Start lagi."), kind="danger")
            else:
                badge = mo.callout(mo.md("**Belum di-start.**"), kind="neutral")
            items = [badge]
            errors = log_errors(VLLM_LOG)
            if errors:
                items += [mo.md("**Baris error di log vLLM (penyebab aslinya biasanya di sini)**"), mo.plain_text(errors)]
            items += [mo.md("**Log vLLM (40 baris terakhir)**"), mo.plain_text(tail(VLLM_LOG, 40))]
            mo.output.replace(mo.vstack(items))
            time.sleep(3)

    mo.Thread(target=_watch_server, daemon=True).start()
    return


@app.cell
def _(PORT, http_get, mo, subprocess, time):
    import altair as alt

    def _gpu_sample():
        r = subprocess.run(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True,
        )
        try:
            util, mem = [float(x) for x in r.stdout.strip().split("\n")[0].split(",")]
            return util, mem / 1024
        except ValueError:
            return None, None

    def _vllm_metrics():
        status, body = http_get(f"http://127.0.0.1:{PORT}/metrics")
        out = {}
        if status != 200:
            return out
        for line in body.splitlines():
            if line.startswith("#") or " " not in line:
                continue
            name = line.split("{")[0].split(" ")[0]
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
        return out

    def _fmt(v, digits=1):
        return "-" if v is None else round(v, digits)

    def _chart(rows, keys, title, domain=None):
        data = [{"detik": r["detik"], "metrik": k, "nilai": r[k]} for r in rows for k in keys if r[k] is not None]
        y = alt.Y("nilai:Q", title=None, scale=alt.Scale(domain=domain) if domain else alt.Undefined)
        return (
            alt.Chart(alt.Data(values=data), title=title)
            .mark_line()
            .encode(x=alt.X("detik:Q", title="detik"), y=y, color=alt.Color("metrik:N", title=None))
            .properties(width=320, height=180)
        )

    def _watch_gpu():
        me = mo.current_thread()
        history = []
        t0 = time.time()
        prev_gen, prev_ts = None, None
        while not me.should_exit:
            now = time.time()
            util, mem = _gpu_sample()
            m = _vllm_metrics()
            kv = m.get("vllm:kv_cache_usage_perc", m.get("vllm:gpu_cache_usage_perc"))
            gen = m.get("vllm:generation_tokens_total")
            tps = None
            if gen is not None and prev_gen is not None and now > prev_ts:
                tps = max(0.0, (gen - prev_gen) / (now - prev_ts))
            prev_gen, prev_ts = gen, now
            history.append(
                {
                    "detik": round(now - t0, 1),
                    "GPU util %": util,
                    "VRAM GB": mem,
                    "KV cache %": None if kv is None else kv * 100,
                    "token/s": tps,
                }
            )
            del history[:-240]
            last = history[-1]
            mo.output.replace(
                mo.vstack(
                    [
                        mo.md(
                            f"**GPU** {_fmt(last['GPU util %'], 0)}%  ·  **VRAM** {_fmt(last['VRAM GB'])} GB  ·  "
                            f"**KV** {_fmt(last['KV cache %'])}%  ·  **token/s** {_fmt(last['token/s'])}  ·  "
                            f"**request** {_fmt(m.get('vllm:num_requests_running'), 0)}"
                        ),
                        mo.hstack(
                            [
                                _chart(history, ["GPU util %", "KV cache %"], "Pemakaian (%)", [0, 100]),
                                _chart(history, ["token/s"], "Kecepatan generate"),
                                _chart(history, ["VRAM GB"], "VRAM terpakai"),
                            ],
                            wrap=True,
                        ),
                    ]
                )
            )
            time.sleep(3)

    mo.Thread(target=_watch_gpu, daemon=True).start()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Hermes Agent
    """)
    return


@app.cell
def _(
    API_KEY,
    BASE_URL,
    HERMES_BIN,
    REASONING,
    SERVED_NAME,
    SERVER_INFO,
    connect_btn,
    http_get,
    json,
    mo,
    run_live,
):
    mo.stop(not connect_btn.value, mo.md("_Klik **Sambungkan Hermes** setelah server READY._"))
    mo.stop(not HERMES_BIN.exists(), mo.callout(mo.md("Hermes belum terpasang. Klik **Install** dulu."), kind="danger"))

    _st, _body = http_get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"})
    mo.stop(_st != 200, mo.callout(mo.md("Server vLLM belum READY."), kind="warn"))
    _ctx = json.loads(_body)["data"][0]["max_model_len"]
    try:
        _model = json.loads(SERVER_INFO.read_text())["model"]
    except (FileNotFoundError, KeyError, ValueError):
        _model = "?"

    _settings = {
        "model.provider": "custom",
        "model.base_url": BASE_URL,
        "model.default": SERVED_NAME,
        "model.api_key": API_KEY,
        "model.context_length": str(_ctx),
        # Hermes sends reasoning_effort on every request, and vLLM lets it override
        # --default-chat-template-kwargs, so the level is pinned here too.
        "agent.reasoning_effort": REASONING,
    }
    _rc = run_live(
        " && ".join(f"{HERMES_BIN} config set {k} '{v}'" for k, v in _settings.items())
        + f" && {HERMES_BIN} -z 'Balas satu kata: siap'",
        f"Konfigurasi Hermes ({_model}, konteks {_ctx}, reasoning {REASONING}) + tes",
    )
    mo.stop(
        _rc != 0,
        mo.callout(mo.md(f"Konfigurasi/tes Hermes gagal (exit {_rc}). Lihat log di atas dan stderr Hermes."), kind="danger"),
    )
    HERMES_READY = True
    return (HERMES_READY,)


@app.cell
def _(mo):
    # The switches mirror their values into OPTS. The chat cell reads only OPTS: referencing a switch
    # there would rebuild the chat on every toggle, and reading .value in its own cell is not allowed.
    OPTS = {"keep_session": True, "yolo": True}
    # Underscore names stay local to this cell, so they can never clash with names in other cells.
    _keep = mo.ui.switch(
        value=True,
        label="Lanjutkan sesi yang sama (--resume)",
        on_change=lambda v: OPTS.update(keep_session=v),
    )
    _yolo = mo.ui.switch(
        value=True,
        label="Tool tanpa konfirmasi (--yolo)",
        on_change=lambda v: OPTS.update(yolo=v),
    )
    mo.hstack([_keep, _yolo], justify="start")
    return (OPTS,)


@app.cell
def _(
    API_KEY,
    BASE_URL,
    HERMES_BIN,
    HERMES_EVENTS,
    HERMES_READY,
    HERMES_STDERR,
    OPTS,
    REASONING,
    SERVED_NAME,
    json,
    mo,
    os,
    signal,
    subprocess,
    time,
):
    mo.stop(not HERMES_READY)

    SESSION = {"id": None}

    def _log_event(text):
        with open(HERMES_EVENTS, "a") as f:
            f.write(time.strftime("%H:%M:%S ") + text + "\n")

    def hermes_model(messages, config):
        prompt = messages[-1].content
        cmd = [
            str(HERMES_BIN), "chat", "-q", prompt,
            "--format", "stream-json",
            "--provider", "custom",
            "--base-url", BASE_URL,
            "--api-key", API_KEY,
            "-m", SERVED_NAME,
            # Per-run override; the config key alone triggers an "unrecognized key" notice.
            "--reasoning", REASONING,
        ]
        if OPTS["yolo"]:
            cmd.append("--yolo")
        if OPTS["keep_session"] and SESSION["id"]:
            cmd += ["--resume", SESSION["id"]]
        _log_event(f"$ hermes chat -q {prompt[:120]!r}")
        with open(HERMES_STDERR, "a") as err:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, text=True,
                start_new_session=True,
                env={**os.environ, "PATH": f"{HERMES_BIN.parent}:{os.environ.get('PATH', '')}"},
            )
            got_result = False
            try:
                for line in proc.stdout:
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        _log_event(line.rstrip())
                        continue
                    kind = ev.get("type")
                    if kind == "system":
                        SESSION["id"] = ev.get("session_id") or SESSION["id"]
                        _log_event(f"[init] model={ev.get('model')} session={ev.get('session_id')}")
                    elif kind == "text":
                        yield ev.get("text", "")
                    elif kind == "tool_use":
                        args = json.dumps(ev.get("input"), ensure_ascii=False)[:600]
                        _log_event(f"[tool] {ev.get('name')} {args}")
                        yield f"\n\n**tool** `{ev.get('name')}`\n```json\n{args}\n```\n"
                    elif kind == "tool_result":
                        out = str(ev.get("output", ""))
                        _log_event(f"[result] {ev.get('name')} {ev.get('duration_ms')}ms error={ev.get('is_error')}")
                        yield f"\n```\n{out[:1500]}{' ...' if len(out) > 1500 else ''}\n```\n\n"
                    elif kind == "result":
                        got_result = True
                        SESSION["id"] = ev.get("session_id") or SESSION["id"]
                        tok = ev.get("tokens") or {}
                        _log_event(
                            f"[done] exit={ev.get('exit_code')} tokens={tok.get('total')} "
                            f"{ev.get('duration_ms')}ms {ev.get('error') or ''}"
                        )
                        if ev.get("exit_code") not in (0, None):
                            SESSION["id"] = None
                            yield f"\n\n_Hermes berhenti (exit {ev.get('exit_code')}): {ev.get('error') or 'lihat log'}_"
                proc.wait()
                if not got_result:
                    SESSION["id"] = None
                    yield f"\n\n_Hermes keluar tanpa event result (exit {proc.returncode}). Lihat stderr Hermes._"
            finally:
                # Stopping the reply in the chat closes this generator; take Hermes and its tools down with it.
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    _log_event("[stop] hermes dihentikan")

    chat = mo.ui.chat(
        hermes_model,
        prompts=[
            "Cek spesifikasi mesin ini pakai terminal, lalu ringkas.",
            "Bikin script Python yang ngitung bilangan prima < 1000, jalankan, tunjukkan hasilnya.",
        ],
    )
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    "`--yolo` membuat Hermes menjalankan perintah terminal tanpa minta izin. "
                    "Tanpa itu, tool yang butuh persetujuan bisa ditolak karena mode ini non-interaktif. "
                    "Perintahnya jalan di sandbox molab ini."
                ),
                kind="warn",
            ),
            chat,
        ]
    )
    return


@app.cell
def _(HERMES_EVENTS, HERMES_STDERR, mo, tail, time):
    def _watch_hermes():
        me = mo.current_thread()
        while not me.should_exit:
            mo.output.replace(
                mo.vstack(
                    [
                        mo.md("**Log Hermes (event stream-json)**"),
                        mo.plain_text(tail(HERMES_EVENTS, 60)),
                        mo.md("**stderr Hermes**"),
                        mo.plain_text(tail(HERMES_STDERR, 30)),
                    ]
                )
            )
            time.sleep(3)

    mo.Thread(target=_watch_hermes, daemon=True).start()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hermes TUI di terminal

    Terminal xterm.js langsung di notebook ini, jalan di mesin molab. Pesannya lewat widget marimo (anywidget),
    jadi tidak ada port yang dibuka dan tidak butuh proxy. `colab-xterm` beda jalur: ia menjalankan server lokal
    lalu menampilkannya lewat `google.colab.kernel.proxyPort`, fitur yang cuma ada di Colab.

    1. Pastikan panel **Server** sudah `READY` dan tombol **Sambungkan Hermes** sudah diklik.
       Tombol itu yang mengisi `~/.hermes/config.yaml` supaya Hermes memakai vLLM lokal.
    2. Disarankan: klik **Pasang tmux**. Shell jalan di dalam tmux, jadi sesinya tetap hidup walau tab browser
       ditutup atau sel terminal dijalankan ulang. Melepas sesi tanpa mematikannya: `Ctrl-b` lalu `d`.
    3. Klik tombol Hermes di bawah terminal, atau ketik sendiri:

       ```bash
       source ~/.bashrc && hermes chat --reasoning xhigh --yolo
       ```

       Tambah `--resume latest` untuk melanjutkan sesi terakhir (atau `--resume <session_id>` dari log Hermes di atas).

    Keluar dari TUI dengan `/exit` (atau `/quit`). Server vLLM tetap jalan, jadi TUI dan chat notebook bisa dipakai bergantian.

    Ketikan diproses kernel yang sama dengan sel notebook, jadi saat sel yang lama (Install, misalnya) sedang jalan,
    terminal ikut menunggu. xterm.js dimuat browser dari CDN (esm.sh, cadangan jsdelivr).
    """)
    return


@app.cell
def _(mo):
    tmux_btn = mo.ui.run_button(label="Pasang tmux")
    tmux_btn
    return (tmux_btn,)


@app.cell
def _(run_live, tmux_btn):
    from shutil import which as _which

    if tmux_btn.value and _which("tmux") is None:
        run_live(
            "apt-get -qq update && apt-get -qq install -y tmux >/dev/null && echo 'tmux terpasang'",
            "Pasang tmux",
        )
    TMUX = _which("tmux")
    return (TMUX,)


@app.cell
def _(mo, os, signal, subprocess, time):
    import fcntl
    import pty
    import select
    import struct
    import sys
    import termios
    import threading

    import anywidget
    import traitlets

    # A fresh interpreter makes the child a session leader that owns the pty as its controlling tty
    # (job control, Ctrl-C). Forking the multi-threaded kernel from Python code instead can deadlock.
    _LAUNCH = (
        "import os, sys, fcntl, termios\n"
        "os.setsid()\n"
        "fcntl.ioctl(0, termios.TIOCSCTTY, 0)\n"
        "os.execvpe(sys.argv[1], sys.argv[1:], os.environ)\n"
    )

    _CSS = r"""
    .hx-root { border: 1px solid #2a3441; border-radius: 8px; overflow: hidden; background: #0b0f14; }
    .hx-bar { display: flex; align-items: center; gap: 8px; padding: 4px 10px; background: #111821; color: #9aa9bb;
      font: 12px/1.5 ui-monospace, Menlo, Consolas, monospace; }
    .hx-dot { width: 8px; height: 8px; border-radius: 50%; background: #6b7785; flex: none; }
    .hx-root[data-state="ok"] .hx-dot { background: #3fb950; }
    .hx-root[data-state="warn"] .hx-dot { background: #d29922; }
    .hx-root[data-state="err"] .hx-dot { background: #f85149; }
    .hx-status { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .hx-spacer { flex: 1; }
    .hx-btn { font: inherit; color: #c9d4e0; background: #1b2531; border: 1px solid #2f3b4a; border-radius: 5px;
      padding: 1px 8px; cursor: pointer; }
    .hx-btn:hover { background: #243142; }
    .hx-box { padding: 6px 0 4px 8px; box-sizing: border-box; }
    .hx-box .xterm-viewport { background-color: transparent !important; }
    """

    _ESM = r"""
    const XTERM = "5.5.0";
    const FIT = "0.10.0";
    const CSS_URL = `https://cdn.jsdelivr.net/npm/@xterm/xterm@${XTERM}/css/xterm.css`;
    // esm.sh first; jsdelivr is the fallback when one CDN is blocked or down.
    const SOURCES = [
      ["esm.sh", (pkg, ver) => `https://esm.sh/${pkg}@${ver}`],
      ["jsdelivr", (pkg, ver) => `https://cdn.jsdelivr.net/npm/${pkg}@${ver}/+esm`],
    ];

    function loadXterm() {
      if (!globalThis.__hermesXterm) {
        const attempt = (async () => {
          const errors = [];
          for (const [name, url] of SOURCES) {
            try {
              const [x, f] = await Promise.all([
                import(url("@xterm/xterm", XTERM)),
                import(url("@xterm/addon-fit", FIT)),
              ]);
              return {
                Terminal: x.Terminal || x.default.Terminal,
                FitAddon: f.FitAddon || f.default.FitAddon,
              };
            } catch (e) {
              errors.push(name + ": " + ((e && e.message) || e));
            }
          }
          throw new Error("xterm.js gagal dimuat (" + errors.join("; ") + ")");
        })();
        attempt.catch(() => {
          globalThis.__hermesXterm = null;
        });
        globalThis.__hermesXterm = attempt;
      }
      return globalThis.__hermesXterm;
    }

    // xterm measures a character cell on open(), so the stylesheet has to be in before that.
    function loadCss(parent) {
      return new Promise((resolve) => {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = CSS_URL;
        link.onload = link.onerror = () => resolve();
        parent.appendChild(link);
        setTimeout(resolve, 4000);
      });
    }

    export default {
      async render({ model, el, signal }) {
        const root = document.createElement("div");
        root.className = "hx-root";
        const bar = document.createElement("div");
        bar.className = "hx-bar";
        const dot = document.createElement("span");
        dot.className = "hx-dot";
        const status = document.createElement("span");
        status.className = "hx-status";
        const spacer = document.createElement("span");
        spacer.className = "hx-spacer";
        const button = (label, title) => {
          const b = document.createElement("button");
          b.type = "button";
          b.className = "hx-btn";
          b.textContent = label;
          b.title = title;
          return b;
        };
        const clearBtn = button("Bersihkan", "Hapus isi layar dan scrollback");
        const freshBtn = button("Sesi baru", "Matikan sesi ini dan mulai dari awal");
        bar.append(dot, status, spacer, clearBtn, freshBtn);
        const box = document.createElement("div");
        box.className = "hx-box";
        box.style.height = model.get("height") + "px";
        root.append(bar, box);
        el.appendChild(root);

        const setStatus = (text, state) => {
          status.textContent = text;
          root.dataset.state = state;
        };
        setStatus("memuat xterm.js...", "load");

        let lib;
        try {
          [lib] = await Promise.all([loadXterm(), loadCss(root)]);
        } catch (e) {
          setStatus(String((e && e.message) || e), "err");
          return;
        }
        if (signal && signal.aborted) return;

        const term = new lib.Terminal({
          cursorBlink: true,
          scrollback: 5000,
          fontSize: 14,
          fontFamily: 'ui-monospace, "Cascadia Mono", "DejaVu Sans Mono", Menlo, Consolas, monospace',
          theme: { background: "#0b0f14", foreground: "#d5dbe3", cursor: "#d5dbe3", selectionBackground: "#2f4a6b" },
        });
        const fit = new lib.FitAddon();
        term.loadAddon(fit);
        term.open(box);
        const doFit = () => {
          try {
            fit.fit();
          } catch (_) {}
        };
        doFit();

        // Every message carries a per-view id and a counter: HTTP may deliver them out of order,
        // and the kernel puts them back in order before touching the pty.
        const view = Math.random().toString(36).slice(2);
        let seq = 0;
        const post = (msg) => model.send({ v: view, n: ++seq, ...msg });

        // Keystrokes within a few ms travel as one message.
        let pending = "";
        let timer = 0;
        const flush = () => {
          timer = 0;
          if (pending) {
            const d = pending;
            pending = "";
            post({ t: "in", d });
          }
        };

        let heard = false;
        const watchdog = setTimeout(() => {
          if (!heard) setStatus("kernel tidak menjawab (ada sel lain yang sedang jalan?)", "warn");
        }, 8000);

        model.on("msg:custom", (m, buffers) => {
          heard = true;
          if (!m) return;
          if (m.t === "out") {
            for (const b of buffers || []) term.write(new Uint8Array(b.buffer, b.byteOffset, b.byteLength));
          } else if (m.t === "reset") {
            term.reset();
          } else if (m.t === "started") {
            setStatus(m.tmux ? "tersambung (tmux)" : "tersambung (tanpa tmux)", "ok");
          } else if (m.t === "exit") {
            setStatus("proses selesai (kode " + (m.code == null ? "?" : m.code) + "). Klik Sesi baru.", "warn");
          } else if (m.t === "error") {
            setStatus(String(m.m), "err");
          }
        });

        term.onData((d) => {
          pending += d;
          if (!timer) timer = setTimeout(flush, 5);
        });
        term.onBinary((d) => {
          flush();
          post({ t: "bin", d });
        });
        term.attachCustomKeyEventHandler((e) => {
          if (e.type !== "keydown" || !e.ctrlKey || e.altKey || e.metaKey) return true;
          if (e.code === "KeyC" && term.hasSelection()) {
            // With text selected, Ctrl-C copies instead of sending SIGINT.
            navigator.clipboard?.writeText(term.getSelection()).catch(() => {});
            e.preventDefault();
            return false;
          }
          // Let the browser paste into xterm's textarea instead of sending a literal ^V.
          if (e.code === "KeyV") return false;
          return true;
        });
        // Keep typing in the terminal from reaching the notebook's own shortcuts.
        for (const type of ["keydown", "keyup", "keypress"]) {
          box.addEventListener(type, (e) => e.stopPropagation());
        }

        clearBtn.onclick = () => {
          term.clear();
          term.focus();
        };
        freshBtn.onclick = () => {
          flush();
          post({ t: "restart" });
          term.focus();
        };

        let raf = 0;
        const ro = new ResizeObserver(() => {
          cancelAnimationFrame(raf);
          raf = requestAnimationFrame(doFit);
        });
        ro.observe(box);

        post({ t: "ready", cols: term.cols, rows: term.rows });
        setStatus("menyambung...", "load");
        term.onResize(({ cols, rows }) => {
          flush();
          post({ t: "size", cols, rows });
        });

        return () => {
          clearTimeout(timer);
          clearTimeout(watchdog);
          cancelAnimationFrame(raf);
          ro.disconnect();
          term.dispose();
        };
      },
    };
    """

    class PtySession:
        """A shell on a pty, bridged to the browser through `emit(content, data)`."""

        RING = 256 * 1024

        def __init__(self, argv, env, emit, session=None, cwd=None):
            self.argv = argv
            self.env = env
            self.emit = emit
            self.session = session  # tmux session name; None for a bare shell
            self.cwd = cwd
            self._lock = threading.RLock()
            self._wake = threading.Event()
            self._proc = None
            self._master = None
            self._size = (24, 80)
            self._wbuf = bytearray()
            self._out = bytearray()
            self._out_t0 = 0.0
            self._ring = bytearray()
            self._views = {}
            self._active = None
            self._attach = False
            self._restart = False
            self._resize = False

        @property
        def alive(self):
            return self._master is not None

        def write(self, text):
            """Type text into the shell. Returns False when there is no shell to type into."""
            return self._queue(text.encode("utf-8", "replace"))

        def _queue(self, data):
            with self._lock:
                if self._master is None or len(self._wbuf) + len(data) > 4 << 20:
                    return False
                self._wbuf += data
            self._wake.set()
            return True

        # Browser -> kernel. Runs on the kernel's message thread, so it only records work for run().
        def handle(self, content):
            v, n = content.get("v"), content.get("n")
            if not isinstance(v, str) or not isinstance(n, int):
                return
            with self._lock:
                st = self._views.get(v)
                if st is None:
                    if len(self._views) >= 8:
                        self._views.pop(next(iter(self._views)))
                    st = self._views[v] = {"next": 1, "stash": {}, "gap": 0.0}
                if n < st["next"]:
                    return
                st["stash"][n] = content
                self._drain(v, st)

        def _drain(self, v, st):
            while st["next"] in st["stash"]:
                self._apply(v, st["stash"].pop(st["next"]))
                st["next"] += 1
            if not st["stash"]:
                st["gap"] = 0.0
            elif not st["gap"]:
                st["gap"] = time.monotonic()

        def _skip_gaps(self):
            # A message that never arrived must not freeze the terminal: after 1 s, skip past the gap.
            now = time.monotonic()
            with self._lock:
                for v, st in self._views.items():
                    if st["gap"] and now - st["gap"] > 1.0:
                        st["next"] = min(st["stash"])
                        st["gap"] = 0.0
                        self._drain(v, st)

        def _apply(self, v, c):
            t = c.get("t")
            if t == "ready":
                self._active = v
                self._set_size(c.get("rows"), c.get("cols"))
                self._attach = True
            elif v != self._active:
                return
            elif t == "in" and isinstance(c.get("d"), str):
                self._queue(c["d"].encode("utf-8", "replace"))
            elif t == "bin" and isinstance(c.get("d"), str):
                self._queue(c["d"].encode("latin-1", "replace"))
            elif t == "size":
                self._set_size(c.get("rows"), c.get("cols"))
            elif t == "restart":
                self._restart = True
            self._wake.set()

        def _set_size(self, rows, cols):
            if isinstance(rows, int) and isinstance(cols, int) and rows > 0 and cols > 0:
                self._size = (min(rows, 500), min(cols, 1000))
                self._resize = True

        # Kernel -> browser, and everything that touches the pty. Runs on one thread, so the fd has one owner.
        def run(self, should_exit):
            errors = 0
            try:
                while not should_exit():
                    try:
                        self._tick()
                        errors = 0
                    except Exception as e:
                        errors += 1
                        self.emit({"t": "error", "m": f"{type(e).__name__}: {e}"}, None)
                        if errors >= 5:
                            break
                        time.sleep(0.5)
            finally:
                self._kill()

        def _tick(self):
            self._skip_gaps()
            if self._restart:
                self._do_restart()
            if self._attach:
                self._do_attach()
            if self._master is None:
                self._wake.wait(0.1)
                self._wake.clear()
                return
            if self._resize:
                self._resize = False
                self._winsz(*self._size)
            self._pump()

        def _do_attach(self):
            self._attach = False
            rows, cols = self._size
            if self._proc is None:
                self._spawn(rows, cols)
                return
            # A new view of a running shell: it starts blank, so bring it up to date.
            self.emit({"t": "reset"}, None)
            if self.session is None:
                ring = bytes(self._ring)
                for i in range(0, len(ring), 32768):
                    self.emit({"t": "out"}, ring[i : i + 32768])
            self.emit({"t": "started", "tmux": self.session is not None}, None)
            self._redraw(rows, cols)

        def _do_restart(self):
            self._restart = False
            self._kill()
            if self.session:
                subprocess.run(
                    ["tmux", "kill-session", "-t", self.session],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    env=self.env,
                    timeout=10,
                    check=False,
                )
            self._ring.clear()
            self.emit({"t": "reset"}, None)
            self._attach = True

        def _spawn(self, rows, cols):
            master, slave = pty.openpty()
            try:
                fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
                proc = subprocess.Popen(
                    [sys.executable, "-c", _LAUNCH, *self.argv],
                    stdin=slave,
                    stdout=slave,
                    stderr=slave,
                    env=self.env,
                    cwd=self.cwd,
                    close_fds=True,
                )
            except Exception as e:
                os.close(master)
                os.close(slave)
                self.emit({"t": "error", "m": f"gagal memulai shell: {e}"}, None)
                return
            # With the parent's copy closed, the master sees EIO as soon as the child side is gone.
            os.close(slave)
            os.set_blocking(master, False)
            self._proc, self._master = proc, master
            self._ring.clear()
            self.emit({"t": "started", "tmux": self.session is not None}, None)

        def _winsz(self, rows, cols):
            if self._master is None:
                return
            try:
                fcntl.ioctl(self._master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
            except OSError:
                pass

        def _redraw(self, rows, cols):
            # Full-screen programs only repaint when the size changes, so nudge it and put it back.
            self._winsz(rows, max(cols - 1, 1))
            time.sleep(0.05)
            self._winsz(rows, cols)

        def _pump(self):
            fd = self._master
            with self._lock:
                want_write = bool(self._wbuf)
            # poll() instead of select(): the kernel may hold more than 1024 open files.
            poller = select.poll()
            poller.register(fd, select.POLLIN | (select.POLLOUT if want_write else 0))
            events = poller.poll(10)
            flags = events[0][1] if events else 0
            if flags & select.POLLOUT:
                with self._lock:
                    chunk = bytes(self._wbuf[:4096])
                try:
                    n = os.write(fd, chunk)
                except BlockingIOError:
                    n = 0
                except OSError:
                    return self._reap()
                with self._lock:
                    del self._wbuf[:n]
            if flags & (select.POLLIN | select.POLLHUP | select.POLLERR | select.POLLNVAL):
                try:
                    data = os.read(fd, 65536)
                except BlockingIOError:
                    data = None
                except OSError:
                    data = b""
                if data == b"":
                    return self._reap()
                if data:
                    if not self._out:
                        self._out_t0 = time.monotonic()
                    self._out += data
            # Output is sent when the pty goes quiet, or after 30 ms / 64 KB of a steady stream.
            self._flush(force=not (flags & select.POLLIN))

        def _flush(self, force):
            if not self._out:
                return
            if force or len(self._out) >= 65536 or time.monotonic() - self._out_t0 >= 0.03:
                data = bytes(self._out)
                self._out.clear()
                self._ring += data
                del self._ring[: max(0, len(self._ring) - self.RING)]
                self.emit({"t": "out"}, data)

        def _reap(self):
            self._flush(True)
            code = None
            if self._proc is not None:
                try:
                    code = self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            self._kill()
            self.emit({"t": "exit", "code": code}, None)

        def _kill(self):
            proc, master = self._proc, self._master
            self._proc = self._master = None
            with self._lock:
                self._wbuf.clear()
            self._out.clear()
            if master is not None:
                try:
                    os.close(master)
                except OSError:
                    pass
            if proc is None:
                return
            # The child called setsid(), so its pid is also its process group. For tmux this only detaches the client.
            for sig in (signal.SIGHUP, signal.SIGKILL):
                if proc.poll() is not None:
                    break
                try:
                    os.killpg(proc.pid, sig)
                except OSError:
                    break
                try:
                    proc.wait(2)
                except subprocess.TimeoutExpired:
                    continue

    class XTerm(anywidget.AnyWidget):
        _esm = _ESM
        _css = _CSS
        height = traitlets.Int(540).tag(sync=True)
        tmux = traitlets.Bool(False).tag(sync=True)

        def __init__(self, argv, env, session=None, cwd=None, **kwargs):
            super().__init__(tmux=session is not None, **kwargs)
            self.shell = PtySession(argv, env, self._emit, session=session, cwd=cwd)
            self.on_msg(self._on_msg)

        def _emit(self, content, data=None):
            try:
                self.send(content, buffers=None if data is None else [data])
            except Exception:
                pass

        def _on_msg(self, _widget, content, _buffers):
            if isinstance(content, dict):
                self.shell.handle(content)

        def start(self):
            """Run the pty loop in a mo.Thread, so it stops when the cell that called this is re-run or deleted."""

            def _loop():
                me = mo.current_thread()
                self.shell.run(lambda: me.should_exit)

            mo.Thread(target=_loop, daemon=True).start()

    return (
        PtySession,
        XTerm,
        anywidget,
        fcntl,
        pty,
        select,
        struct,
        sys,
        termios,
        threading,
        traitlets,
    )


@app.cell
def _(Path, TMUX, XTerm, mo, os):
    # The shell gets the kernel's environment minus its Python venv, plus ~/.local/bin so `hermes` resolves.
    _env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP")}
    _venv = _env.pop("VIRTUAL_ENV", None)
    _path = [p for p in _env.get("PATH", "").split(os.pathsep) if p and p != f"{_venv}/bin"]
    _env.update(
        PATH=os.pathsep.join([str(Path.home() / ".local" / "bin"), *_path]),
        TERM="xterm-256color",
        COLORTERM="truecolor",
        # The molab terminal prints setlocale warnings for en_US.UTF-8; C.UTF-8 always exists.
        LANG="C.UTF-8",
        LC_ALL="C.UTF-8",
    )
    if TMUX:
        # The second command after `;` keeps Esc snappy in TUIs (tmux waits 500 ms by default).
        _argv = [TMUX, "new-session", "-A", "-s", "hermes", "bash -i", ";", "set-option", "-s", "escape-time", "10"]
        _session = "hermes"
        _note = (
            "Shell jalan di tmux (sesi `hermes`): tetap hidup walau tab ditutup atau sel ini dijalankan ulang. "
            "Lepas dengan `Ctrl-b` lalu `d`."
        )
    else:
        _argv = ["bash", "-i"]
        _session = None
        _note = "tmux belum terpasang, jadi shell ini mati kalau sel ini dijalankan ulang. Klik **Pasang tmux** di atas."

    term_widget = XTerm(_argv, _env, session=_session, cwd=str(Path.home()))
    term_ui = mo.ui.anywidget(term_widget)
    term_widget.start()
    mo.vstack([mo.md(_note), term_ui])
    return term_ui, term_widget


@app.cell
def _(mo):
    hermes_btn = mo.ui.run_button(label="Ketik: hermes (sesi baru)")
    hermes_resume_btn = mo.ui.run_button(label="Ketik: hermes --resume latest")
    mo.hstack([hermes_btn, hermes_resume_btn], justify="start")
    return hermes_btn, hermes_resume_btn


@app.cell
def _(REASONING, hermes_btn, hermes_resume_btn, mo, term_widget):
    mo.stop(not (hermes_btn.value or hermes_resume_btn.value))
    _cmd = f"source ~/.bashrc && hermes chat --reasoning {REASONING} --yolo"
    if hermes_resume_btn.value:
        _cmd += " --resume latest"
    if term_widget.shell.write(_cmd + "\r"):
        mo.output.replace(
            mo.md("Perintah diketik di terminal. Tombol ini cuma mengetik, jadi pastikan terminal lagi di prompt bash.")
        )
    else:
        mo.output.replace(mo.callout(mo.md("Terminal belum tersambung. Tunggu statusnya `tersambung`."), kind="warn"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Shell cepat
    """)
    return


@app.cell
def _(mo):
    shell_cmd = mo.ui.text_area(value="nvidia-smi && ls ~/.hermes", label="Perintah bash", full_width=True)
    shell_btn = mo.ui.run_button(label="Jalankan")
    mo.vstack([shell_cmd, shell_btn])
    return shell_btn, shell_cmd


@app.cell
def _(mo, run_live, shell_btn, shell_cmd):
    mo.stop(not shell_btn.value)
    run_live(shell_cmd.value, "Shell")
    return


if __name__ == "__main__":
    app.run()
