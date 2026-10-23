# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "altair",
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
    - Terminal VM ada di panel developer marimo: `Ctrl-J`, tab **Terminal** (atau ``Ctrl-` ``).
      Dari situ kamu juga bisa menjalankan `~/.local/bin/hermes` versi TUI.
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

    Selain chat di atas, Hermes versi TUI (tampilan terminal lengkap) bisa dijalankan di terminal bawaan marimo.
    `colab-xterm` tidak bisa dipakai di molab: paket itu butuh magic `%xterm` IPython dan proxy port Colab.

    1. Pastikan panel **Server** sudah `READY` dan tombol **Sambungkan Hermes** sudah diklik.
       Tombol itu yang mengisi `~/.hermes/config.yaml` supaya Hermes memakai vLLM lokal.
    2. Tekan `Ctrl-J` untuk membuka panel developer, lalu pilih tab **Terminal** (atau langsung ``Ctrl-` ``).
    3. Jalankan:

       ```bash
       source ~/.bashrc && hermes chat --reasoning xhigh --yolo
       ```

    4. Mau melanjutkan obrolan dari chat notebook? Ambil `session=` dari log Hermes di atas, lalu:

       ```bash
       hermes chat --reasoning xhigh --yolo --resume <session_id>
       ```

       Atau `--resume latest` untuk sesi terakhir.

    Keluar dari TUI dengan mengetik `/exit` (atau `/quit`).
    Server vLLM tetap jalan, jadi TUI dan chat notebook bisa dipakai bergantian.
    """)
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
