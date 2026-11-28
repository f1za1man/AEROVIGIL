"""Step K, plan M13: screenshots of the offline demo dashboard.

Starts the Streamlit app headless on a spare port, opens it in headless Microsoft Edge
over the DevTools protocol, waits a real 25 s for every chart to draw (Edge's own
--virtual-time-budget screenshot catches the page half-drawn), and saves a full-page
PNG of the default view. Skipped, with a message, where Edge, the websockets package or
the demo data (data/processed/demo, built by `.\\make.ps1 demo-data`) is missing. The
owner records the video separately.
"""

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from aerovigil.figures import style as S

ME = "src/aerovigil/figures/demo_figs.py"
APP = S.ROOT / "app" / "app.py"
DATA = S.ROOT / "data" / "processed" / "demo" / "scenarios.parquet"
EDGE = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]  # fmt: skip
APP_PORT, CDP_PORT = 8599, 9339
WIDTH, HEIGHT, WAIT_S = 1500, 4200, 25.0


def edge():
    found = next((p for p in EDGE if p.is_file()), None)
    return found or (Path(shutil.which("msedge")) if shutil.which("msedge") else None)


def wait_for(url, seconds):
    for _ in range(int(seconds * 5)):
        try:
            return urllib.request.urlopen(url, timeout=2).read()
        except Exception:  # noqa: BLE001  not up yet
            time.sleep(0.2)
    raise SystemExit(f"{url} did not answer within {seconds:g} s")


async def capture(ws_url, url, out):
    import websockets

    async with websockets.connect(ws_url, max_size=None) as c:
        n = 0

        async def call(method, **params):
            nonlocal n
            n += 1
            await c.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                m = json.loads(await c.recv())
                if m.get("id") == n:
                    return m.get("result", {})

        await call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT,
                   deviceScaleFactor=1, mobile=False)  # fmt: skip
        await call("Page.navigate", url=url)
        await asyncio.sleep(WAIT_S)
        r = await call("Page.captureScreenshot", format="png")
        out.write_bytes(base64.b64decode(r["data"]))


def m13_dashboard():
    browser = edge()
    try:
        import websockets  # noqa: F401
    except ImportError:
        browser = None
    if browser is None or not DATA.is_file():
        print("  M13 skipped: needs Microsoft Edge, websockets and the demo data")
        return None
    S.OUT.mkdir(parents=True, exist_ok=True)
    out = S.OUT / "M13_dashboard.png"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(APP),
                            "--server.headless", "true", "--server.port", str(APP_PORT),
                            "--browser.gatherUsageStats", "false"],
                           cwd=S.ROOT, env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)  # fmt: skip
    profile = tempfile.mkdtemp(prefix="av_edge_")
    br = None
    try:
        wait_for(f"http://localhost:{APP_PORT}/_stcore/health", 90)
        br = subprocess.Popen([str(browser), "--headless=new", "--disable-gpu",
                               f"--remote-debugging-port={CDP_PORT}",
                               f"--user-data-dir={profile}",
                               f"--window-size={WIDTH},{HEIGHT}", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
        tabs = json.loads(wait_for(f"http://127.0.0.1:{CDP_PORT}/json", 30))
        ws = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        asyncio.run(capture(ws, f"http://localhost:{APP_PORT}", out))
    finally:
        for p in (br, app):
            if p is not None:
                p.terminate()
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
        shutil.rmtree(profile, ignore_errors=True)
    S.REGISTRY.append({"file": out.name,
                       "caption": "The offline demo dashboard (default view): flight, "
                                  "attack and detector controls, the path and offset, "
                                  "the detector comparison, the out-drift grid, the "
                                  "signal charts and the alarm explanation.",
                       "data": ["data/processed/demo/*.parquet", "app/app.py"],
                       "script": ME})  # fmt: skip
    print(f"  {out.name}", flush=True)
    return out


def make():
    m13_dashboard()
