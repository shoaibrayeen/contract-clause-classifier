"""Record a GIF of the playground being used with a real trained model.

Drives headless Google Chrome (or Chromium) over the DevTools protocol: opens /playground, types a clause,
classifies it, loads the sample contract, classifies it, and scrolls through the per-clause results. Every
step is captured as a frame with a caption, then assembled into a GIF.

    uv run python scripts/08_record_playground.py                       # -> demo/playground_usage.gif
    uv run python scripts/08_record_playground.py --url http://localhost:8000 --out demo/playground.gif

If no API is reachable at --url, the script starts `scripts/05_serve.py` itself (serving the latest model)
and stops it afterwards. Needs a trained model and Chrome/Chromium (set CHROME=/path/to/chrome if it is not
in the default macOS / Linux location). Uses `websockets` (installed with uvicorn[standard]) and Pillow.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

import typer
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
app = typer.Typer(add_completion=False)

CLAUSE = ("This Agreement shall be governed by and construed in accordance with the laws of the State of "
          "New York, without giving effect to any choice or conflict of law provision.")
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
]
WIDTH, HEIGHT = 1280, 860


# ----------------------------------------------------------------------------- helpers

def find_chrome() -> str:
    for c in [os.environ.get("CHROME", "")] + CHROME_CANDIDATES:
        if c and (Path(c).exists() or shutil.which(c)):
            return c if Path(c).exists() else shutil.which(c)
    raise typer.BadParameter("Chrome/Chromium not found; set CHROME=/path/to/chrome")


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
            return json.load(r).get("status") == "ok"
    except Exception:
        return False


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.load(r)


def font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


class CDP:
    """Minimal Chrome DevTools Protocol client over one page websocket."""

    def __init__(self, ws):
        self.ws, self.next_id = ws, 0

    async def call(self, method: str, **params):
        self.next_id += 1
        my_id = self.next_id
        await self.ws.send(json.dumps({"id": my_id, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == my_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expression: str):
        r = await self.call("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    async def wait_for(self, expression: str, timeout: float = 60.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if await self.js(expression):
                return
            await asyncio.sleep(0.25)
        raise TimeoutError(f"timed out waiting for: {expression}")

    async def shot(self) -> Image.Image:
        r = await self.call("Page.captureScreenshot", format="png")
        return Image.open(io.BytesIO(base64.b64decode(r["data"]))).convert("RGB")

    async def center(self, selector: str) -> tuple[int, int]:
        r = await self.js(f"(() => {{ const b = document.querySelector({json.dumps(selector)}).getBoundingClientRect();"
                          f" return [b.x + b.width / 2, b.y + b.height / 2]; }})()")
        return int(r[0]), int(r[1])


# ----------------------------------------------------------------------------- frames

class Recorder:
    def __init__(self):
        self.frames: list[tuple[Image.Image, int]] = []
        self.cap_font, self.step_font = font(26), font(18)

    def add(self, img: Image.Image, caption: str, ms: int, click: Optional[tuple[int, int]] = None,
            step: str = "") -> None:
        img = img.copy()
        d = ImageDraw.Draw(img, "RGBA")
        if click:  # highlight the element about to be clicked
            x, y = click
            for r, a in ((26, 70), (16, 120)):
                d.ellipse((x - r, y - r, x + r, y + r), fill=(42, 120, 214, a))
            d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 255, 255, 255), outline=(42, 120, 214, 255), width=3)
        # caption bar
        bar = 64
        d.rectangle((0, img.height - bar, img.width, img.height), fill=(20, 20, 19, 225))
        tx = 24
        if step:
            d.rounded_rectangle((tx, img.height - bar + 16, tx + 36, img.height - bar + 48), radius=8,
                                fill=(42, 120, 214, 255))
            d.text((tx + 18, img.height - bar + 32), step, font=self.step_font, fill="white", anchor="mm")
            tx += 52
        d.text((tx, img.height - bar / 2), caption, font=self.cap_font, fill="white", anchor="lm")
        self.frames.append((img, ms))

    def save(self, out: Path, width: int) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        scaled = []
        for img, ms in self.frames:
            h = int(img.height * width / img.width)
            small = img.resize((width, h), Image.LANCZOS)
            scaled.append((small.quantize(colors=192, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE), ms))
        scaled[0][0].save(out, save_all=True, append_images=[f for f, _ in scaled[1:]],
                          duration=[ms for _, ms in scaled], loop=0, optimize=True, disposal=1)
        self.frames[-1][0].save(out.with_name(out.stem + "_final_frame.png"))


async def record(url: str, chrome: str, rec: Recorder) -> None:
    import websockets

    port = 9333
    profile = tempfile.mkdtemp(prefix="ccc-chrome-")
    proc = subprocess.Popen(
        [chrome, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
         f"--window-size={WIDTH},{HEIGHT}", "--hide-scrollbars", "--force-color-profile=srgb",
         "--no-first-run", "--no-default-browser-check", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                pages = [t for t in get_json(f"http://127.0.0.1:{port}/json/list") if t["type"] == "page"]
                if pages:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        async with websockets.connect(pages[0]["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
            cdp = CDP(ws)
            await cdp.call("Page.enable")
            await cdp.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=1, mobile=False)
            await cdp.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
            await cdp.call("Page.navigate", url=f"{url}/playground")
            await cdp.wait_for("document.readyState === 'complete' && !!document.getElementById('run')")
            await cdp.wait_for("document.getElementById('status').textContent.includes('Latest')")
            await cdp.wait_for("document.getElementById('model').options.length > 1")
            mode = await cdp.js("document.getElementById('model').value")
            label = await cdp.js("document.getElementById('model').selectedOptions[0].textContent")
            compare = mode == "__compare__"
            rec.add(await cdp.shot(), ("Open the playground: compare mode runs both models" if compare
                                       else f"Open the playground: {label}"), 2200, step="1")

            # --- scene 1: type a clause and classify it
            await cdp.js("document.getElementById('text').focus()")
            for k in range(0, len(CLAUSE) + 1, max(1, len(CLAUSE) // 14)):
                await cdp.js(f"(() => {{ const t = document.getElementById('text'); t.value = {json.dumps(CLAUSE[:k])};"
                             f" t.dispatchEvent(new Event('input')); }})()")
                rec.add(await cdp.shot(), "Paste a clause or a whole contract", 110, step="2")
            await cdp.js(f"(() => {{ const t = document.getElementById('text'); t.value = {json.dumps(CLAUSE)};"
                         f" t.dispatchEvent(new Event('input')); }})()")
            run_xy = await cdp.center("#run")
            rec.add(await cdp.shot(), "Press Classify (or Ctrl/Cmd + Enter)", 1100, click=run_xy, step="3")
            await cdp.js("document.getElementById('run').click()")
            await cdp.wait_for("!!document.querySelector('#out .meta') && document.getElementById('run').textContent === 'Classify'")
            await asyncio.sleep(0.3)
            rec.add(await cdp.shot(), ("Transformer and classical predictions, each with its confidence" if compare
                                       else "Clause type with a calibrated confidence"), 3200, step="4")

            # --- scene 2: a whole contract
            sample_xy = await cdp.center('[data-sample="contract"]')
            rec.add(await cdp.shot(), "Load a whole contract (or paste your own)", 1100, click=sample_xy, step="5")
            await cdp.js("document.querySelector('[data-sample=\"contract\"]').click()")
            await asyncio.sleep(0.2)
            rec.add(await cdp.shot(), "Load a whole contract (or paste your own)", 1300, step="5")
            run_xy = await cdp.center("#run")
            rec.add(await cdp.shot(), "Classify: it is split into provisions automatically", 1000, click=run_xy, step="6")
            await cdp.js("document.getElementById('run').click()")
            await cdp.wait_for("document.getElementById('run').textContent === 'Classify' && "
                               "document.querySelectorAll('#out .clause').length > 1")
            await asyncio.sleep(0.3)
            rec.add(await cdp.shot(), ("Clause types found: blue = transformer, orange = classical" if compare
                                       else "Every clause type found in the contract, with confidence"), 3200, step="7")

            # --- scroll through per-clause predictions
            total = await cdp.js("document.documentElement.scrollHeight - window.innerHeight")
            for y in range(0, int(total) + 1, max(60, int(total) // 10)):
                await cdp.js(f"window.scrollTo(0, {y})")
                await asyncio.sleep(0.05)
                rec.add(await cdp.shot(), ("Per clause: both models side by side, agree or differ" if compare
                                           else "Per clause: top predictions and confidence for each provision"), 160, step="8")
            await cdp.js(f"window.scrollTo(0, {int(total)})")
            await asyncio.sleep(0.1)
            rec.add(await cdp.shot(), ("Per clause: both models side by side, agree or differ" if compare
                                       else "Per clause: top predictions and confidence for each provision"), 3200, step="8")
            await cdp.js("window.scrollTo(0, 0)")
            await asyncio.sleep(0.1)
            rec.add(await cdp.shot(), ("Same comparison via the API: POST /v1/compare" if compare
                                       else "Same results via the OpenAI-compatible API: POST /v1/chat/completions"), 2600, step="9")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


@app.command()
def main(
    url: str = typer.Option("http://localhost:8000", help="Base URL of the API (started automatically if down)"),
    out: Path = typer.Option(Path("demo/playground_usage.gif"), help="Output GIF"),
    width: int = typer.Option(960, help="GIF width in pixels"),
    port: int = typer.Option(8000, help="Port for the API if this script has to start it"),
) -> None:
    chrome = find_chrome()
    server = None
    if not healthy(url):
        url = f"http://127.0.0.1:{port}"
        print(f"Starting the API on {url} ...")
        server = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "05_serve.py"), "--port", str(port),
                                   "--log-level", "warning"], cwd=ROOT)
        t0 = time.time()
        while not healthy(url):
            if server.poll() is not None or time.time() - t0 > 180:
                raise typer.Exit(code=f"API did not become healthy; is a model trained? (make models)" and 1)
            time.sleep(1)
    try:
        rec = Recorder()
        asyncio.run(record(url, chrome, rec))
        out = out if out.is_absolute() else ROOT / out
        rec.save(out, width)
        size = out.stat().st_size / 1e6
        print(f"Wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} ({len(rec.frames)} frames, {size:.1f} MB)")
    finally:
        if server is not None:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    app()
