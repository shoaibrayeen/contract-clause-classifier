"""The playground recorder's frame/GIF assembly works without Chrome or a model."""

import importlib.util
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("record_playground", ROOT / "scripts" / "08_record_playground.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["record_playground"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_recorder_builds_captioned_gif(tmp_path):
    mod = _module()
    rec = mod.Recorder()
    for i, color in enumerate(("white", "lightgray", "white")):
        rec.add(Image.new("RGB", (mod.WIDTH, mod.HEIGHT), color), f"step {i}", 200,
                click=(100, 100) if i == 1 else None, step=str(i + 1))
    out = tmp_path / "pg.gif"
    rec.save(out, width=480)
    gif = Image.open(out)
    assert gif.n_frames == 3 and gif.size[0] == 480
    assert (tmp_path / "pg_final_frame.png").exists()
    # the caption bar darkens the bottom of the frame
    assert sum(rec.frames[0][0].getpixel((5, mod.HEIGHT - 5))) < 150
