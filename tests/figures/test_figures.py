"""Step K: the final figures are all listed, their sources exist, colours are stable."""

import os
import re
from pathlib import Path

import pytest

from aerovigil.figures import style as S

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "reports" / "figures" / "final"
INDEX = FINAL / "FIGURES.md"

pytestmark = pytest.mark.skipif(not INDEX.is_file(), reason="figures not built")


def rows():
    out = []
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| \[(.+?\.png)\]\(.+?\) \| (.+?) \| (.+?) \| `(.+?)` \|$", line)
        if m:
            out.append((m.group(1), m.group(2), re.findall(r"`([^`]+)`", m.group(3)),
                        m.group(4)))  # fmt: skip
    return out


def test_every_png_is_listed_once_with_a_caption():
    listed = [r[0] for r in rows()]
    assert len(listed) == len(set(listed))
    pngs = sorted(p.name for p in FINAL.glob("*.png"))
    assert sorted(listed) == pngs
    assert all(len(r[1]) > 20 for r in rows())


def test_every_png_has_an_svg_except_screenshots():
    for name, *_ in rows():
        if name.startswith("M13_"):
            continue
        assert (FINAL / name.replace(".png", ".svg")).is_file(), name


def test_scripts_exist_and_tracked_sources_exist():
    for name, _, data, script in rows():
        assert (ROOT / script).is_file(), (name, script)
        for d in data:
            path = d.split(" (")[0].split(" + ")[0]
            if "*" in path or "{" in path:
                continue
            if path.startswith("data/") and os.environ.get(
                "AEROVIGIL_ALLOW_MISSING_DATA"
            ):
                continue  # a fresh clone has no processed data (smoke / CI)
            assert (ROOT / path).is_file(), (name, path)


def test_detector_colours_and_markers_are_unique():
    cols = [S.colour(d) for d in S.DETECTORS]
    assert len(set(cols)) == len(cols) == 8
    styles = [(S.MODELS[m][1], S.MODELS[m][2], S.MODELS[m][3]) for m in S.MODELS]
    assert len(set(map(str, styles))) == len(styles)


def test_cell_label_is_three_short_lines():
    for f in S.FAMILIES:
        for m in S.MODES:
            lines = S.cell_label(f, m).split("\n")
            assert len(lines) == 3 and max(map(len, lines)) <= 14
