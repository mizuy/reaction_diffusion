#!/usr/bin/env python3
"""Reference sheet of real pit-pattern images for the simulation goal.

Collects the clinical / schematic images already in the repository
(gr1_lrg.jpg: Kudo classification table; docs/serrated_pitpattern.pptx) into
one page ordered by the goal's main sequence I -> IIIL -> IV_B -> IV_V, with
the serrated subtypes (reference only) below.  Labels follow the captions on
the source slides.  For morphology reference only, not a diagnostic basis.

    uv run python src/make_pit_reference_sheet.py --out docs/images/pit_reference_sheet.png
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, SURFACE, RULE = "#0b0b0b", "#52514e", "#fcfcfb", "#d6d5cf"
JP = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
JP_BOLD = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"


def font(size: int, bold: bool = False):
    for name in ((JP_BOLD if bold else JP), JP, "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def pptx_media(path: Path) -> dict[str, Image.Image]:
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.startswith("ppt/media/"):
                try:
                    out[n.split("/")[-1].split(".")[0]] = Image.open(io.BytesIO(z.read(n))).convert("RGB")
                except OSError:
                    pass
    return out


def kudo_rows(path: Path) -> dict[str, dict[str, Image.Image]]:
    """Schematic circle and endoscopic photo per row of the Kudo table."""
    im = Image.open(path).convert("RGB")
    names = ["I", "II", "IIIs", "IIIL", "IV", "VI", "VN"]
    top, step = 4, 171.4
    rows = {}
    for i, n in enumerate(names):
        y0 = int(top + i * step)
        rows[n] = dict(schematic=im.crop((292, y0 + 12, 444, y0 + 164)),
                       photo=im.crop((1096, y0 + 4, 1286, y0 + 168)))
    return rows


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    """Scale to cover w x h and center-crop (no distortion)."""
    s = max(w / img.width, h / img.height)
    r = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.Resampling.LANCZOS)
    x, y = (r.width - w) // 2, (r.height - h) // 2
    return r.crop((x, y, x + w, y + h))


def contain(img: Image.Image, w: int, h: int) -> Image.Image:
    s = min(w / img.width, h / img.height)
    r = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.Resampling.LANCZOS)
    c = Image.new("RGB", (w, h), SURFACE)
    c.paste(r, ((w - r.width) // 2, (h - r.height) // 2))
    return c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "docs/images/pit_reference_sheet.png")
    a = p.parse_args()

    m = pptx_media(ROOT / "docs/serrated_pitpattern.pptx")
    k = kudo_rows(ROOT / "gr1_lrg.jpg")
    strip = m["image19"]  # slide 7 top row: binary RD panels aligned with Type I / IIIL / IVB / IVV
    import numpy as np

    white = (np.asarray(strip.convert("L")) > 240).mean(0) > 0.98
    edges = np.flatnonzero(np.diff(white.astype(int)))  # panel boundaries at white-gap edges
    starts = [e + 1 for e in edges if not white[e + 1]]
    ends = [e + 1 for e in edges if white[e + 1]]
    if not white[0]:
        starts = [0] + starts
    if not white[-1]:
        ends = ends + [strip.width]
    rd = [strip.crop((s0, 0, e0, strip.height)) for s0, e0 in zip(starts, ends)][:4]
    pair = m["image10"]  # slide 3: left 鋸IV_B, right 鋸IV_V
    half = pair.width // 2
    ivb_sl3, ivv_sl3 = pair.crop((0, 0, half, pair.height)), pair.crop((half, 0, pair.width, pair.height))

    W, pad, colw = 1560, 24, 330
    left = 190
    img = Image.new("RGB", (W, 4000), SURFACE)
    d = ImageDraw.Draw(img)
    d.text((pad, 20), "Pit pattern 参照シート（実画像）", fill=INK, font=font(30, True))
    d.text((pad, 64), "ゴール主系列 I → IIIL → IV_B → IV_V に対応する手元の画像。ラベルは元スライドの表記に従う。"
                      "形態の参照であり診断の根拠ではない。", fill=MUTED, font=font(16))

    # ---- main sequence ----------------------------------------------------------
    y = 110
    d.text((pad, y), "主系列", fill=INK, font=font(22, True))
    heads = ["Type I", "Type IIIL", "Type IV_B（鋸IV_B）", "Type IV_V（鋸IV_V）"]
    subs = ["規則的な丸 pit", "やや大きい管状・類円形 pit", "短い分岐（樹枝状）", "密な脳回状（gyrus-like）"]
    for j, (h, s) in enumerate(zip(heads, subs)):
        x = left + j * (colw + 10)
        d.text((x, y + 36), h, fill=INK, font=font(19, True))
        d.text((x, y + 62), s, fill=MUTED, font=font(14))
    y += 92

    def row(label: str, note: str, tiles: list, h: int, mode: str = "fit"):
        nonlocal y
        d.line((pad, y - 6, W - pad, y - 6), fill=RULE)
        d.text((pad, y + 4), label, fill=INK, font=font(15, True))
        d.multiline_text((pad, y + 30 + 20 * label.count("\n")), note, fill=MUTED, font=font(12), spacing=3)
        for j, t in enumerate(tiles):
            if t is None:
                continue
            im, cap = t
            x = left + j * (colw + 10)
            img.paste((fit if mode == "fit" else contain)(im, colw, h), (x, y))
            if cap:
                d.text((x + 2, y + h + 4), cap, fill=MUTED, font=font(12))
        y += h + 34

    row("工藤分類\n模式図", "gr1_lrg.jpg", [(k["I"]["schematic"], "Type I"), (k["IIIL"]["schematic"], "Type IIIL"),
                                         (k["IV"]["schematic"], "Type IV（原分類は B/V を区別しない）"), None],
        170, "contain")
    row("工藤分類\n内視鏡像", "gr1_lrg.jpg\nクリスタルバイオレット", [(k["I"]["photo"], "Type I"), (k["IIIL"]["photo"], "Type IIIL"),
                                                       (k["IV"]["photo"], "Type IV"), None], 280)
    row("鋸歯状\n病変", "スライド3（松下 2020）\nスライド7", [None, None, (ivb_sl3, "鋸IV_B型（スライド3）"), (ivv_sl3, "鋸IV_V型（スライド3）")], 280)
    row("", "", [None, None, (m["image23"], "IV_H または 鋸IV_B（スライド7）"), (m["image22"], "IV_H または 鋸IV_V（スライド7）")], 280)
    row("反応拡散\n（参照）", "スライド7上段の\n二値像。実画像\nではない", [(rd[0], "Type I 相当"), (rd[1], "Type IIIL 相当"),
                                                       (rd[2], "Type IV_B 相当"), (rd[3], "Type IV_V 相当")], 200)

    d.line((pad, y - 6, W - pad, y - 6), fill=RULE)
    d.text((pad, y + 4), "スライド6", fill=INK, font=font(15, True))
    d.multiline_text((pad, y + 30), "反応拡散スイープ\n（左 I → 中央 IV_V\nの迷路相）", fill=MUTED, font=font(12), spacing=3)
    wide = contain(m["image18"], 4 * colw + 30, 340)
    img.paste(wide, (left, y))
    d.text((left + 2, y + 344), "I型 → IIIL型 → IV_B型 → IV_V型 の変化は連続的に表現できる（スライド6）", fill=MUTED, font=font(12))
    y += 340 + 34

    # ---- serrated subtypes -----------------------------------------------------
    y += 10
    d.text((pad, y), "鋸歯状の亜型（ゴールでは参照のみ・主系列の外）", fill=INK, font=font(22, True))
    y += 46
    row("II 系", "スライド7\nスライド3（Aoki 2018）", [(m["image20"], "Type II（スライド7）"), (m["image21"], "III_H または II-L（スライド7）"),
                                                (m["image9"], "Classical II / II-L / II-O（スライド3）"), None], 260, "contain")
    row("H 系", "スライド2（浦岡 2011、\nSano 2005、藤井 1999）", [(m["image3"], "III_H（シダの葉様）"), (m["image5"], "III_H"),
                                                         (m["image4"], "IV_H"), (m["image7"], "松毬様 pit")], 260)

    y += 6
    d.line((pad, y, W - pad, y), fill=RULE)
    src = ("出典（元スライドの記載）: 工藤分類表 gr1_lrg.jpg ／ スライド2: 浦岡ら 胃と腸 46(4) 2011; Sano Y et al. Dig Endosc 2005; "
           "藤井ら 胃と腸 34 1999 ／ スライド3: Aoki H et al. Dig Dis Sci 2018; 松下ら 胃と腸 55(5) 2020 ／ "
           "スライド6・7: docs/serrated_pitpattern.pptx（スライド7の写真は出典記載なし）")
    words, line, lines = src.split(" "), "", []
    for w_ in words:
        if d.textlength(line + w_ + " ", font=font(12)) > W - 2 * pad:
            lines.append(line)
            line = ""
        line += w_ + " "
    lines.append(line)
    for i, l in enumerate(lines):
        d.text((pad, y + 10 + i * 18), l, fill=MUTED, font=font(12))
    y += 20 + 18 * len(lines)
    img = img.crop((0, 0, W, y + 10))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(a.out)
    print("saved", a.out, img.size)


if __name__ == "__main__":
    main()
