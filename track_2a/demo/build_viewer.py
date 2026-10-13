#!/usr/bin/env python3
"""Build dist/polyjudge_viewer.html from viewer_data.json + assets/*.png (stdlib only).

Offline, no API calls, no network. Usage:  python build_viewer.py [--full-res]
"""
import base64, hashlib, json, struct, sys, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "viewer_data.json"
TEMPLATE = ROOT / "src" / "template.html"
OUT = ROOT / "dist" / "polyjudge_viewer.html"
ASSETS = ROOT / "assets"
# role key -> (file, initial for placeholder, placeholder colour)
IMAGES = {
    "extractor_en": ("char_de.png", "E", "#6b8cae"),
    "extractor_tgt": ("char_fr.png", "T", "#ae6b7e"),
    "compare": ("char_it.png", "C", "#6bae85"),
    "verifier": ("char_judge.png", "V", "#7d6bae"),
}
TARGET_PX = 480  # avatars are shown <= ~140 css px; downscale big portraits unless --full-res


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def downscale_png(raw: bytes, target: int) -> bytes:
    """Box-filter downscale of an 8-bit RGB/RGBA non-interlaced PNG; returns raw unchanged if unsupported."""
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        return raw
    pos, idat, hdr = 8, [], None
    while pos < len(raw):
        ln, typ = struct.unpack(">I4s", raw[pos:pos + 8])
        body = raw[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", body)
        elif typ == b"IDAT":
            idat.append(body)
        pos += 12 + ln
    w, h, depth, ctype, _, _, interlace = hdr
    if depth != 8 or ctype not in (2, 6) or interlace != 0 or w <= target:
        return raw
    bpp = 3 if ctype == 2 else 4
    stride = w * bpp
    data = zlib.decompress(b"".join(idat))
    f = max(2, -(-w // target))  # integer factor
    rows, prev = [], bytearray(stride)
    for y in range(h):
        ft = data[y * (stride + 1)]
        cur = bytearray(data[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        if ft == 1:
            for i in range(bpp, stride):
                cur[i] = (cur[i] + cur[i - bpp]) & 255
        elif ft == 2:
            for i in range(stride):
                cur[i] = (cur[i] + prev[i]) & 255
        elif ft == 3:
            for i in range(stride):
                l = cur[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + ((l + prev[i]) >> 1)) & 255
        elif ft == 4:
            for i in range(stride):
                l = cur[i - bpp] if i >= bpp else 0
                ul = prev[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + _paeth(l, prev[i], ul)) & 255
        rows.append(cur)
        prev = cur
    nw, nh = w // f, h // f
    out = bytearray()
    area = f * f
    for oy in range(nh):
        block = rows[oy * f:(oy + 1) * f]
        out.append(0)
        chans = []
        for c in range(bpp):
            acc = [0] * nw
            for r in block:
                ch = r[c::bpp]
                for k in range(f):
                    sl = ch[k::f][:nw]
                    acc = [x + y for x, y in zip(acc, sl)]
            chans.append([(v + area // 2) // area for v in acc])
        for px in zip(*chans):
            out.extend(px)

    def chunk(t, b):
        return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", nw, nh, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(out), 9)) + chunk(b"IEND", b""))


def placeholder(letter, colour):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="#e9e4d8"/>'
           f'<circle cx="50" cy="38" r="20" fill="{colour}"/><path d="M12 100c0-26 17-38 38-38s38 12 38 38z" fill="{colour}"/>'
           f'<text x="50" y="46" font-family="sans-serif" font-size="22" font-weight="700" fill="#fff" text-anchor="middle">{letter}</text></svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def main():
    full = "--full-res" in sys.argv
    raw = DATA.read_bytes()
    docs = json.loads(raw.decode("utf-8"))
    imgs, notes = {}, []
    for role, (fn, letter, colour) in IMAGES.items():
        p = ASSETS / fn
        if p.is_file():
            b = p.read_bytes()
            small = b if full else downscale_png(b, TARGET_PX)
            imgs[role] = "data:image/png;base64," + base64.b64encode(small).decode()
            notes.append(f"{fn}: {len(b)//1024} KB -> {len(small)//1024} KB")
        else:
            imgs[role] = placeholder(letter, colour)
            notes.append(f"{fn}: MISSING -> placeholder avatar")
    info = {"source_sha256": hashlib.sha256(raw).hexdigest(), "docs": len(docs)}
    # Re-serialise compactly; escape "</" so the JSON can never close its <script> tag.
    def j(o):
        return json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\/").replace("\u2028", "\u2028").replace("\u2029", "\u2029")
    html = TEMPLATE.read_text(encoding="utf-8")
    for key, val in (("__DATA_JSON__", j(docs)), ("__IMAGES_JSON__", j(imgs)), ("__BUILD_INFO__", j(info))):
        assert html.count(key) == 1, key
        html = html.replace(key, val)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print("\n".join(notes))
    print(f"wrote {OUT} ({OUT.stat().st_size/1048576:.2f} MB), {len(docs)} documents")


if __name__ == "__main__":
    main()
