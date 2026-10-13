#!/usr/bin/env python3
"""Smoke test for dist/polyjudge_viewer.html (stdlib only). Exit code 1 on any failure.

1. parses the BUILT html, extracts the embedded JSON and checks: no split=test document has non-null gold,
   prediction lengths match token counts, data equals viewer_data.json, no external resources, required texts.
2. if a Chromium-based browser (Edge/Chrome) is found, opens the file headless with #selftest, which renders
   every document/method/mode and the PolyJudge scenes, and fails on any console/page error.
"""
import json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "dist" / "polyjudge_viewer.html"
SRC = ROOT / "viewer_data.json"
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
        print("FAIL:", msg)


def main():
    if not HTML.is_file():
        print("FAIL: build first (python build_viewer.py)")
        return 1
    html = HTML.read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    check(m, "embedded data script not found")
    docs = json.loads(m.group(1).replace("<\/", "</")) if m else []
    check(len(docs) == 183, f"expected 183 docs, got {len(docs)}")
    tests = [d for d in docs if d["split"] == "test"]
    check(len(tests) == 168, f"expected 168 test docs, got {len(tests)}")
    scene = [d for d in docs if d.get("m2_transcript")]
    check(len(scene) == 174 and sum(d["split"] == "test" for d in scene) == 168 and sum(d["split"] == "dev_select" for d in scene) == 6,
          f"PolyJudge-tab documents: expected 174 (168 test + 6 dev) with transcript, got {len(scene)}")
    check(all(g.get("messages") for d in scene for g in d["m2_transcript"]), "a PolyJudge group has no messages")
    for d in tests:
        check(d.get("gold_a") is None and d.get("gold_b") is None, f"test doc {d['id']} has non-null gold")
    for d in docs:
        for side in "ab":
            n = len(d["text_" + side].split())
            for meth, p in d["pred"].items():
                check(len(p[side]) == n, f"{d['id']} {meth}.{side}: {len(p[side])} values vs {n} tokens")
    if SRC.is_file():
        check(docs == json.loads(SRC.read_text(encoding="utf-8")), "embedded data differs from viewer_data.json")
    page = html[:m.start()] + html[m.end():] if m else html
    page = re.sub(r'<script type="application/json" id="imgs">.*?</script>', "", page, flags=re.S)
    check(not re.search(r'(?:src|href)\s*=\s*["\']?https?:|url\(\s*["\']?https?:|@import|<link\b', page), "external resource reference found")
    check("Replay of recorded M2 outputs (not live)" in html, "replay label missing")
    check("M1 vs B0: only +0.003" in html, "scoreboard note missing")
    check(not re.search(r"winner", page, re.I), "'winner' appears in UI text")
    check(len(re.findall(r"\bbest\b", page, re.I)) == 1, "'best' must appear only in the required scoreboard note")
    check("Data: SwissGov-RSD, CC-BY-4.0 (Wastl, Vamvas, Sennrich). Character illustrations were generated with an AI image generator." in html, "footer missing")
    navm = re.search(r'<nav id="bnav"[^>]*aria-label="[^"]+"[^>]*>(.*?)</nav>', html, re.S)
    check(navm and len(re.findall(r"<button[\s>]", navm.group(1))) == 3, "bottom nav must have exactly three items")
    check(navm and len(re.findall(r'aria-current="page"', navm.group(1))) == 1, "nav must have exactly one aria-current in the static HTML")
    labels = re.findall(r'<span class="lbl">([^<]*)</span>', navm.group(1)) if navm else []
    check(labels == ["Viewer", "PolyJudge", "Scoreboard"], f"nav order/labels wrong: {labels}")
    cur = re.search(r'data-tab="(\w+)" aria-current="page"', navm.group(1)) if navm else None
    check(cur and cur.group(1) == "scene", "default tab (aria-current) must be PolyJudge")
    check('<section id="tab-scene">' in html and '<section id="tab-viewer" hidden>' in html, "PolyJudge section must be visible by default")
    check("Table scene" not in page, "old name 'Table scene' still present in the UI")
    check("<header" not in page and "<h1" not in page, "old header band / left title still present")
    check(re.search(r"#bnav\{position:fixed;top:0;left:0;right:0;[^}]*grid-template-columns:repeat\(3", html), "nav must be full-width (fixed, left/right 0, three equal columns)")
    check("click to expand" not in html.lower(), "'click to expand' text still present")
    check("<aside" not in page and 'id="side"' not in page, "left sidebar element still present")
    check('id="pk-panel"' in html and 'id="pk-btn"' in html and 'id="pk-list"' in html, "document picker missing")
    check("● Replay" in html and 'id="info-pop"' in html, "Replay chip / info popover missing")
    for sent in ("Roles are shown by function. Spans come from the recorded M2 prediction for the group.", "PolyJudge here is the multi-agent method M2, shown as a replay. It is a comparison method; the main submitted system is M1 (see Scoreboard)."):
        check(sent in html, "info popover sentence missing: " + sent[:40])
    check(html.count('<nav ') == 1 and 'class="ind"' in html and 'class="lamp"' in html, "tubelight indicator (shared .ind with .lamp) missing")
    check(navm and len(re.findall(r"<svg[\s>]", navm.group(1))) == 3, "bottom nav items need inline SVG icons")
    check("<nav role=\"tablist\"" not in html, "old top tab bar still present")
    check("offline · recorded predictions only · no API calls</footer>" in html, "footer offline note missing")
    check(not re.search(r"pres-toggle|pres-exit|sc-pres", html), "an old presentation button / its CSS or JS is still present")
    check("Exit presentation" not in html and ">Presentation" not in html and "Presentation (P)" not in html, "visible 'Presentation' / 'Exit presentation' text still present")
    check("Keyboard: P = presentation mode on/off, Esc = leave, 1/2/3 = tabs, [ ] = previous/next document." in html, "keyboard help line missing in the info popover")
    check("#3a2c22" not in html, "old brown seat background colour still present")
    m_chair = re.search(r"\.chair\{([^}]*)\}", html)
    check(m_chair and "background" not in m_chair.group(1) and "shadow" not in m_chair.group(1) and "border" not in m_chair.group(1), ".chair (seat) must have no background, shadow or border")
    check("data:image/" in html, "no embedded images")

    cands = [shutil.which("msedge"), shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("chromium"),
                            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                            r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
    exe = next((p for p in cands if where_ok(p)), None)
    if not exe:
        print("SKIP browser self-test (no Edge/Chrome found)")
    else:
        with tempfile.TemporaryDirectory() as prof:
            cmd = [exe, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-first-run", "--window-size=1920,1080", "--virtual-time-budget=60000",
                   "--dump-dom", HTML.as_uri() + "#selftest"]
            r = subprocess.run(cmd, capture_output=True, timeout=300)
        dom = r.stdout.decode("utf-8", "replace")
        status = re.search(r'data-selftest="(\w+)"', dom)
        rep = re.search(r'<pre id="selftest-report">(.*?)</pre>', dom, re.S)
        check(status and status.group(1) == "ok", "browser self-test: " + (rep.group(1)[:1500] if rep else "no report (page did not finish / crashed)"))
        if status and status.group(1) == "ok":
            print("browser self-test OK:", rep.group(1).strip().splitlines()[0])
    print("SMOKE TEST", "FAILED" if fails else "PASSED")
    return 1 if fails else 0


def where_ok(p):
    return p and os.path.isfile(p)


if __name__ == "__main__":
    sys.exit(main())
