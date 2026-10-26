---
name: pixel-art
description: Use when asked to make pixel art, sprites, characters, UI or card frames, tiles, tilesets, or pixel backgrounds; enforces brief-first questions, locked palette, tiles over raw art, and pixelkit checks.
---

# Pixel art (Jason's workflow)

Who uses it: Claude, whenever Jason asks for pixel art.
What breaks without it: raw generation breaks the grid, blurs, drifts off-palette, adds noise and muddy shading.

## 0. Tooling
- Need `pixelkit.py`. If it isn't in the working dir: `git clone https://github.com/jasonex64/pixelkit` and `pip install -r requirements.txt`.
- Run `python3 pixelkit.py selftest`; it must pass. Read `GUIDELINES.md` once.
- Rules live in `specs/style.json`. Reference images are not in the repo; ask Jason for them only when re-running `extract`/`validate`.

## 1. Never assume: brief first
- `python3 pixelkit.py brief new brief.json`
- Fill ONLY what Jason actually said. Ask every other field (AskUserQuestion, up to 4 per call; give each option's trade-off).
- `python3 pixelkit.py brief check brief.json` must print BRIEF COMPLETE before drawing anything.
- `display_scale` is a list per breakpoint (e.g. `3,6`): art must look right on both mobile and desktop.

## 2. Locked palette
- Palette = a hex file. No file but a moodboard: `python3 pixelkit.py palette board*.png -o pal.hex`, show the swatch, and get Jason's yes.
- Ramps under 3 shades are flagged: ask, never invent colors.
- `python3 pixelkit.py extract <refs> -o style.json --palette pal.hex --line-refs <mood refs with good lines>`
- Plor/Offbeat: one palette per weather stage (calm, windy, storm). Other games get a completely different palette and theme.

## 3. Tiles over raw art
- Scenes are tilemaps or layers. Author each tile/sprite as a hex-grid .txt (one row per line, space-separated #rrggbb, `.` = transparent) at native size.
- Use `python3 pixelkit.py prompt style.json --mode sprite|scene` as hard rules.
- Never ship image-model sprite sheets raw. If one is unavoidable: `snap` onto the palette and grid first.
- Non-pixel refs (photos, paintings, MJ) inform palette, mood, and shapes only, never the quality thresholds.

## 4. Check loop
- `python3 pixelkit.py render x.txt -o x.png`
- `python3 pixelkit.py check style.json x.png --mark x_mark.png`
- On FAIL: fix only the marked pixels, re-render, re-check. Max 5 loops, then show Jason the mark and ask.
- Tiles: `tilemap tiles/ map.txt -o scene.png`, `seams tiles/ map.txt`, `seams tiles/ --self NAME`.

## 5. Deliver
- Native png plus integer-scaled pngs (nearest-neighbor) for each breakpoint scale, and the .txt source.
- Report the result: PASS/FAIL plus every WARN in one line each. Never claim PASS without the command output.
- Known weak spots: stray pixels in palette colors (they look like glints), and sprite line rhythm (too small to measure). Eye-check the mark.

## Open decisions (ask, don't pick)
Outline style (selout / black / none), minimum phone width, and canvas size (proposed: 360px min, 3x on phones, 120px safe area, ~320px-wide scenes for desktop) are unconfirmed.
