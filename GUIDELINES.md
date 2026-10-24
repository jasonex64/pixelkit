# pixel art: how Claude builds it

**who uses it:** Claude (any session) whenever Jason asks for pixel art, sprites, tiles, or scenes.
**what breaks without it:** raw-generated art breaks the grid, blurs, drifts off-palette, gets noise and muddy shading, and nobody can point at the bad pixel.

## hard rules
1. **Never assume. Brief first.** `pixelkit.py brief new brief.json`. Every unanswered field becomes a question for Jason. `brief check` must say COMPLETE before any pixel is drawn.
2. **Palette is locked.** Use a hex file (`.hex`/`.png`/`.json`) only. From an MJ moodboard: `pixelkit.py palette board*.png -o pal.hex`, show the swatch, and get a yes. Ramps under 3 shades get flagged: ask, don't invent colors.
3. **Tiles over raw art.** Scenes are tilemaps of small tiles. Each tile is authored as a hex-grid `.txt`, rendered, checked, and seam-tested.
4. **No raw image-model sprite sheets.** If one is unavoidable: `snap` onto the locked palette and grid, then `check`.
5. **Ship only on PASS.** Fix only the pixels `--mark` paints, and re-check. Report every WARN.

## loop
```
brief new brief.json        -> ask Jason every open question -> brief check   (exit 0)
palette moodboard*.png -o pal.hex                               (Jason confirms swatch)
extract refs... -o style.json --palette pal.hex --line-refs ref6.png ref7.png   (per-mode rules)
prompt style.json --mode sprite|scene                           -> paste into the gen step
write tile/sprite as hex grid .txt -> render x.txt -o x.png
check style.json x.png --mark x_mark.png                        (exit 0 = PASS)
tiles: tilemap tiles/ map.txt -o scene.png ; seams tiles/ map.txt ; seams tiles/ --self grass
```
Grid format: one row per line, space-separated `#rrggbb`, `.` = transparent.

## what check enforces
| group | FAIL | WARN |
|---|---|---|
| artifact | on_grid (incl. non-integer upscales), no_partial_alpha, no_grain (invisible-noise neighbors), no_near_dup_colors (rare stray shades), stray_pixels (1px dot unrelated to all neighbors), islands/pinholes (sprites) | orphans, speckle (1px detail density: style, not error) |
| color | on_palette (exact, when locked) | color_count |
| line (sprites) | outline_dark, outline_coverage (dark band: tinted near-blacks all count), outline_doubles | |
| line (scenes) | line_rhythm: shape-edge stair-steps must flow (1-1-1, 2-2-2, 1-2-1-2, 1-2-3); spikes like 1-3-1 = jaggies. needs 300+ edge runs, else n/a | |
| shading | ramp_length (>=3 analogous shades per material), ramp_contrast (muddy lit steps), shading_jumps (dark touching light of the same ramp) | ramp_hue_shift, no_pillow, light_direction (heuristics) |

Thresholds are calibrated **per mode** (sprite = has transparency, scene = full-bleed) from clean refs only.

## refs (current style.json)
- **sprites:** 12 items split out of the two item sheets (`split` removes frame, noisy checker, drop shadows, and enclosed bg pockets)
- **scenes:** ref1, ref2 (cropped), ref3, ref5
- **lines only:** ref6 (beach), ref7 (mountain): mood refs with good line detail, so they calibrate `line_rhythm` but not color/shading
- **style only:** ref4 (jpg), autumn scene (mood; its lines measure 0.086 spike vs 0.080 limit, slightly jaggier than ref6/7)

## is the checker trustworthy? (`pixelkit.py validate ...`, leave-one-out)
| | sprites (12) | scenes (4) |
|---|---|---|
| good refs passing when held out | 9/12 | 4/4 |
| grain, aa blur, off-grid, hue drift, baked shadow | 12/12 each | 4/4 each |
| flat shading | 12/12 (via locked palette; only 4/12 without) | 4/4 |
| broken outline | 11/12 | n/a |
| salt noise (random *palette* colors) | **6/12** | 4/4 |
| jaggies (content resampled, grid kept) | not measurable at 30px (eye review) | 3/4 (miss = ref2, too few edge runs to measure) |

The 3 held-out sprite fails are each a trait no other ref has (key glints, coral holes, spider gem). More sprite refs shrink this. Salt noise in palette colors looks like deliberate accent glints, so it's the weak spot. Eye-review the mark.

## known limits
- a bg pocket fully covered by shadow stays opaque in `split` (looks like a shadow-colored face)
- 2x2-block art reads as half-res; sanity-check `refs[].pitch`
- sprite line rhythm isn't measurable (30x30 = 9–34 edge runs, pure noise); sprites lean on outline checks + on_grid
- not checked, eye review: silhouette readability, composition
