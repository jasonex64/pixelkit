#!/usr/bin/env python3
"""
pixelkit — deterministic pixel-art spec extraction + lint.

who uses it:  you (or an agent) before/after generating pixel art from refs.
what breaks without it: artifacts (stray px, holes, soft alpha, near-dup colors),
  wrong colors, inconsistent lines, bad shading (flat/muddy ramps, pillow
  shading, banding jumps, wrong light) slip through and nobody can say *where*.

commands
  extract  REF [REF...] -o style.json [--lock REF...] [--calibrate-all]
                                           measure refs -> spec. thresholds calibrated
                                           from clean refs only; palette locked only
                                           to --lock refs (else rules-only)
  prompt   style.json                      spec -> short prompt block
  check    style.json IMG [--json] [--mark out.png] [--strict]
                                           lint image vs spec. exit 1 on any FAIL
                                           (--strict: WARN fails too). --mark paints
                                           FAIL px magenta, WARN px yellow
  snap     style.json IMG -o out.png --size WxH [--scale N]
                                           force image onto grid + palette
  render   grid.txt -o out.png [--scale N] hex-grid text -> png
  split    SHEET -o DIR [--preview p.png]  sprite sheet on bg/checker/shadows -> one
                                           transparent png per sprite
  selftest                                 built-in tests (one per check)

deterministic: no randomness, ties broken by palette index / color key.
deps: pillow, numpy
"""
import argparse, colorsys, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# ---- knobs (all thresholds live here, so they're auditable) ----
ALPHA_CUT = 128          # alpha >= this counts as opaque
MAX_CLEAN_COLORS = 64    # more unique colors than this = not clean pixel art
SCALE_TOL = 0.02         # max fraction of mismatched px to accept a grid scale
RARE_SHARE = 0.001       # a color on <= this share of px (min 3px) counts as "rare"
NEAR_DUP_DIST = 12       # RGB distance under which 2 used colors are "the same color twice"
ORPHAN_MARGIN = 0.005    # orphans allowed above the refs' rate (~5px on a 32x32)
ISLAND_MAX_PX = 2       # detached opaque blobs this small = artifact
PINHOLE_MAX_PX = 2       # enclosed transparent holes this small = artifact
MIN_RAMP_LEN = 3         # every major material needs >= this many analogous colors
MAJOR_RAMP_SHARE = {"sprite": 0.10, "scene": 0.02}  # ramp share of fill px to count as "major"
MUDDY_MIN_L = 20         # steps darker than this L* are exempt (subtle dark atmosphere is ok)
MIN_RAMP_STEP = 4        # min CIE L* gap between neighbors in a ramp (else muddy)
ANALOG_HUE = 45          # max hue gap (deg) for two colors to be analogous
GREY_SAT, DARK_VAL = 0.15, 0.12
PILLOW_DARK = 10         # lit-side rim darker than core by this ...
SPLIT_FRAME_SHARE = 0.5  # edge line with one color on >= this share = frame, peel it
SPLIT_WALL = 0.45        # bg flood stops at px darker than this x bg L* (outlines)
SPLIT_SHADOW_TOL = 0.12  # per-channel ratio slack around the learned shadow tint
SPLIT_SHADOW_MODES = 4   # max distinct shadow tints learned per sheet
SPLIT_MERGE_GAP = 3      # detached pieces this close to a big sprite join it
SPLIT_POCKET_DIST = 8    # enclosed pocket px within this RGB of a known bg color = bg
SPLIT_MIN_PX = 12        # smaller detached blobs are dropped
OUTLINE_MAX_L = 30       # edge px with CIE L* <= this count as outline (dark band)
GRAIN_DIST = 10          # neighbors differing by < this RGB = invisible noise (grain)
SEAM_NOVEL_MAX = 0.10    # max share of seam color-pairs never seen inside any tile
SEAM_MARGIN = 0.03       # seam jump density allowed above tile-interior density
OFFGRID_FIT = 0.6        # comb-fit score above which a native-looking image is really a non-integer upscale
LINE_MIN_RUNS = 300      # fewer edge runs than this = line rhythm not measurable (n/a)
PILLOW_EVEN = 12         # ... and lit/shadow rims within this = pillow shading


# ---------- io / color utils ----------
def load(path):
    return np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)

def hx(rgb):
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb[:3])

def unhx(s):
    s = s.lstrip("#"); return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))

def k2rgb(k):
    return ((k >> 16) & 255, (k >> 8) & 255, k & 255)

def lum(rgb):
    r, g, b = rgb[:3]; return 0.299*r + 0.587*g + 0.114*b

def lstar(rgb):
    """CIE L* (0..100) — perceptual lightness, used for ramp ordering + step contrast."""
    lin = [((c / 255) / 12.92) if c / 255 <= 0.04045 else (((c / 255) + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
    Y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    f = Y ** (1 / 3) if Y > (6 / 29) ** 3 else Y / (3 * (6 / 29) ** 2) + 4 / 29
    return 116 * f - 16

def hsv(rgb):
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb[:3])); return h * 360, s, v

def is_grey(rgb):
    _, s, v = hsv(rgb); return s < GREY_SAT or v < DARK_VAL

def hue_diff(a, b):
    d = abs(hsv(a)[0] - hsv(b)[0]) % 360; return min(d, 360 - d)

def analogous(a, b):
    """same hue family. near-black can cap the bottom of any dark ramp and near-white
    the top of any light ramp (blown highlights), so they bridge to colored shades."""
    for x, y in ((a, b), (b, a)):
        _, sx, vx = hsv(x)
        if vx < DARK_VAL and lstar(y) < 40:
            return True
        if sx < 0.10 and vx > 0.95 and lstar(y) > 70:
            return True
    ga, gb = is_grey(a), is_grey(b)
    if ga or gb:
        return ga and gb
    return hue_diff(a, b) <= ANALOG_HUE


# ---------- grid ----------
def detect_scale(img, max_scale=32):
    """largest integer block size s where the image is (almost) s x s blocks."""
    h, w = img.shape[:2]
    for s in range(min(max_scale, h, w), 1, -1):
        if h % s or w % s:
            continue
        up = img[::s, ::s].repeat(s, 0).repeat(s, 1)
        mm = float(np.any(up != img, axis=2).mean())
        if mm <= SCALE_TOL:
            return s, mm
    return 1, 0.0

def _axis_pitch(img, axis, lo=1.5, hi=40.0, step=0.005):
    """fractional pixel pitch + grid offset along one axis via a Fourier comb over
    edge strength. returns (pitch, offset, score 0..1)."""
    a = img[..., :3].astype(np.float32)
    sig = np.abs(np.diff(a, axis=axis)).sum(-1).mean(axis=1 - axis)
    sig = np.clip(sig - np.median(sig), 0, None)
    x = np.arange(len(sig)) + 1.0          # diff i = boundary between px i and i+1
    tot = max(sig.sum(), 1e-9)
    ps = np.arange(lo, hi, step)
    S = np.array([(sig * np.exp(-2j * np.pi * x / p)).sum() for p in ps])
    sc = np.abs(S) / tot
    # harmonics (p/2, p/3) score as high as p, so take the largest strong peak, then refine locally
    i = int(np.nonzero(sc >= 0.85 * sc.max())[0].max())
    win = (ps >= ps[i] * 0.97) & (ps <= ps[i] * 1.01)
    j = int(np.nonzero(win)[0][np.argmax(sc[win])])
    p = float(ps[j]); off = (-np.angle(S[j]) / (2 * np.pi) * p) % p
    return p, off, float(sc[j])

def _sample_grid(img, px, ox, py, oy):
    """median of the central half of each cell -> native image. deterministic."""
    H, W = img.shape[:2]
    xs = np.arange(ox - px * np.ceil(ox / px), W, px); ys = np.arange(oy - py * np.ceil(oy / py), H, py)
    xs = [x for x in xs if x + px * 0.75 <= W and x + px * 0.25 >= 0]
    ys = [y for y in ys if y + py * 0.75 <= H and y + py * 0.25 >= 0]
    out = np.zeros((len(ys), len(xs), 4), np.uint8)
    for j, y in enumerate(ys):
        y0, y1 = int(max(y + py * 0.25, 0)), int(min(y + py * 0.75, H))
        y1 = max(y1, y0 + 1)
        for i, x in enumerate(xs):
            x0, x1 = int(max(x + px * 0.25, 0)), int(min(x + px * 0.75, W))
            x1 = max(x1, x0 + 1)
            out[j, i] = np.median(img[y0:y1, x0:x1].reshape(-1, 4), axis=0).astype(np.uint8)
    return out, xs, ys

def merge_close(img, dist):
    """collapse colors within `dist` RGB onto the most common one (greedy by count,
    ties by color value). cleans jpg/resample noise without a fixed color budget."""
    m = img[..., 3] >= ALPHA_CUT
    cnt = Counter(map(tuple, img[m][:, :3].tolist()))
    keep, remap = [], {}
    for c, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        hit = next((k for k in keep if sum((a - b) ** 2 for a, b in zip(c, k)) <= dist * dist), None)
        if hit is None:
            keep.append(c); remap[c] = c
        else:
            remap[c] = hit
    out = img.copy()
    px = out[m][:, :3]
    uniq, inv = np.unique(px, axis=0, return_inverse=True)
    lut = np.array([remap[tuple(u.tolist())] for u in uniq], np.uint8)
    out[m, :3] = lut[inv.reshape(-1)]
    return out

def to_native(img):
    """-> (native, info). integer upscale is exact; otherwise fit a fractional grid."""
    s, mm = detect_scale(img)
    if s > 1 or max(img.shape[:2]) <= 400:
        if s == 1 and min(img.shape[:2]) >= 24:     # non-integer upscale? (runs of 3,4,3,4…)
            px, _, sx = _axis_pitch(img, 1, hi=min(40.0, img.shape[1] / 6))
            py, _, sy = _axis_pitch(img, 0, hi=min(40.0, img.shape[0] / 6))
            if min(sx, sy) >= OFFGRID_FIT and min(px, py) >= 1.5:
                return img.copy(), {"method": "off-grid", "pitch": [round(px, 3), round(py, 3)],
                                    "fit": [round(sx, 3), round(sy, 3)], "grid_mismatch": None}
        return img[::s, ::s].copy(), {"method": "integer", "pitch": [s, s], "grid_mismatch": round(mm, 4)}
    px, ox, sx = _axis_pitch(img, 1); py, oy, sy = _axis_pitch(img, 0)
    if min(sx, sy) < 0.5 or px < 2 or py < 2:
        return img.copy(), {"method": "none", "pitch": [1, 1], "grid_mismatch": 0.0, "fit": [sx, sy]}
    nat, xs, ys = _sample_grid(img, px, ox, py, oy)
    return nat, {"method": "fractional", "pitch": [round(px, 3), round(py, 3)],
                 "fit": [round(sx, 3), round(sy, 3)], "grid_mismatch": None}


# ---------- primitives ----------
def components(mask, conn):
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    nb = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if conn == 8:
        nb += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not seen[y, x]:
                seen[y, x] = True; st = [(y, x)]; pts = []
                while st:
                    cy, cx = st.pop(); pts.append((cy, cx))
                    for dy, dx in nb:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; st.append((ny, nx))
                out.append(pts)
    return out

def mask_of(pts, shape):
    m = np.zeros(shape, bool)
    for y, x in pts:
        m[y, x] = True
    return m

def adj_pairs(key, sel):
    """yield (ya,xa,yb,xb) for 4-adjacent px both in sel with different keys."""
    h, w = key.shape
    for (dy, dx) in ((0, 1), (1, 0)):
        A, B = key[:h-dy, :w-dx], key[dy:, dx:]
        ok = sel[:h-dy, :w-dx] & sel[dy:, dx:] & (A != B)
        for y, x in zip(*np.nonzero(ok)):
            yield y, x, y + dy, x + dx

LIGHT_VEC = {"top-left": (-1, -1), "top-right": (1, -1), "bottom-left": (-1, 1), "bottom-right": (1, 1)}


# ---------- analysis ----------
def same_color_components(key, sel):
    """4-connected same-color blobs inside sel. returns label array + sizes."""
    H, W = key.shape
    lab = -np.ones((H, W), np.int64); sizes = []
    for y in range(H):
        for x in range(W):
            if sel[y, x] and lab[y, x] < 0:
                k = key[y, x]; idx = len(sizes); lab[y, x] = idx; st = [(y, x)]; n = 0
                while st:
                    cy, cx = st.pop(); n += 1
                    for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                        if 0 <= ny < H and 0 <= nx < W and sel[ny, nx] and lab[ny, nx] < 0 and key[ny, nx] == k:
                            lab[ny, nx] = idx; st.append((ny, nx))
                sizes.append(n)
    return lab, np.array(sizes, np.int64)

def line_rhythm(lab, sizes, shape, min_px=20):
    """stair-step rhythm of shape edges. for every same-color region >= min_px, read its
    top/bottom/left/right edge profile; inside each line stretch (steps of <= 1px) the run
    lengths must flow (1-1-1, 2-2-2, 1-2-1-2, 1-2-3). a run longer/shorter than BOTH
    neighbors by > 1 is a spike (1-3-1) = jaggy. returns (runs, spike_ratio, mask)."""
    H, W = shape; mask = np.zeros(shape, bool); total = spikes = 0
    for idx in np.nonzero(sizes >= min_px)[0]:
        ys, xs = np.nonzero(lab == idx); y0, x0 = ys.min(), xs.min()
        m = np.zeros((ys.max() - y0 + 1, xs.max() - x0 + 1), bool); m[ys - y0, xs - x0] = True
        h, w = m.shape
        views = ((m, lambda e, j: (e, j)), (m[::-1], lambda e, j: (h - 1 - e, j)),
                 (m.T, lambda e, j: (j, e)), (m.T[::-1], lambda e, j: (j, w - 1 - e)))
        for mm, back in views:
            prof = [int(np.argmax(c)) if c.any() else None for c in mm.T]
            seg = []
            for j, v in enumerate(prof + [None]):
                if v is None or (seg and abs(v - seg[-1][1]) > 1):
                    if len(seg) >= 4:
                        runs = []; start = 0
                        for i in range(1, len(seg) + 1):
                            if i == len(seg) or seg[i][1] != seg[i - 1][1]:
                                runs.append((start, i)); start = i
                        inner = runs[1:-1]                       # end runs are partial
                        for i in range(1, len(inner) - 1):
                            a_, b_, c_ = (r[1] - r[0] for r in inner[i - 1:i + 2])
                            total += 1
                            if (b_ - a_ > 1 and b_ - c_ > 1) or (a_ - b_ > 1 and c_ - b_ > 1):
                                spikes += 1
                                for q in range(*inner[i]):
                                    jj, ee = seg[q]; yy, xx = back(ee, jj); mask[y0 + yy, x0 + xx] = True
                    seg = [] if v is None else [(j, v)]
                else:
                    seg.append((j, v))
    return total, (spikes / total if total else 0.0), mask

def analyze(img, outline_max_l=None, light=None, mode=None):
    """all measurements + per-check masks for one native-res RGBA image.
    mode: 'sprite' (has transparency) or 'scene' (full-bleed); auto if None."""
    H, W = img.shape[:2]
    m = img[..., 3] >= ALPHA_CUT
    mode = mode or ("sprite" if (~m).any() else "scene")
    rgb = img[..., :3].astype(np.int64)
    key = np.where(m, (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2], -1)
    counts = Counter(key[m].tolist())
    n_op = int(m.sum())
    R = {"mode": mode, "size": [W, H], "opaque_px": n_op, "n_colors": len(counts),
         "colors": {hx(k2rgb(k)): n for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))}}
    M = {}

    # --- artifacts ---
    a = img[..., 3]
    M["partial_alpha"] = (a > 0) & (a < 255)

    pad = np.pad(key, 1, constant_values=-2)
    same = np.zeros_like(m)
    for n in (pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]):
        same |= (n == key)
    M["orphans"] = m & ~same
    R["orphan_ratio"] = round(M["orphans"].sum() / n_op, 4) if n_op else 0.0
    # foreign orphan = 1px dot whose color is unrelated (not analogous) to ALL 4 neighbors.
    # highlights/details belong to a neighbor's ramp; strays from another material don't.
    fo = np.zeros_like(m)
    acache = {}
    def rel(a_, b_):
        kk = (a_, b_) if a_ < b_ else (b_, a_)
        if kk not in acache:
            acache[kk] = analogous(k2rgb(a_), k2rgb(b_))
        return acache[kk]
    for y, x in zip(*np.nonzero(M["orphans"])):
        k0 = int(key[y, x]); nb_ = []
        for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= ny < H and 0 <= nx < W and key[ny, nx] >= 0:
                nb_.append(int(key[ny, nx]))
        if len(nb_) >= 3 and not any(rel(k0, k1) for k1 in nb_):
            fo[y, x] = True
    M["foreign_orphans"] = fo
    R["foreign_orphan_ratio"] = round(float(fo.sum() / max(n_op, 1)), 4)

    comps = sorted(components(m, 8), key=len, reverse=True)
    isl = [c for c in comps[1:] if len(c) <= ISLAND_MAX_PX]
    M["islands"] = mask_of([p for c in isl for p in c], m.shape)
    R["islands"] = len(isl)

    holes = [c for c in components(~m, 4)
             if len(c) <= PINHOLE_MAX_PX and not any(y in (0, H-1) or x in (0, W-1) for y, x in c)]
    M["pinholes"] = mask_of([p for c in holes for p in c], m.shape)
    R["pinholes"] = len(holes)

    # near-dup artifact = a RARE color sitting within NEAR_DUP_DIST of a much more
    # common one (AA / compression residue). deliberate close shades are both common.
    rare_max = max(3, int(n_op * RARE_SHARE))
    used = sorted(counts, key=lambda k: (-counts[k], k))
    dups, flagged = [], []
    for kb in used:
        if counts[kb] > rare_max:
            continue
        for ka in used:
            if counts[ka] >= 10 * counts[kb] and \
               np.linalg.norm(np.subtract(k2rgb(ka), k2rgb(kb))) < NEAR_DUP_DIST:
                dups.append([hx(k2rgb(ka)), hx(k2rgb(kb))]); flagged.append(kb); break
    M["near_dup"] = np.isin(key, flagged) & m
    R["near_dup_pairs"] = dups[:20]
    R["near_dup_colors"] = len(flagged)

    # grain: neighboring px that differ, but by an invisible amount (noise filter, jpg, AI)
    a3 = img[..., :3].astype(np.int32); tot_d = gr = 0; gm = np.zeros_like(m)
    _u, _inv = np.unique(a3.reshape(-1, 3), axis=0, return_inverse=True)
    Lgr = np.array([lstar(u) for u in _u])[_inv.reshape(-1)].reshape(H, W)
    for dy, dx in ((0, 1), (1, 0)):
        A_, B_ = a3[:H-dy, :W-dx], a3[dy:, dx:]
        both = m[:H-dy, :W-dx] & m[dy:, dx:]
        dist = np.sqrt(((A_ - B_) ** 2).sum(-1))
        darkpair = (lum_a := np.minimum(Lgr[:H-dy, :W-dx], Lgr[dy:, dx:])) < MUDDY_MIN_L  # subtle dark atmosphere ok
        diff = both & (dist > 0) & ~darkpair; g_ = diff & (dist < GRAIN_DIST)
        tot_d += int(diff.sum()); gr += int(g_.sum())
        gm[:H-dy, :W-dx] |= g_; gm[dy:, dx:] |= g_
    M["grain"] = gm
    R["grain_ratio"] = round(gr / max(tot_d, 1), 4)

    # speckle: px in same-color blobs of 1-2 px (noise; stars/glints count too)
    lab, sizes = same_color_components(key, m)
    small = np.isin(lab, np.nonzero(sizes <= 2)[0]) & m
    M["speckle"] = small
    if mode == "scene":
        R["line_runs"], lr, M["line_spikes"] = line_rhythm(lab, sizes, m.shape)
        R["line_spike_ratio"] = round(lr, 4)
    else:
        R["line_runs"], R["line_spike_ratio"], M["line_spikes"] = 0, 0.0, np.zeros_like(m)
    R["speckle_ratio"] = round(float(small.sum() / max(n_op, 1)), 4)

    # --- lines ---
    padm = np.pad(m, 1, constant_values=False)
    edge = m & ((~padm[:-2, 1:-1]) | (~padm[2:, 1:-1]) | (~padm[1:-1, :-2]) | (~padm[1:-1, 2:]))
    ec = Counter(key[edge].tolist())
    dom = sorted(ec.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if ec else None
    R["edge_color"] = hx(k2rgb(dom)) if dom is not None else None
    # outline = a DARK BAND on the silhouette edge, not one hex: good sprites use
    # tinted near-blacks (selout) — #000000, #1c091c, #240b0a … all count.
    uniq_k = np.unique(key[m]) if n_op else np.array([], np.int64)
    Lk = {int(k): lstar(k2rgb(int(k))) for k in uniq_k}
    Lmap = np.zeros(key.shape)
    for k_, v_ in Lk.items():
        Lmap[key == k_] = v_
    R["edge_lstar_median"] = round(float(np.median(Lmap[edge])), 2) if edge.any() else None
    ol_max = outline_max_l if outline_max_l is not None else OUTLINE_MAX_L
    if mode == "scene":
        is_ol = np.zeros_like(m)
    else:
        is_ol = edge & (Lmap <= ol_max)
    R["outline_coverage"] = round(float(is_ol.sum() / max(edge.sum(), 1)), 4)
    M["outline_breaks"] = edge & ~is_ol if mode == "sprite" else np.zeros_like(m)

    darkm = m & (Lmap <= ol_max) if mode == "sprite" else np.zeros_like(m)
    blk = darkm[:-1, :-1] & darkm[1:, :-1] & darkm[:-1, 1:] & darkm[1:, 1:]
    blk &= edge[:-1, :-1] | edge[1:, :-1] | edge[:-1, 1:] | edge[1:, 1:]
    dm = np.zeros_like(m)
    for y, x in zip(*np.nonzero(blk)):
        dm[y:y+2, x:x+2] = True
    M["outline_doubles"] = dm
    R["doubles_ratio"] = round(float(blk.sum() / max(is_ol.sum(), 1)), 4)

    # --- shading: ramps = local shade chains. from each color, step to the analogous
    # neighbor it touches most that is lighter (up) / darker (down). no transitive
    # merging, so a whole blue scene doesn't collapse into one giant "ramp".
    fill = m & ~is_ol
    n_fill = max(int(fill.sum()), 1)
    L = {k: lstar(k2rgb(k)) for k in counts}
    touch = Counter()
    for ya, xa, yb, xb in adj_pairs(key, fill):
        touch[tuple(sorted((int(key[ya, xa]), int(key[yb, xb]))))] += 1
    nbrs = {}
    for (ka, kb), n in touch.items():
        if n >= 2 and analogous(k2rgb(ka), k2rgb(kb)):
            nbrs.setdefault(ka, []).append((n, kb)); nbrs.setdefault(kb, []).append((n, ka))
    def step(k, up):
        c = [(n, -j) for n, j in nbrs.get(k, []) if (L[j] > L[k] if up else L[j] < L[k])]
        return -max(c)[1] if c else None          # most contacts, ties -> lowest key
    def walk(k, up):
        out = []; cur = step(k, up)
        while cur is not None and len(out) < 64:
            out.append(cur); cur = step(cur, up)
        return out
    upc = {k: walk(k, True) for k in counts}
    fill_counts = Counter(key[fill].tolist())
    major_k = sorted(k for k, n in fill_counts.items() if n / n_fill >= MAJOR_RAMP_SHARE[mode] / 2)
    chains = {}
    for k in major_k:
        ch = tuple(walk(k, False)[::-1] + [k] + upc[k])
        chains.setdefault(ch, []).append(k)
    ramps, muddy_px, short_px, n_steps, n_steps_lit, n_mud = [], [], [], 0, 0, 0
    for ch, owners in sorted(chains.items(), key=lambda kv: (-sum(fill_counts[o] for o in kv[1]), kv[0])):
        steps = [round(L[b] - L[a], 1) for a, b in zip(ch, ch[1:])]
        hues = [hsv(k2rgb(k))[0] for k in ch if not is_grey(k2rgb(k))]
        d = [((b - a + 180) % 360) - 180 for a, b in zip(hues, hues[1:])]
        d = [x for x in d if abs(x) >= 3]
        ramps.append({"colors": [hx(k2rgb(k)) for k in ch], "for": [hx(k2rgb(o)) for o in owners],
                      "share": round(sum(fill_counts[o] for o in owners) / n_fill, 3), "major": True,
                      "steps": steps,
                      "hue_shift_consistent": not (any(x > 0 for x in d) and any(x < 0 for x in d))})
        if len(ch) < MIN_RAMP_LEN:
            short_px += owners
        n_steps += len(steps)
        for b, st in zip(ch[1:], steps):
            if L[ch[ch.index(b) - 1]] < MUDDY_MIN_L:
                continue
            n_steps_lit += 1
            if st < MIN_RAMP_STEP:
                n_mud += 1; muddy_px.append(b)
    R["ramps"] = ramps
    R["short_ramp_share"] = round(sum(fill_counts[k] for k in short_px) / n_fill, 4)
    R["muddy_ratio"] = round(n_mud / n_steps_lit, 4) if n_steps_lit else 0.0
    R["muddy_steps"] = n_mud
    M["ramp_short"] = np.isin(key, short_px) & fill
    M["ramp_muddy"] = np.isin(key, muddy_px) & fill

    # jumps: two analogous colors touch but the darker one's chain has shades between them
    jm, pairs, jumps = np.zeros_like(m), 0, 0
    for ya, xa, yb, xb in adj_pairs(key, fill):
        a_, b_ = int(key[ya, xa]), int(key[yb, xb])
        if not analogous(k2rgb(a_), k2rgb(b_)):
            continue
        lo_, hi_ = (a_, b_) if L[a_] <= L[b_] else (b_, a_)
        pairs += 1
        ch = upc[lo_]
        if hi_ in ch and ch.index(hi_) >= 1:
            jumps += 1; jm[ya, xa] = jm[yb, xb] = True
    M["shading_jumps"] = jm
    R["jump_ratio"] = round(jumps / pairs, 4) if pairs else 0.0

    # light direction from luminance-weighted centroid
    R["light_from"], sx, sy = "unknown", 0, 0
    if n_op:
        ys, xs = np.nonzero(m)
        L = np.array([lum(c) for c in img[m][:, :3]])
        dy = float(((ys - ys.mean()) * L).sum() / max(L.sum(), 1))
        dx = float(((xs - xs.mean()) * L).sum() / max(L.sum(), 1))
        if abs(dx) + abs(dy) > 0.1:
            sx, sy = (-1 if dx < 0 else 1), (-1 if dy < 0 else 1)
            R["light_from"] = ("top" if sy < 0 else "bottom") + "-" + ("left" if sx < 0 else "right")

    # pillow shading: rim evenly darker than core, regardless of light side
    if mode == "scene":
        R["light_from"] = "unknown"   # centroid heuristic is meaningless on full scenes
    R["pillow"], M["pillow"] = None, np.zeros_like(m)
    lv = LIGHT_VEC.get(light or R["light_from"])
    if lv and fill.any() and mode == "sprite":
        padf = np.pad(fill, 1, constant_values=False)
        inner = fill & padf[:-2, 1:-1] & padf[2:, 1:-1] & padf[1:-1, :-2] & padf[1:-1, 2:]
        rim = fill & ~inner
        if inner.any() and rim.any():
            ys, xs = np.nonzero(m); cy, cx = ys.mean(), xs.mean()
            Ly = np.vectorize(lambda k: lum(k2rgb(k)) if k >= 0 else 0.0)(key)
            yy, xx = np.mgrid[0:H, 0:W]
            lit = ((xx - cx) * lv[0] + (yy - cy) * lv[1]) > 0
            lr, sr = rim & lit, rim & ~lit
            if lr.any() and sr.any():
                core, litm, shm = float(Ly[inner].mean()), float(Ly[lr].mean()), float(Ly[sr].mean())
                R["pillow"] = bool(litm < core - PILLOW_DARK and abs(litm - shm) < PILLOW_EVEN)
                R["pillow_lum"] = {"core": round(core, 1), "lit_rim": round(litm, 1), "shadow_rim": round(shm, 1)}
                if R["pillow"]:
                    M["pillow"] = lr

    # dithering: 2x2 checker windows
    k = key
    chk = (k[:-1, :-1] == k[1:, 1:]) & (k[:-1, 1:] == k[1:, :-1]) & \
          (k[:-1, :-1] != k[:-1, 1:]) & (k[:-1, :-1] >= 0) & (k[:-1, 1:] >= 0)
    R["dither_ratio"] = round(float(chk.mean()), 4) if chk.size else 0.0
    return R, M


# ---------- palette mapping ----------
def nearest_idx(px, pal):
    d = ((px[:, None, :].astype(np.int32) - pal[None, :, :].astype(np.int32)) ** 2).sum(-1)
    return d.argmin(1), np.sqrt(d.min(1))

def hkey(h):
    c = unhx(h); return (c[0] << 16) | (c[1] << 8) | c[2]


# ---------- ref prep ----------
def prepare(path):
    """load -> native grid -> classify. returns (native_img, info).
    kind: clean     = lossless, on a grid, <= MAX_CLEAN_COLORS  -> calibrates thresholds
          lossy     = jpg / needed color merging                 -> style only
          painterly = clean grid but hundreds of colors          -> style only"""
    nat, info = to_native(load(path))
    n0 = len({tuple(x) for x in nat[nat[..., 3] >= ALPHA_CUT][:, :3].tolist()})
    jpg = Path(path).suffix.lower() in (".jpg", ".jpeg")
    if jpg or (n0 > MAX_CLEAN_COLORS and info["method"] != "integer"):
        nat = merge_close(nat, 10); kind = "lossy"
    elif n0 > MAX_CLEAN_COLORS:
        kind = "painterly"
    else:
        kind = "clean"
    info.update({"kind": kind, "raw_colors": n0})
    return nat, info

def zigzag_ratio(ramps):
    return round(sum(not r["hue_shift_consistent"] for r in ramps) / len(ramps), 4) if ramps else 0.0


# ---------- palette files ----------
def load_palette(path):
    """.hex/.txt (one hex per line, '#' optional, Lospec format), .json {"palette": [...]},
    or .png swatch (its unique opaque colors, in first-seen order). -> list of '#rrggbb'."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        d = json.loads(p.read_text()); cols = d["palette"] if isinstance(d, dict) else d
    elif p.suffix.lower() in (".png", ".gif"):
        a = load(p); seen, cols = set(), []
        for px in a[a[..., 3] >= ALPHA_CUT][:, :3].tolist():
            if tuple(px) not in seen:
                seen.add(tuple(px)); cols.append(hx(px))
    else:
        cols = [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith(";")]
    out = []
    for c in cols:
        c = c.strip().lstrip("#").lower()
        if len(c) != 6 or any(ch not in "0123456789abcdef" for ch in c):
            raise ValueError(f"bad hex in palette {path}: {c!r}")
        out.append("#" + c)
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate colors in palette {path}")
    return out

def group_ramps(cols):
    """palette -> ramps. darks (outline near-blacks), neutrals, and hue families whose
    total hue span is <= ANALOG_HUE (no transitive chaining red->orange->yellow).
    each sorted dark->light by L*. deterministic."""
    rgb = [unhx(c) for c in cols]
    darks = sorted([c for c in rgb if is_grey(c) and lstar(c) < MUDDY_MIN_L], key=lambda c: (lstar(c), c))
    neut = sorted([c for c in rgb if is_grey(c) and lstar(c) >= MUDDY_MIN_L], key=lambda c: (lstar(c), c))
    chrom = sorted([c for c in rgb if not is_grey(c)], key=lambda c: (hsv(c)[0], c))
    if chrom:                           # start the wheel at the widest hue gap so no family wraps
        hs = [hsv(c)[0] for c in chrom]
        gaps = [((hs[(i + 1) % len(hs)] - hs[i]) % 360, i) for i in range(len(hs))]
        start = (max(gaps)[1] + 1) % len(hs)
        chrom = chrom[start:] + chrom[:start]
    fams = []
    for c in chrom:
        if fams and hue_diff(fams[-1][0], c) <= ANALOG_HUE:
            fams[-1].append(c)
        else:
            fams.append([c])
    ramps = [sorted(f, key=lambda c: (lstar(c), c)) for f in fams]
    ramps += [g for g in (neut, darks) if g]
    return [[hx(c) for c in r] for r in ramps]

def cmd_palette(images, out, n, min_share):
    """moodboard (MJ renders, photos, anything) -> strict hex palette. deterministic:
    box-downscale, median-cut, merge near-dups, drop rare, group into ramps."""
    acc = Counter()
    for f in images:
        im = Image.open(f).convert("RGB")
        im.thumbnail((256, 256), Image.BOX)
        q = im.quantize(colors=n, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        pal = q.getpalette()[:n * 3]
        for idx, cnt in sorted(Counter(np.array(q).ravel().tolist()).items()):
            acc[tuple(pal[idx*3:idx*3+3])] += cnt / (im.width * im.height) / len(images)
    keep = []
    for c, sh in sorted(acc.items(), key=lambda kv: (-kv[1], kv[0])):
        if sh < min_share:
            continue
        hit = next((k for k in keep if np.linalg.norm(np.subtract(c, k[0])) < NEAR_DUP_DIST * 1.5), None)
        if hit: hit[1] += sh
        else: keep.append([c, sh])
    cols = [hx(c) for c, _ in keep]
    ramps = group_ramps(cols)
    flat = [c for r in ramps for c in r]
    Path(out).write_text("\n".join(c.lstrip("#") for c in flat) + "\n")
    sw = np.zeros((len(ramps) * 16, max(len(r) for r in ramps) * 16, 4), np.uint8)
    for j, r in enumerate(ramps):
        for i, c in enumerate(r):
            sw[j*16:(j+1)*16, i*16:(i+1)*16] = (*unhx(c), 255)
    swp = Path(out).with_suffix(".png"); Image.fromarray(sw).save(swp)
    print(f"wrote {out} ({len(flat)} colors, {len(ramps)} ramps) + swatch {swp}")
    for r in ramps:
        flag = "" if len(r) >= MIN_RAMP_LEN else f"   <-- only {len(r)} shade(s): ask before adding colors"
        print("  ramp:", " ".join(r) + flag)
    return flat


# ---------- extract ----------
def calibrate(cal, line_cal=None):
    """thresholds = worst value seen in good refs + small margin. per mode."""
    def mx(k, margin=0.0):
        """worst ref + max(abs margin, half the spread above the median): heavy-tailed
        metrics get proportionally more room, so an unseen-but-good ref still passes"""
        v = sorted(p[k] for p in cal); med = v[len(v) // 2]
        return round(v[-1] + max(margin, 0.5 * (v[-1] - med)), 6)
    mode = cal[0]["mode"]
    lights = Counter(p["light_from"] for p in cal if p["light_from"] != "unknown")
    r = {
        "n_refs": len(cal),
        "size_w": [min(p["size"][0] for p in cal), max(p["size"][0] for p in cal)],
        "size_h": [min(p["size"][1] for p in cal), max(p["size"][1] for p in cal)],
        "max_colors": max(p["n_colors"] for p in cal),
        "max_orphan_ratio": mx("orphan_ratio", ORPHAN_MARGIN),
        "max_foreign_orphan_ratio": mx("foreign_orphan_ratio", 0.01),
        "max_speckle_ratio": mx("speckle_ratio", ORPHAN_MARGIN),
        "max_grain_ratio": mx("grain_ratio", 0.01),
        "max_rare_dup_rate": mx("rare_dup_rate"),
        "allow_islands": mode == "scene" or any(p["islands"] for p in cal),
        "allow_pinholes": mode == "scene" or any(p["pinholes"] for p in cal),
        "min_ramp_len": MIN_RAMP_LEN,
        "max_short_ramp_share": mx("short_ramp_share", 0.02),
        "min_ramp_step_lstar": MIN_RAMP_STEP,
        "max_muddy_ratio": mx("muddy_ratio", 0.05),
        "max_jump_ratio": mx("jump_ratio", 0.1 if mode == "sprite" else 0.05),
        "max_hue_zigzag_ratio": mx("hue_zigzag_ratio", 0.1),
        "uses_dither": max(p["dither_ratio"] for p in cal) > 0.01,
        "light_from": lights.most_common(1)[0][0] if lights else "unknown",
    }
    if mode == "scene":
        lc = [p for p in (line_cal or cal) if p["line_runs"] >= LINE_MIN_RUNS]
        if lc:
            v = sorted(p["line_spike_ratio"] for p in lc); med = v[len(v) // 2]
            r["max_line_spike_ratio"] = round(v[-1] + max(0.01, 0.5 * (v[-1] - med)), 4)
            r["line_calibrated_from"] = [p["file"] for p in lc]
    if mode == "sprite":
        r.update({
            "outline_max_lstar": round(min(OUTLINE_MAX_L + 10, max(p["edge_lstar_median"] for p in cal) + 5), 2),
            "min_outline_coverage": round(min(p["outline_coverage"] for p in cal) - 0.05, 4),
            "max_doubles_ratio": mx("doubles_ratio", 0.03),
        })
    return r

def cmd_extract(refs, out, lock=None, calibrate_all=False, palette_file=None, line_refs=None):
    per = []
    for r in refs:
        nat, info = prepare(r)
        R, _ = analyze(nat)
        R.update({"file": Path(r).name, **info, "hue_zigzag_ratio": zigzag_ratio(R["ramps"]),
                  "rare_dup_rate": round(R["near_dup_colors"] / max(R["opaque_px"], 1), 6)})
        per.append(R)
    usable = [p for p in per if calibrate_all or p["kind"] == "clean"]
    line_names = {Path(x).name for x in (line_refs or [])}
    missing = line_names - {p["file"] for p in per}
    if missing:
        raise SystemExit(f"--line-refs not among refs: {sorted(missing)}")
    rules = {}
    for mode in ("sprite", "scene"):
        cal = [p for p in usable if p["mode"] == mode]
        if cal:
            lcal = cal + [p for p in per if p["file"] in line_names and p["mode"] == mode and p not in cal]
            rules[mode] = calibrate(cal, lcal)
    if not rules:
        raise SystemExit("no clean refs to calibrate from (use --calibrate-all to force)")
    lock = [Path(x).name for x in (lock or [])]
    locked = [p for p in per if p["file"] in lock]
    if lock and len(locked) != len(lock):
        raise SystemExit(f"--lock refs not found: {set(lock) - {p['file'] for p in locked}}")
    if palette_file:
        palette, pal_src, tol = load_palette(palette_file), Path(palette_file).name, 0
    elif locked:
        pc = Counter()
        for p in locked:
            pc.update(p["colors"])
        palette = [c for c, _ in sorted(pc.items(), key=lambda kv: (-kv[1], kv[0]))]
        pal_src, tol = "refs:" + ",".join(lock), 0 if all(p["kind"] == "clean" for p in locked) else 24
    else:
        palette, pal_src, tol = [], None, 0
    style = {
        "version": 4,
        "palette": palette, "palette_source": pal_src, "palette_tolerance": tol,
        "palette_ramps": group_ramps(palette) if palette else [],
        "example_ramps": {m: [p["ramps"][0]["colors"] for p in usable if p["mode"] == m and p["ramps"]][:6]
                          for m in rules},
        "rules": rules,
        "refs": [{k: v for k, v in p.items() if k not in ("colors", "near_dup_pairs", "ramps")} for p in per],
    }
    Path(out).write_text(json.dumps(style, indent=2))

    hdr = (f"{'ref':16s} {'mode':6s} {'kind':9s} {'colors':>6s} {'orph':>6s} {'speck':>6s} {'grain':>6s} "
           f"{'rdup':>4s} {'short':>6s} {'muddy':>6s} {'jump':>6s} {'olcov':>6s}")
    print(hdr); print("-" * len(hdr))
    for p in per:
        print(f"{p['file'][:16]:16s} {p['mode']:6s} {p['kind']:9s} {p['n_colors']:6d} {p['orphan_ratio']:6.3f} "
              f"{p['speckle_ratio']:6.3f} {p['grain_ratio']:6.3f} {p['near_dup_colors']:4d} "
              f"{p['short_ramp_share']:6.3f} {p['muddy_ratio']:6.3f} {p['jump_ratio']:6.3f} "
              f"{p['outline_coverage'] if p['mode'] == 'sprite' else 0:6.3f}"
              f"{'' if p in usable else ('   (lines only)' if p['file'] in line_names else '   (style only)')}")
    print(f"\nwrote {out}: rules for {', '.join(f'{m} ({rules[m][chr(110)+'_refs']} refs)' for m in rules)}; "
          f"palette: {pal_src or 'not locked'}")
    if palette:
        short = [r for r in style["palette_ramps"] if len(r) < MIN_RAMP_LEN]
        if short:
            print(f"warn: palette ramps with < {MIN_RAMP_LEN} shades: {short}")
    return style


# ---------- prompt ----------
def build_prompt(style, mode):
    if mode not in style["rules"]:
        raise SystemExit(f"no {mode} rules in style (have: {list(style['rules'])})")
    r = style["rules"][mode]
    L = ["PIXEL ART SPEC (hard rules, machine-checked by pixelkit). light direction, view, size: from the brief.",
         "- 1 pixel = 1 grid cell; flat colors only; alpha 0 or 255; no AA, blur, gradients, glow, noise/grain"]
    if style["palette"]:
        L.append("- palette LOCKED. use ONLY these hex codes, no others, no 'close enough':")
        for rp in style["palette_ramps"]:
            L.append(f"    ramp: {' '.join(rp)}")
    else:
        L.append(f"- limited palette (<= {r['max_colors']} colors); no near-identical stray shades")
    L += [f"- every material shaded with >= {r['min_ramp_len']} shades from ONE ramp, stepping 1 shade at a time",
          f"- lit-side ramp steps clearly distinct (>= {r['min_ramp_step_lstar']} L*); never dark next to light of same ramp"]
    if mode == "sprite":
        L += [f"- closed 1px outline of dark tinted near-blacks (L* <= {r['outline_max_lstar']}), "
              "tint toward the local color; never 2px thick",
              "- transparent background; no drop shadow baked in; no detached specks or 1px holes"]
    else:
        L.append("- full-bleed scene; atmosphere via ramps, not noise")
    if not style["palette"]:
        for ex in style["example_ramps"].get(mode, [])[:3]:
            L.append(f"- example ramp from refs (dark->light): {' '.join(ex)}")
    L.append("- dither: sparse checker only" if r["uses_dither"] else "- no dithering")
    return "\n".join(L)

def cmd_prompt(style_path, mode):
    print(build_prompt(json.loads(Path(style_path).read_text()), mode))


# ---------- check ----------
def audit(style, img, palette=None):
    """returns (ok, report, masks, native_img). picks sprite/scene rules by transparency.
    levels: FAIL blocks, WARN informs; non-applicable checks pass as n/a."""
    nat, info = to_native(img)
    mode = "sprite" if (nat[..., 3] < ALPHA_CUT).any() else "scene"
    sprite = mode == "sprite"
    if mode not in style["rules"]:
        raise SystemExit(f"image is a {mode} but style has no {mode} rules (have: {list(style['rules'])})")
    r = style["rules"][mode]
    light = r["light_from"] if (sprite and r["light_from"] != "unknown") else None
    R, M = analyze(nat, light=light, mode=mode)
    none = np.zeros(nat.shape[:2], bool); NA = "n/a"

    pal = palette if palette is not None else style["palette"]
    tol = 0 if palette is not None else style.get("palette_tolerance", 0)
    off, offm = [], none
    if pal and R["colors"]:
        P = np.array([unhx(c) for c in pal])
        _, dist = nearest_idx(np.array([unhx(c) for c in R["colors"]]), P)
        off = [c for c, d in zip(R["colors"], dist) if d > tol]
        key = nat[..., :3].astype(np.int64)
        key = (key[..., 0] << 16) | (key[..., 1] << 8) | key[..., 2]
        offm = np.isin(key, [hkey(h) for h in off]) & (nat[..., 3] >= ALPHA_CUT)

    w, h = R["size"]
    allowed_dups = max(1, int(np.ceil(r["max_rare_dup_rate"] * R["opaque_px"])))
    zz = zigzag_ratio(R["ramps"])
    lf, lw = R["light_from"], r["light_from"]
    light_lvl, light_ok, light_d = "WARN", True, f"{lf} vs {lw} (heuristic)"
    if not sprite:
        light_d = NA + " (scene)"
    elif "unknown" not in (lf, lw) and lf != lw:
        a_, b_ = lf.split("-"), lw.split("-")
        light_ok = False
    grid_ok = info["method"] == "integer" and info["grid_mismatch"] == 0
    sc = NA + " (scene)"

    C = [  # (group, name, level, pass, detail, mask)
        ("artifact", "on_grid", "FAIL", grid_ok, f"{info['method']} pitch {info['pitch']}", none),
        ("artifact", "no_partial_alpha", "FAIL", not M["partial_alpha"].any(),
         f"{int(M['partial_alpha'].sum())} soft-alpha px", M["partial_alpha"]),
        ("artifact", "no_grain", "FAIL", R["grain_ratio"] <= r["max_grain_ratio"],
         f"{R['grain_ratio']} <= {r['max_grain_ratio']} (neighbors differing invisibly)", M["grain"]),
        ("artifact", "no_near_dup_colors", "FAIL", R["near_dup_colors"] <= allowed_dups,
         f"{R['near_dup_colors']} rare near-dup colors <= {allowed_dups} {R['near_dup_pairs'][:3]}", M["near_dup"]),
        ("artifact", "no_islands", "FAIL", r["allow_islands"] or R["islands"] == 0,
         f"{R['islands']} detached specks" if sprite else sc, M["islands"]),
        ("artifact", "no_pinholes", "FAIL", r["allow_pinholes"] or R["pinholes"] == 0,
         f"{R['pinholes']} enclosed holes" if sprite else sc, M["pinholes"]),
        ("artifact", "stray_pixels", "FAIL", R["foreign_orphan_ratio"] <= r["max_foreign_orphan_ratio"],
         f"{R['foreign_orphan_ratio']} <= {r['max_foreign_orphan_ratio']} (1px dots unrelated to all neighbors)",
         M["foreign_orphans"]),
        ("artifact", "orphans", "WARN", R["orphan_ratio"] <= r["max_orphan_ratio"],
         f"{R['orphan_ratio']} <= {r['max_orphan_ratio']} (1px detail density; style-dependent)", M["orphans"]),
        ("artifact", "speckle", "WARN", R["speckle_ratio"] <= r["max_speckle_ratio"],
         f"{R['speckle_ratio']} <= {r['max_speckle_ratio']} (px in 1-2px blobs)", M["speckle"]),
        ("color", "on_palette", "FAIL", not off,
         f"{len(off)} off-palette {off[:5]}" if pal else NA + " (no locked palette)", offm),
        ("color", "color_count", "WARN", R["n_colors"] <= r["max_colors"],
         f"{R['n_colors']} <= {r['max_colors']}", none),
        ("line", "outline_dark", "FAIL", not sprite or (R["edge_lstar_median"] or 0) <= r["outline_max_lstar"],
         f"edge median L* {R['edge_lstar_median']} <= {r['outline_max_lstar']}" if sprite else sc, none),
        ("line", "outline_coverage", "FAIL", not sprite or R["outline_coverage"] >= r["min_outline_coverage"],
         f"{R['outline_coverage']} >= {r['min_outline_coverage']}" if sprite else sc, M["outline_breaks"]),
        ("line", "outline_doubles", "FAIL", not sprite or R["doubles_ratio"] <= r["max_doubles_ratio"],
         f"{R['doubles_ratio']} <= {r['max_doubles_ratio']}" if sprite else sc, M["outline_doubles"]),
        ("line", "line_rhythm", "FAIL",
         not ("max_line_spike_ratio" in r and R["line_runs"] >= LINE_MIN_RUNS) or R["line_spike_ratio"] <= r["max_line_spike_ratio"],
         (f"{R['line_spike_ratio']} <= {r['max_line_spike_ratio']} spiky stair-steps ({R['line_runs']} runs)"
          if "max_line_spike_ratio" in r and R["line_runs"] >= LINE_MIN_RUNS
          else NA + (f" ({R['line_runs']} runs < {LINE_MIN_RUNS})" if not sprite else " (sprite)")), M["line_spikes"]),
        ("shading", "ramp_length", "FAIL", R["short_ramp_share"] <= r["max_short_ramp_share"],
         f"{R['short_ramp_share']} of fill on ramps < {r['min_ramp_len']} shades (<= {r['max_short_ramp_share']})",
         M["ramp_short"]),
        ("shading", "ramp_contrast", "FAIL", R["muddy_ratio"] <= r["max_muddy_ratio"] or (R["muddy_steps"] < 2 and R["muddy_ratio"] < 0.5),
         f"{R['muddy_ratio']} of lit steps < {r['min_ramp_step_lstar']} L* (<= {r['max_muddy_ratio']})", M["ramp_muddy"]),
        ("shading", "ramp_hue_shift", "WARN", zz <= r["max_hue_zigzag_ratio"],
         f"{zz} of ramps zig-zag in hue (<= {r['max_hue_zigzag_ratio']})", none),
        ("shading", "shading_jumps", "FAIL", R["jump_ratio"] <= r["max_jump_ratio"],
         f"{R['jump_ratio']} <= {r['max_jump_ratio']} (skipped ramp steps)", M["shading_jumps"]),
        ("shading", "no_pillow", "WARN", not R["pillow"], str(R.get("pillow_lum", NA)) + " (heuristic)", M["pillow"]),
        ("shading", "light_direction", light_lvl, light_ok, light_d, none),
        ("size", "size_in_range", "WARN",
         r["size_w"][0] <= w <= r["size_w"][1] and r["size_h"][0] <= h <= r["size_h"][1],
         f"{w}x{h} vs w{r['size_w']} h{r['size_h']}", none),
    ]
    rep = {name: {"group": g, "level": lvl, "pass": bool(ok), "detail": d} for g, name, lvl, ok, d, _ in C}
    rep["_mode"] = {"group": "info", "level": "INFO", "pass": True, "detail": mode}
    masks = {name: mk for _, name, _, _, _, mk in C}
    ok = all(v["pass"] for v in rep.values() if v["level"] == "FAIL")
    return ok, rep, masks, nat

AREA_CHECKS = {"ramp_length", "ramp_contrast", "no_near_dup_colors", "on_palette", "no_grain"}

def mark(nat, rep, masks, scale=None):
    """white bg, image greyed. area-wide FAILs = magenta tint, point FAILs = solid
    magenta, WARN = yellow. point marks drawn last so they stay visible."""
    scale = scale or max(1, 512 // max(nat.shape[:2]))
    op = nat[..., 3:] > 0
    g = (nat[..., :3].astype(float).mean(-1, keepdims=True) * 0.6 + 40)
    out = np.where(op, g, 255.0).repeat(3, -1)
    mag = np.array([255, 0, 255.0])
    for name, v in rep.items():
        if name in masks and not v["pass"] and name in AREA_CHECKS:
            out[masks[name]] = out[masks[name]] * 0.5 + mag * 0.5
    for name, v in rep.items():
        if name in masks and not v["pass"] and name not in AREA_CHECKS:
            out[masks[name]] = mag if v["level"] == "FAIL" else (255, 220, 0)
    out = np.dstack([out.clip(0, 255).astype(np.uint8), np.full(nat.shape[:2], 255, np.uint8)])
    return out.repeat(scale, 0).repeat(scale, 1)

def cmd_check(style_path, img_path, as_json, mark_path, strict, palette_file=None):
    style = json.loads(Path(style_path).read_text())
    pal = load_palette(palette_file) if palette_file else None
    ok, rep, masks, nat = audit(style, load(img_path), pal)
    if strict:
        ok = all(v["pass"] for v in rep.values())
    if mark_path:
        Image.fromarray(mark(nat, rep, masks)).save(mark_path)
    if as_json:
        print(json.dumps({"pass": ok, "checks": rep}, indent=2))
    else:
        print(f"mode: {rep['_mode']['detail']}")
        for k, v in rep.items():
            if k.startswith("_"):
                continue
            tag = "PASS" if v["pass"] else v["level"]
            print(f"{tag:4s}  {v['group']:8s} {k:19s} {v['detail']}")
        print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------- split (sprite sheet -> transparent sprites) ----------
def strip_frame(nat, share=SPLIT_FRAME_SHARE):
    """peel edge rows/cols dominated by one color (frames, mats). returns (img, (y0,x0))."""
    y0, y1, x0, x1 = 0, nat.shape[0], 0, nat.shape[1]
    def dominated(line):
        c = Counter(map(tuple, line[:, :3].tolist()))
        return c.most_common(1)[0][1] / len(line) >= share
    changed = True
    while changed and y1 - y0 > 4 and x1 - x0 > 4:
        changed = False
        if dominated(nat[y0, x0:x1]): y0 += 1; changed = True
        if dominated(nat[y1-1, x0:x1]): y1 -= 1; changed = True
        if dominated(nat[y0:y1, x0]): x0 += 1; changed = True
        if dominated(nat[y0:y1, x1-1]): x1 -= 1; changed = True
    return nat[y0:y1, x0:x1], (y0, x0)

def bg_mask(panel, ring_px=2):
    """background+shadow = 4-connected flood from the panel edge through px that are
    (a) not darker than the outline wall, (b) not lighter than the bg, (c) a scaled
    version of the bg color (shadows = bg darkened), (d) then enclosed pockets made
    only of colors already classed as bg (e.g. gaps between spider legs)."""
    H, W = panel.shape[:2]
    rgb = panel[..., :3].astype(np.float64)
    ring = np.zeros((H, W), bool)
    ring[:ring_px] = ring[-ring_px:] = True; ring[:, :ring_px] = ring[:, -ring_px:] = True
    rc = rgb[ring]
    mu = np.median(rc, 0)
    k = (rgb * mu).sum(-1) / (mu * mu).sum()
    resid = np.sqrt(((rgb - k[..., None] * mu) ** 2).sum(-1))
    r95 = float(np.percentile(resid[ring], 95))
    uniq, inv = np.unique(panel[..., :3].reshape(-1, 3), axis=0, return_inverse=True)
    Ls = np.array([lstar(u) for u in uniq])[inv.reshape(-1)].reshape(H, W)
    Lbg = float(np.median(Ls[ring]))
    Lcap = float(np.percentile(Ls[ring], 95)) + 2
    Llo = float(np.percentile(Ls[ring], 5)) - 2
    # stage 1: the bg itself — tight color + lightness band
    bg_ok = (Ls >= Llo) & (Ls <= Lcap) & (resid < r95 * 1.5 + 4)
    # stage 2: shadows = bg under a tinted overlay -> constant per-channel ratio px/bg.
    # learn the modal ratio from darker px, accept only px near it (outlines differ in tint)
    ratio = rgb / np.maximum(mu, 1)
    dark = (Ls < Llo) & (Ls >= Lbg * SPLIT_WALL)
    sh_ok = np.zeros((H, W), bool)
    if dark.sum() >= 20:
        q = np.round(ratio[dark] / 0.05).astype(int)
        cnt = Counter(map(tuple, q.tolist())).most_common()
        modes = [np.array(c) * 0.05 for c, n in cnt[:SPLIT_SHADOW_MODES] if n >= cnt[0][1] * 0.25]
        for md in modes:          # several tints: shadow over light vs dark checker tiles
            sh_ok |= dark & (np.abs(ratio - md).max(-1) <= SPLIT_SHADOW_TOL)
    def flood(seed, ok):
        seen = seed.copy(); st = list(zip(*np.nonzero(seed)))
        while st:
            y, x = st.pop()
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= ny < H and 0 <= nx < W and ok[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; st.append((ny, nx))
        return seen
    seen = flood(ring & bg_ok, bg_ok)
    grown = np.zeros_like(seen)               # shadow px must touch stage-1 bg
    grown[1:] |= seen[:-1]; grown[:-1] |= seen[1:]; grown[:, 1:] |= seen[:, :-1]; grown[:, :-1] |= seen[:, 1:]
    seen |= flood(grown & sh_ok, sh_ok)
    seen = flood(seen, bg_ok | sh_ok)         # bg patches reachable through shadow
    ok = bg_ok | sh_ok
    key = (panel[..., 0].astype(np.int64) << 16) | (panel[..., 1].astype(np.int64) << 8) | panel[..., 2]
    plain = np.unique(panel[..., :3][seen & bg_ok], axis=0).astype(np.int32)
    allbg = np.unique(panel[..., :3][seen], axis=0).astype(np.int32)
    def near(px, cols):
        return np.sqrt(((px[:, None, :] - cols[None, :, :]) ** 2).sum(-1)).min(1) <= SPLIT_POCKET_DIST
    for comp in components(ok & ~seen, 4):          # enclosed pockets (bow string, key loops)
        if len(comp) < 3:
            continue
        px = np.array([panel[y, x, :3] for y, x in comp], np.int32)
        # must be mostly bg/shadow AND contain real plain bg (shadow-colored faces of
        # an object, e.g. slate ore, contain none)
        if near(px, allbg).mean() >= 0.9 and near(px, plain).mean() >= 0.3:
            for y, x in comp:
                seen[y, x] = True
    return seen

def split_sheet(img, min_px=SPLIT_MIN_PX):
    nat, info = to_native(img)
    panel, (oy, ox) = strip_frame(nat)
    bg = bg_mask(panel)
    H, W = panel.shape[:2]
    sprites, dropped = [], []
    comps = [c for c in components(~bg, 8) if len(c) >= 2]
    merged = True                      # pieces within SPLIT_MERGE_GAP px belong to one item
    while merged:
        merged = False
        bbs = [(min(y for y, _ in c), max(y for y, _ in c), min(x for _, x in c), max(x for _, x in c)) for c in comps]
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a, b = bbs[i], bbs[j]
                if a[0] - SPLIT_MERGE_GAP <= b[1] and b[0] - SPLIT_MERGE_GAP <= a[1] and \
                   a[2] - SPLIT_MERGE_GAP <= b[3] and b[2] - SPLIT_MERGE_GAP <= a[3] and \
                   max(len(comps[i]), len(comps[j])) >= min_px * 4:
                    comps[i] = comps[i] + comps[j]; del comps[j]; merged = True; break
            if merged:
                break
    for comp in sorted(comps, key=lambda c: (min(y for y, _ in c) // 8, min(x for _, x in c))):
        ys = [y for y, _ in comp]; xs = [x for _, x in comp]
        y0, y1, x0, x1 = min(ys), max(ys) + 1, min(xs), max(xs) + 1
        why = None
        if len(comp) < min_px: why = "tiny"
        elif y0 == 0 or x0 == 0 or y1 == H or x1 == W: why = "touches edge (signature/frame?)"
        if why:
            dropped.append({"bbox": [int(x0 + ox), int(y0 + oy), int(x1 - x0), int(y1 - y0)], "px": len(comp), "why": why})
            continue
        spr = np.zeros((y1 - y0 + 2, x1 - x0 + 2, 4), np.uint8)
        for y, x in comp:
            spr[y - y0 + 1, x - x0 + 1] = panel[y, x, :3].tolist() + [255]
        sprites.append((spr, [int(x0 + ox), int(y0 + oy), int(x1 - x0), int(y1 - y0)]))
    return sprites, dropped, info

def cmd_split(sheet, outdir, preview):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    img = load(sheet)
    sprites, dropped, info = split_sheet(img)
    stem = Path(sheet).stem
    for i, (spr, bb) in enumerate(sprites):
        Image.fromarray(spr).save(out / f"{stem}_{i:02d}.png")
        print(f"{stem}_{i:02d}.png  {spr.shape[1]-2}x{spr.shape[0]-2}  at {bb[:2]}")
    for d in dropped:
        print(f"dropped {d['px']}px at {d['bbox'][:2]}: {d['why']}")
    if preview:
        nat, _ = to_native(img)
        prev = np.full((nat.shape[0], nat.shape[1], 4), (255, 0, 255, 255), np.uint8)
        for spr, (x, y, w, h) in sprites:
            m = spr[1:-1, 1:-1, 3] > 0
            prev[y:y+h, x:x+w][m] = spr[1:-1, 1:-1][m]
        s = max(1, 512 // max(prev.shape[:2]))
        Image.fromarray(prev.repeat(s, 0).repeat(s, 1)).save(preview)
        print(f"preview -> {preview} (magenta = removed)")
    return sprites


# ---------- snap / render ----------
def snap(style, img, size, scale=1):
    W, H = size
    if not style["palette"]:
        raise SystemExit("snap needs a locked palette (extract --palette FILE or --lock REF)")
    pal = np.array([unhx(c) for c in style["palette"]])
    h, w = img.shape[:2]
    ys = np.arange(H + 1) * h // H; xs = np.arange(W + 1) * w // W
    out = np.zeros((H, W, 4), np.uint8)
    for j in range(H):
        for i in range(W):
            blk = img[ys[j]:ys[j+1], xs[i]:xs[i+1]].reshape(-1, 4)
            op = blk[blk[:, 3] >= ALPHA_CUT]
            if len(op) * 2 < len(blk):
                continue
            idx, _ = nearest_idx(op[:, :3], pal)
            out[j, i, :3] = pal[int(np.bincount(idx, minlength=len(pal)).argmax())]
            out[j, i, 3] = 255
    return out.repeat(scale, 0).repeat(scale, 1) if scale > 1 else out

def cmd_snap(style_path, img_path, out, size, scale):
    style = json.loads(Path(style_path).read_text())
    W, H = map(int, size.lower().split("x"))
    Image.fromarray(snap(style, load(img_path), (W, H), scale)).save(out)
    print(f"wrote {out}")

def parse_grid(text):
    rows = [ln.split() for ln in text.strip().splitlines() if ln.strip()]
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError(f"ragged grid: row widths {sorted(set(len(r) for r in rows))}")
    out = np.zeros((len(rows), w, 4), np.uint8)
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            if c != ".":
                out[y, x] = (*unhx(c), 255)
    return out

def cmd_render(grid_path, out, scale):
    img = parse_grid(Path(grid_path).read_text())
    Image.fromarray(img.repeat(scale, 0).repeat(scale, 1)).save(out)
    print(f"wrote {out} ({img.shape[1]}x{img.shape[0]} native, x{scale})")


# ---------- brief (never assume: every question answered before generating) ----------
BRIEF = [  # (field, question, applies to kinds, validator)
    ("subject",       "What exactly is it? Must-have details? Anything it must NOT look like?", "*", None),
    ("use",           "Where is it used? (in-world tile, item icon, card art, UI, background…) Which game?", "*", None),
    ("kind",          "sprite | tile | scene", "*", "kind"),
    ("native_size",   "Native canvas in px, WxH (e.g. 32x32). Not the display size.", "*", "size"),
    ("display_scale", "Integer scale per breakpoint, comma list (e.g. mobile 3, desktop 6 -> \"3,6\"). Nearest-neighbor only.", "*", "ints"),
    ("palette",       "Path to locked palette file (.hex/.png/.json). No file = ask for the moodboard first.", "*", "palette"),
    ("view",          "Camera: top-down | side | 3/4 | isometric", "*", "view"),
    ("light_from",    "Light direction: top-left | top-right | bottom-left | bottom-right", "*", "light"),
    ("outline",       "dark selout (tinted near-blacks) | single color hex | none", "sprite,tile", None),
    ("background",    "transparent | solid hex", "sprite", None),
    ("shadow",        "Drop shadow baked in? (usually no: engine draws it)", "sprite", "yesno"),
    ("animation",     "static, or N frames per action (list actions)", "sprite", None),
    ("tiling",        "Which edges must tile seamlessly, and with which neighbor tiles?", "tile", None),
    ("variants",      "How many variants (to break repetition)? Transitions needed (grass->dirt…)?", "tile", None),
    ("refs",          "Which refs to match most closely, and what to take from each?", "*", None),
    ("deliverable",   "Files: native png, scaled png(s), sheet layout, naming", "*", None),
]
_EMPTY = {"", "?", "tbd", "todo", "unknown", "idk"}

def cmd_brief_new(out):
    d = {f: "" for f, *_ in BRIEF}
    d["_questions"] = {f: q for f, q, *_ in BRIEF}
    Path(out).write_text(json.dumps(d, indent=2)); print(f"wrote {out}: fill every field, then `brief check`")

def brief_problems(d, base=Path(".")):
    probs = []
    kind = str(d.get("kind", "")).strip().lower()
    for f, q, kinds, val in BRIEF:
        if kinds != "*" and kind not in kinds.split(","):
            continue
        v = str(d.get(f, "")).strip()
        if v.lower() in _EMPTY:
            probs.append((f, "unanswered", q)); continue
        try:
            if val == "kind" and v.lower() not in ("sprite", "tile", "scene"): raise ValueError("sprite|tile|scene")
            if val == "size":
                w, h = map(int, v.lower().split("x"))
                if w < 4 or h < 4: raise ValueError("too small")
            if val == "ints":
                xs = [int(x) for x in v.replace(" ", "").split(",")]
                if not xs or min(xs) < 1: raise ValueError("integers >= 1, comma-separated")
            if val == "view" and v.lower() not in ("top-down", "side", "3/4", "isometric"): raise ValueError("bad view")
            if val == "light" and v.lower() not in LIGHT_VEC: raise ValueError("bad direction")
            if val == "yesno" and v.lower() not in ("yes", "no"): raise ValueError("yes|no")
            if val == "palette":
                n = len(load_palette(base / v))
                if n < MIN_RAMP_LEN: raise ValueError(f"only {n} colors")
        except Exception as e:
            probs.append((f, f"invalid: {v!r} ({e})", q))
    return probs

def cmd_brief_check(path):
    d = json.loads(Path(path).read_text())
    probs = brief_problems(d, Path(path).parent)
    if not probs:
        print("BRIEF COMPLETE"); return 0
    print(f"BRIEF INCOMPLETE: {len(probs)} question(s) for the user. do not generate yet.")
    for f, why, q in probs:
        print(f"  [{f}] {why}\n      ask: {q}")
    return 1


# ---------- tilemaps (build scenes from small checked tiles, not raw generation) ----------
def load_tile(path):
    p = Path(path)
    if p.suffix.lower() == ".txt":
        return parse_grid(p.read_text())
    nat, info = to_native(load(p))
    return nat

def load_tileset(d):
    files = sorted([f for f in Path(d).iterdir() if f.suffix.lower() in (".png", ".txt")])
    if not files:
        raise SystemExit(f"no .png/.txt tiles in {d}")
    ts = {}
    for f in files:
        if f.stem in ts:
            raise SystemExit(f"duplicate tile name {f.stem!r} (.png and .txt?)")
        ts[f.stem] = load_tile(f)
    sizes = {t.shape[:2] for t in ts.values()}
    if len(sizes) != 1:
        raise SystemExit(f"tiles differ in size: { {k: v.shape[1::-1] for k, v in ts.items()} }")
    return ts

def parse_map(text, tileset):
    rows = [ln.split() for ln in text.strip().splitlines() if ln.strip() and not ln.startswith("#")]
    if not rows:
        raise SystemExit("empty map")
    if len({len(r) for r in rows}) != 1:
        raise SystemExit(f"ragged map: row lengths {sorted({len(r) for r in rows})}")
    unknown = sorted({c for r in rows for c in r if c != "." and c not in tileset})
    if unknown:
        raise SystemExit(f"map uses unknown tiles {unknown}; have {sorted(tileset)}")
    return rows

def render_map(tileset, rows):
    th, tw = next(iter(tileset.values())).shape[:2]
    out = np.zeros((len(rows) * th, len(rows[0]) * tw, 4), np.uint8)
    for j, r in enumerate(rows):
        for i, name in enumerate(r):
            if name != ".":
                out[j*th:(j+1)*th, i*tw:(i+1)*tw] = tileset[name]
    return out

def seam_report(tileset, rows):
    """seams must look like tile interiors. two deterministic tests:
    novel: color pairs that touch across a seam but never touch inside any tile
    jumps: skipped-ramp-step density on seams vs inside tiles"""
    img = render_map(tileset, rows)
    th, tw = next(iter(tileset.values())).shape[:2]
    H, W = img.shape[:2]
    key = (img[..., 0].astype(np.int64) << 16) | (img[..., 1].astype(np.int64) << 8) | img[..., 2]
    key = np.where(img[..., 3] >= ALPHA_CUT, key, -1)
    vocab = set()
    for t in tileset.values():
        k = (t[..., 0].astype(np.int64) << 16) | (t[..., 1].astype(np.int64) << 8) | t[..., 2]
        k = np.where(t[..., 3] >= ALPHA_CUT, k, -1)
        for a_, b_ in ((k[:, :-1], k[:, 1:]), (k[:-1], k[1:])):
            vocab |= {(min(x, y), max(x, y)) for x, y in zip(a_.ravel().tolist(), b_.ravel().tolist())}
    novel_m = np.zeros((H, W), bool); seam_pairs = novel = 0
    for x in range(tw, W, tw):
        for y in range(H):
            a_, b_ = int(key[y, x-1]), int(key[y, x])
            if a_ < 0 and b_ < 0: continue
            seam_pairs += 1
            if (min(a_, b_), max(a_, b_)) not in vocab:
                novel += 1; novel_m[y, x-1] = novel_m[y, x] = True
    for y in range(th, H, th):
        for x in range(W):
            a_, b_ = int(key[y-1, x]), int(key[y, x])
            if a_ < 0 and b_ < 0: continue
            seam_pairs += 1
            if (min(a_, b_), max(a_, b_)) not in vocab:
                novel += 1; novel_m[y-1, x] = novel_m[y, x] = True
    seam = np.zeros((H, W), bool)
    for x in range(tw, W, tw): seam[:, x-1:x+1] = True
    for y in range(th, H, th): seam[y-1:y+1, :] = True
    mode = "sprite" if (img[..., 3] < ALPHA_CUT).any() else "scene"
    _, M = analyze(img, mode=mode)
    jm = M["shading_jumps"]
    js = float(jm[seam].mean()) if seam.any() else 0.0
    ji = float(jm[~seam].mean()) if (~seam).any() else 0.0
    rep = {
        "novel_pairs": {"pass": novel / max(seam_pairs, 1) <= SEAM_NOVEL_MAX,
                        "detail": f"{novel}/{seam_pairs} seam pairs never seen inside tiles (<= {SEAM_NOVEL_MAX:.0%})"},
        "seam_jumps": {"pass": js <= ji + SEAM_MARGIN,
                       "detail": f"jump density seam {js:.3f} vs interior {ji:.3f} (+{SEAM_MARGIN})"},
    }
    masks = {"novel_pairs": novel_m, "seam_jumps": jm & seam}
    return all(v["pass"] for v in rep.values()), rep, masks, img

def cmd_tilemap(tdir, map_path, out, scale):
    ts = load_tileset(tdir); rows = parse_map(Path(map_path).read_text(), ts)
    img = render_map(ts, rows)
    Image.fromarray(img.repeat(scale, 0).repeat(scale, 1)).save(out)
    th, tw = next(iter(ts.values())).shape[:2]
    print(f"wrote {out}: {len(rows[0])}x{len(rows)} tiles of {tw}x{th} = {img.shape[1]}x{img.shape[0]} native, x{scale}")

def cmd_seams(tdir, map_path, self_tile, mark_path):
    ts = load_tileset(tdir)
    if self_tile:
        if self_tile not in ts: raise SystemExit(f"no tile {self_tile!r}")
        rows = [[self_tile] * 3 for _ in range(3)]
    else:
        rows = parse_map(Path(map_path).read_text(), ts)
    ok, rep, masks, img = seam_report(ts, rows)
    for k, v in rep.items():
        print(f"{'PASS' if v['pass'] else 'FAIL'}  {k:12s} {v['detail']}")
    print("RESULT:", "PASS" if ok else "FAIL")
    if mark_path:
        rep2 = {k: {"pass": v["pass"], "level": "FAIL"} for k, v in rep.items()}
        Image.fromarray(mark(img, rep2, masks)).save(mark_path)
    return 0 if ok else 1


# ---------- validate (is the checker trustworthy on THESE refs?) ----------
def _edge(m):
    pm = np.pad(m, 1)
    return m & ((~pm[:-2, 1:-1]) | (~pm[2:, 1:-1]) | (~pm[1:-1, :-2]) | (~pm[1:-1, 2:]))

def degradations(pal_rgb):
    """realistic AI failure modes, seeded -> deterministic."""
    from PIL import ImageFilter
    P = np.array(pal_rgb)
    def grain(a, r):
        o = a.copy(); m = a[..., 3] >= ALPHA_CUT; g = r.integers(-4, 5, a[..., :3].shape)
        o[..., :3] = np.where(m[..., None], (a[..., :3].astype(int) + g).clip(0, 255), a[..., :3]); return o
    def aa_blur(a, r):
        im = Image.fromarray(a).resize((a.shape[1]*4, a.shape[0]*4), Image.NEAREST).filter(ImageFilter.GaussianBlur(3))
        return np.array(im.resize((a.shape[1], a.shape[0]), Image.BOX))
    def off_grid(a, r):
        return np.array(Image.fromarray(a).resize((int(a.shape[1]*3.5), int(a.shape[0]*3.5)), Image.NEAREST))
    def flat_shading(a, r):
        m = a[..., 3] >= ALPHA_CUT; o = a.copy()
        q = Image.fromarray(a[..., :3]).quantize(colors=4, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        o[..., :3] = np.where(m[..., None], np.array(q.convert("RGB")), a[..., :3]); return o
    def salt_noise(a, r):
        o = a.copy(); ys, xs = np.nonzero(a[..., 3] >= ALPHA_CUT)
        for i in r.choice(len(ys), max(3, len(ys) // 20), replace=False):
            o[ys[i], xs[i], :3] = P[r.integers(len(P))]
        return o
    def broken_outline(a, r):
        o = a.copy(); m = a[..., 3] >= ALPHA_CUT; ys, xs = np.nonzero(_edge(m))
        lit = a[m][:, :3]; light = lit[np.argsort([lstar(c) for c in lit], kind="stable")[len(lit) * 3 // 4]]
        for i in r.choice(len(ys), len(ys) * 4 // 10, replace=False):
            o[ys[i], xs[i], :3] = light
        return o
    def hue_drift(a, r):
        o = a.copy(); m = a[..., 3] >= ALPHA_CUT; sft = a[..., :3].astype(int); sft[..., 0] = (sft[..., 0] + 40).clip(0, 255)
        o[..., :3] = np.where(m[..., None], sft, a[..., :3]); return o
    def baked_shadow(a, r):
        H, W = a.shape[:2]; o = np.zeros((H + 4, W + 4, 4), np.uint8); m = a[..., 3] >= ALPHA_CUT
        sh = np.zeros((H + 4, W + 4), bool); sh[4:4+H, 0:W] |= m
        o[sh] = (70, 55, 80, 255); o[0:H, 4:4+W][m] = a[m]; return o
    def jaggies(a, r):           # content resampled at a non-integer factor, grid kept: broken stair-steps
        h, w = a.shape[:2]
        sm = Image.fromarray(a).resize((max(4, int(w * 0.87)), max(4, int(h * 0.87))), Image.NEAREST)
        return np.array(sm.resize((w, h), Image.NEAREST))
    D = dict(grain=grain, aa_blur=aa_blur, off_grid=off_grid, flat_shading=flat_shading, salt_noise=salt_noise,
             hue_drift=hue_drift)
    return D, dict(broken_outline=broken_outline, baked_shadow=baked_shadow), dict(jaggies=jaggies)

def cmd_validate(refs, palette_file=None, line_refs=None):
    """leave-one-out: calibrate on all-but-one clean ref of a mode, check the held-out
    ref (should PASS) and its degraded copies (should FAIL). prints the confusion table."""
    import contextlib, io, tempfile
    tmp = Path(tempfile.mkdtemp())
    info = {}
    for r in refs:
        nat, inf = prepare(r)
        if inf["kind"] == "clean":
            mode = "sprite" if (nat[..., 3] < ALPHA_CUT).any() else "scene"
            info[r] = (nat, mode)
    pal = load_palette(palette_file) if palette_file else None
    exit_code = 0
    for mode in ("sprite", "scene"):
        group = [r for r, (_, md) in info.items() if md == mode]
        if len(group) < 3:
            if group: print(f"[{mode}] only {len(group)} clean refs; need >= 3 to validate")
            continue
        cols = pal or sorted({hx(c) for r in group for c in info[r][0][info[r][0][..., 3] >= ALPHA_CUT][:, :3].tolist()})
        pf = tmp / f"{mode}.hex"; pf.write_text("\n".join(c[1:] for c in cols))
        D_all, D_spr, D_scn = degradations([unhx(c) for c in cols])
        D = {**D_all, **(D_spr if mode == "sprite" else D_scn)}
        false_fail, caught, by = [], Counter(), defaultdict(Counter)
        for held in group:
            with contextlib.redirect_stdout(io.StringIO()):
                lr_ = [x for x in (line_refs or []) if Path(x).name != Path(held).name]
                st = cmd_extract([g for g in group if g != held] + lr_, str(tmp / "loo.json"), palette_file=str(pf),
                                 line_refs=[Path(x).name for x in lr_])
            a = info[held][0]
            ok, rep, _, _ = audit(st, a)
            if not ok:
                false_fail.append((Path(held).name, [k for k, v in rep.items() if not v["pass"] and v["level"] == "FAIL"]))
            for name, fn in D.items():
                ok, rep, _, _ = audit(st, fn(a, np.random.default_rng(42)))
                if not ok:
                    caught[name] += 1
                for k, v in rep.items():
                    if not v["pass"] and v["level"] == "FAIL":
                        by[name][k] += 1
        n = len(group)
        print(f"[{mode}] {n} clean refs, palette locked ({len(cols)} colors)")
        print(f"  good refs passing when held out: {n - len(false_fail)}/{n}")
        for f, why in false_fail:
            print(f"    held-out FAIL {f}: {why}")
        for name in D:
            print(f"  {name:15s} caught {caught[name]:2d}/{n}   by {dict(by[name].most_common(3))}")
        if false_fail:
            exit_code = 1
    return exit_code


# ---------- tests ----------
O, A, B, C = (20, 12, 28, 255), (200, 80, 60, 255), (240, 140, 90, 255), (120, 40, 40, 255)
PAL4 = [(30, 40, 90), (60, 80, 140), (110, 130, 190), (170, 190, 230)]

def _sprite():
    """16x16: closed 1px outline, 3-color ramp C<A<B, lit top-left."""
    img = np.zeros((16, 16, 4), np.uint8)
    img[3:13, 3:13] = A
    img[4:7, 4:7] = B
    img[9:12, 9:12] = C
    img[3, 3:13] = O; img[12, 3:13] = O; img[3:13, 3] = O; img[3:13, 12] = O
    return img

def _scene(seed=1):
    rng = np.random.default_rng(seed)
    s = np.zeros((50, 40, 4), np.uint8); s[..., 3] = 255
    s[..., :3] = np.array(PAL4, np.uint8)[rng.integers(0, 4, (50, 40))]
    return s

def cmd_selftest():
    import tempfile, contextlib, io
    d = Path(tempfile.mkdtemp())
    base, scene = _sprite(), _scene()
    Image.fromarray(base.repeat(4, 0).repeat(4, 1)).save(d / "ref.png")
    Image.fromarray(scene.repeat(3, 0).repeat(3, 1)).save(d / "scene.png")
    (d / "pal.hex").write_text("\n".join(hx(c)[1:] for c in [O, A, B, C] + PAL4) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        style = cmd_extract([str(d / "ref.png"), str(d / "scene.png")], str(d / "style.json"),
                            palette_file=str(d / "pal.hex"))
    r = style["rules"]["sprite"]
    res = []
    def t(name, got, want):
        res.append(got == want); print(f"{'ok  ' if got == want else 'FAIL'} {name}: got {got!r} want {want!r}")
    def fails(img, check):
        _, rep, _, _ = audit(style, img); return not rep[check]["pass"]
    def var(fn):
        im = base.copy(); fn(im); return im

    # extraction
    t("per-mode rules", sorted(style["rules"]), ["scene", "sprite"])
    t("scale detected", style["refs"][0]["pitch"], [4, 4])
    t("palette locked from file", len(style["palette"]), 8)
    t("palette ramps grouped", sorted(style["palette_ramps"]), sorted([["#782828", "#c8503c", "#f08c5a"], ["#1e285a", "#3c508c", "#6e82be", "#aabee6"], ["#140c1c"]]))
    t("ramp found", style["example_ramps"]["sprite"][0], ["#782828", "#c8503c", "#f08c5a"])
    t("light dir", r["light_from"], "top-left")
    _, rep, _, _ = audit(style, base)
    t("clean sprite passes all", [k for k, v in rep.items() if not v["pass"]], [])
    _, rep, _, _ = audit(style, scene)
    t("clean scene passes all", [k for k, v in rep.items() if not v["pass"]], [])

    # one variant per check — each must trip its own check
    def set_(y, x, c): return lambda im: im.__setitem__((y, x), c)
    t("artifact: island",      fails(var(set_(0, 0, O)), "no_islands"), True)
    t("artifact: pinhole",     fails(var(set_(7, 7, (0, 0, 0, 0))), "no_pinholes"), True)
    t("artifact: soft alpha",  fails(var(set_(2, 5, (20, 12, 28, 100))), "no_partial_alpha"), True)
    t("artifact: orphan",      fails(var(set_(7, 8, C)), "orphans"), True)
    def dup2(im): im[7, 8] = (203, 83, 63, 255); im[8, 6] = (197, 78, 58, 255)
    t("artifact: near-dup",    fails(var(dup2), "no_near_dup_colors"), True)
    def grain(im):
        g = np.random.default_rng(3).integers(-3, 4, im[..., :3].shape)
        f = im[..., 3] > 0
        im[..., :3] = np.where(f[..., None], (im[..., :3].astype(int) + g).clip(0, 255), im[..., :3]).astype(np.uint8)
    t("artifact: grain/noise", fails(var(grain), "no_grain"), True)
    t("color: off-palette",    fails(var(set_(7, 8, (0, 255, 0, 255))), "on_palette"), True)
    t("line: doubled outline", fails(var(lambda im: im.__setitem__((4, slice(5, 10)), O)), "outline_doubles"), True)
    t("line: outline break",   fails(var(lambda im: im.__setitem__((3, slice(4, 11)), A)), "outline_coverage"), True)
    def pale(im): im[(im == O).all(-1)] = (150, 140, 160, 255)
    t("line: pale outline",    fails(var(pale), "outline_dark"), True)
    def selout(im): im[3, 3:8] = (45, 10, 12, 255)    # tinted near-black: still an outline
    _, rep, _, _ = audit(style, var(selout))
    t("line: tinted selout ok", rep["outline_coverage"]["pass"] and rep["outline_dark"]["pass"], True)
    def flat(im):
        for c in (B, C): im[(im == c).all(-1)] = A
    t("shading: flat ramp",    fails(var(flat), "ramp_length"), True)
    def mud(im): im[(im == C).all(-1)] = (190, 75, 60, 255)
    t("shading: muddy ramp",   fails(var(mud), "ramp_contrast"), True)
    def jump(im): im[7:9, 7:12] = B
    t("shading: skipped step", fails(var(jump), "shading_jumps"), True)
    def pillow(im):
        im[4:12, 4:12] = C; im[5:11, 5:11] = A
    t("shading: pillow",       fails(var(pillow), "no_pillow"), True)
    t("shading: light flipped", fails(np.rot90(base, 2).copy(), "light_direction"), True)

    # grids + scenes
    up = np.array(Image.fromarray(scene).resize((373, 467), Image.NEAREST))   # 9.33x, non-integer
    nat, info = to_native(up)
    t("fractional grid: pitch found", info["method"], "fractional")
    t("fractional grid: exact recovery", bool(nat.shape == scene.shape and np.array_equal(nat, scene)), True)
    _, rep_s, _, _ = audit(style, scene)
    t("scene mode skips outline checks", rep_s["outline_coverage"]["detail"].startswith("n/a"), True)
    close = scene.copy(); close[0:25, :, :3] = np.where(close[0:25, :, :3] == PAL4[0], (34, 44, 94), close[0:25, :, :3])
    t("deliberate close shades != artifact", analyze(close)[0]["near_dup_colors"], 0)
    stray = scene.copy(); stray[10, 10, :3] = (33, 43, 93)
    t("rare stray shade = artifact", analyze(stray)[0]["near_dup_colors"], 1)

    # line rhythm: slope-1/2 bands (runs 2-2-2) vs jaggy bands (runs 1-3-1-3)
    def bands(f):
        im = np.zeros((96, 96, 4), np.uint8); im[..., 3] = 255
        yy_, xx_ = np.mgrid[0:96, 0:96]
        im[..., :3] = np.array(PAL4, np.uint8)[((yy_ - f(xx_)) // 6) % 4]
        return im
    def lr(im):
        k_ = (im[..., 0].astype(np.int64) << 16) | (im[..., 1].astype(np.int64) << 8) | im[..., 2]
        lab_, sz_ = same_color_components(k_, np.ones(k_.shape, bool))
        return line_rhythm(lab_, sz_, k_.shape)
    clean_l, jag_l = lr(bands(lambda x: x // 2)), lr(bands(lambda x: (x // 4) * 2 + (x % 4 >= 1)))
    t("lines: clean slope has no spikes", (clean_l[0] > 300, clean_l[1]), (True, 0.0))
    t("lines: 1-3-1-3 jaggies flagged", jag_l[1] > 0.5, True)

    # split: frame + noisy checker + tinted drop shadow + sprite with an enclosed pocket
    rng = np.random.default_rng(7)
    sh = np.zeros((60, 60, 4), np.uint8); sh[..., 3] = 255
    yy, xx = np.mgrid[0:60, 0:60]
    chk = np.where(((yy // 8) + (xx // 8)) % 2 == 0, 138, 130)
    sh[..., 0] = chk + rng.integers(-3, 4, (60, 60)); sh[..., 1] = chk - 20 + rng.integers(-3, 4, (60, 60))
    sh[..., 2] = chk - 30 + rng.integers(-3, 4, (60, 60))
    sh[16:30, 4:18, :3] = (sh[16:30, 4:18, :3] * np.array([0.62, 0.58, 0.8])).astype(np.uint8)   # shadow
    ring = np.zeros((14, 14, 4), np.uint8); ring[..., :] = O; ring[3:11, 3:11, 3] = 0   # pocket = bg
    ring[1:13, 1:13] = A; ring[1:3, 1:13] = B; ring[11:13, 1:13] = C
    ring[3:11, 3:11] = 0; ring[2:12, 2] = ring[2:12, 11] = O; ring[2, 2:12] = ring[11, 2:12] = O
    for yy_, xx_ in zip(*np.nonzero(ring[..., 3] > 0)):
        sh[10 + yy_, 10 + xx_] = ring[yy_, xx_]
    sh[34:50, 34:50][base[..., 3] > 0] = base[base[..., 3] > 0]
    sh[:2] = sh[-2:] = (237, 237, 216, 255); sh[:, :2] = sh[:, -2:] = (237, 237, 216, 255)
    sprites, dropped, _ = split_sheet(sh.repeat(4, 0).repeat(4, 1))
    t("split: finds 2 sprites", len(sprites), 2)
    got = sprites[1][0][1:-1, 1:-1] if len(sprites) == 2 else None
    crop = base[3:13, 3:13]
    t("split: sprite recovered exactly", bool(got is not None and np.array_equal(got, crop)), True)
    rg = sprites[0][0][1:-1, 1:-1] if sprites else None
    t("split: pocket made transparent", bool(rg is not None and (rg[4:9, 4:9, 3] == 0).all()), True)
    t("split: shadow removed", bool(rg is not None and rg.shape[:2] == (14, 14)), True)

    # palette from moodboard
    mb = np.zeros((40, 60, 3), np.uint8)
    for i, c in enumerate([(200, 80, 60), (240, 140, 90), (120, 40, 40)] + PAL4[:3]):
        mb[:, i * 10:(i + 1) * 10] = c
    Image.fromarray(mb).save(d / "mood.png")
    with contextlib.redirect_stdout(io.StringIO()):
        p1 = cmd_palette([str(d / "mood.png")], str(d / "mood.hex"), 16, 0.004)
        p2 = cmd_palette([str(d / "mood.png")], str(d / "mood2.hex"), 16, 0.004)
    t("palette: extracts all 6 colors", len(p1), 6)
    t("palette: deterministic", p1 == p2, True)
    (d / "bad.hex").write_text("ff00zz\n")
    try: load_palette(d / "bad.hex"); t("palette: bad hex rejected", False, True)
    except ValueError: t("palette: bad hex rejected", True, True)

    # brief gate
    with contextlib.redirect_stdout(io.StringIO()):
        cmd_brief_new(d / "brief.json")
    bj = json.loads((d / "brief.json").read_text())
    t("brief: blank fails", len(brief_problems(bj, d)) > 0, True)
    bj.update({"subject": "iron key", "use": "item icon", "kind": "sprite", "native_size": "32x32",
               "display_scale": "3,6", "palette": "pal.hex", "view": "3/4", "light_from": "top-left",
               "outline": "dark selout", "background": "transparent", "shadow": "no", "animation": "static",
               "refs": "items1", "deliverable": "png x1 and x4"})
    t("brief: complete passes", brief_problems(bj, d), [])
    bj["light_from"] = "above"
    t("brief: invalid value caught", [f for f, *_ in brief_problems(bj, d)], ["light_from"])

    # tiles + seams
    td = d / "tiles"; td.mkdir()
    a_t = np.zeros((8, 8, 4), np.uint8); a_t[..., 3] = 255
    a_t[..., :3] = np.array(PAL4, np.uint8)[(np.add.outer(np.arange(8), np.arange(8)) // 2) % 4]  # wraps cleanly
    b_t = a_t.copy(); b_t[:, 7, :3] = B[:3]; b_t[:, 0, :3] = C[:3]              # edges never meet inside
    Image.fromarray(a_t).save(td / "water.png"); Image.fromarray(b_t).save(td / "bad.png")
    ts = load_tileset(td)
    t("seams: seamless tile passes", seam_report(ts, [["water"] * 3] * 3)[0], True)
    t("seams: mismatched tile fails", seam_report(ts, [["bad"] * 3] * 3)[0], False)
    try: parse_map("water bad\nwater", ts); t("map: ragged rejected", False, True)
    except SystemExit: t("map: ragged rejected", True, True)
    try: parse_map("water lava", ts); t("map: unknown tile rejected", False, True)
    except SystemExit: t("map: unknown tile rejected", True, True)

    # tools
    rng = np.random.default_rng(0)
    noisy = base.repeat(8, 0).repeat(8, 1).astype(np.int16)
    noisy[..., :3] += rng.integers(-15, 16, noisy[..., :3].shape).astype(np.int16)
    t("snap recovers sprite", bool(np.array_equal(snap(style, noisy.clip(0, 255).astype(np.uint8), (16, 16)), base)), True)
    grid = "\n".join(" ".join("." if px[3] == 0 else hx(px) for px in row) for row in base)
    t("render round-trips", bool(np.array_equal(parse_grid(grid), base)), True)
    try: parse_grid("#000000 .\n."); t("ragged grid rejected", False, True)
    except ValueError: t("ragged grid rejected", True, True)
    t("deterministic", json.dumps(audit(style, base)[1]) == json.dumps(audit(style, base)[1]), True)
    print(f"\n{sum(res)}/{len(res)} passed")
    return 0 if all(res) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("brief"); b.add_argument("action", choices=["new", "check"]); b.add_argument("path")
    pl = sp.add_parser("palette"); pl.add_argument("images", nargs="+"); pl.add_argument("-o", required=True)
    pl.add_argument("-n", type=int, default=32); pl.add_argument("--min-share", type=float, default=0.004)
    e = sp.add_parser("extract"); e.add_argument("refs", nargs="+"); e.add_argument("-o", default="style.json")
    e.add_argument("--palette", help="strict locked palette file (.hex/.png/.json)")
    e.add_argument("--lock", nargs="+", help="lock palette to these refs' colors")
    e.add_argument("--calibrate-all", action="store_true", help="also calibrate from lossy/painterly refs")
    e.add_argument("--line-refs", nargs="+", help="mood/painterly refs trusted for line rhythm only")
    pr = sp.add_parser("prompt"); pr.add_argument("style"); pr.add_argument("--mode", choices=["sprite", "scene"], required=True)
    c = sp.add_parser("check"); c.add_argument("style"); c.add_argument("img")
    c.add_argument("--json", action="store_true"); c.add_argument("--mark"); c.add_argument("--strict", action="store_true")
    c.add_argument("--palette", help="override the style's palette with this file")
    s_ = sp.add_parser("snap"); s_.add_argument("style"); s_.add_argument("img"); s_.add_argument("-o", required=True)
    s_.add_argument("--size", required=True); s_.add_argument("--scale", type=int, default=1)
    rd = sp.add_parser("render"); rd.add_argument("grid"); rd.add_argument("-o", required=True)
    rd.add_argument("--scale", type=int, default=1)
    sl = sp.add_parser("split"); sl.add_argument("sheet"); sl.add_argument("-o", required=True); sl.add_argument("--preview")
    tm = sp.add_parser("tilemap"); tm.add_argument("tiles"); tm.add_argument("map"); tm.add_argument("-o", required=True)
    tm.add_argument("--scale", type=int, default=1)
    se = sp.add_parser("seams"); se.add_argument("tiles"); se.add_argument("map", nargs="?")
    se.add_argument("--self", dest="self_tile"); se.add_argument("--mark")
    va = sp.add_parser("validate"); va.add_argument("refs", nargs="+"); va.add_argument("--palette")
    va.add_argument("--line-refs", nargs="+", help="mood refs trusted for line rhythm only")
    sp.add_parser("selftest")
    a = p.parse_args()
    if a.cmd == "brief": sys.exit(cmd_brief_new(a.path) if a.action == "new" else cmd_brief_check(a.path))
    elif a.cmd == "palette": cmd_palette(a.images, a.o, a.n, a.min_share)
    elif a.cmd == "extract": cmd_extract(a.refs, a.o, a.lock, a.calibrate_all, a.palette, a.line_refs)
    elif a.cmd == "prompt": cmd_prompt(a.style, a.mode)
    elif a.cmd == "check": sys.exit(cmd_check(a.style, a.img, a.json, a.mark, a.strict, a.palette))
    elif a.cmd == "snap": cmd_snap(a.style, a.img, a.o, a.size, a.scale)
    elif a.cmd == "render": cmd_render(a.grid, a.o, a.scale)
    elif a.cmd == "split": cmd_split(a.sheet, a.o, a.preview)
    elif a.cmd == "tilemap": cmd_tilemap(a.tiles, a.map, a.o, a.scale)
    elif a.cmd == "seams":
        if not a.map and not a.self_tile: raise SystemExit("seams needs MAP or --self TILE")
        sys.exit(cmd_seams(a.tiles, a.map, a.self_tile, a.mark))
    elif a.cmd == "validate": sys.exit(cmd_validate(a.refs, a.palette, a.line_refs))
    elif a.cmd == "selftest": sys.exit(cmd_selftest())

if __name__ == "__main__":
    main()
