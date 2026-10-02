# Panel layout — two functions, a rule catalog, and candidate rule sets

This supersedes the earlier single-function draft.  The layout is now split the
way you described:

1. **`frames_per_row(a, b, ctx) -> 1 | 2`** — decides only *how many* frames the
   current row holds.  Think of it as the catalog of **exception rules that force
   one frame**; the default is two.
2. **`plan_two(a, b, layout, rules) -> (PanelPlan, PanelPlan)`** — places
   *exactly two* frames into one row.  Because the count decision already
   removed solo, odd-tail, wide-panorama and (optionally) infeasible cases, this
   function never branches on "only one frame."

A trivial helper **`plan_solo(f, layout) -> PanelPlan`** places a single frame,
and a driver **`plan_rows(frames, ...)`** walks the sequence, calls
`frames_per_row`, then dispatches to `plan_two` or `plan_solo`, chunking rows
into pages of three.

Why the split pays off: `plan_two` only has to solve the coupled
`w_a + w_b ≤ R` allocation, and (when the count policy includes the feasibility
exception) may assume a feasible pair — no fallback branches.  Test sets can
then vary the two policies **independently**, which is exactly what you want to
evaluate.

```python
def plan_rows(frames, layout, count_rules, place_rules):
    rows, i, page_row = [], 0, 0
    while i < len(frames):
        if i == len(frames) - 1:                 # odd tail
            rows.append(Row([plan_solo(frames[i])], reason="last frame")); break
        a, b = frames[i], frames[i + 1]
        n = frames_per_row(a, b, ctx(page_row=page_row), layout, count_rules)
        if n == 1:
            rows.append(Row([plan_solo(a)], reason=...)); i += 1
        else:
            pa, pb = plan_two(a, b, layout, place_rules)
            rows.append(Row([pa, pb])); i += 2
        page_row = (page_row + 1) % 3            # 3 rows per page
    return chunk_pages(rows, 3)
```

`ctx` carries `page_row` (0-based position within the page, for K2) and the
frame index.  Order is always preserved: when a row is solo, the frame that
would have shared it simply becomes the `a` of the next row.

---

## 1. `frames_per_row` — the count-rule catalog

Each rule is an **exception that returns 1**; if none fire, return **2**.  The
hard safety rules are always on; the rest are per-set.

### 1.1 Set K — your count rules (evaluated, kept as written)

| ID | Condition | Result | Validation / concern |
| --- | --- | --- | --- |
| `K1` | either frame is a **wide panorama** — `is_panorama` and aspect ratio `= u_p·(16/9) > 16/9` (i.e. `u_p > 1.0`) | 1 | Only a pano wider than 16:9 is forced solo.  A square or narrower pano is *not* wide and may share a row when it fits; if it does not fit, `K_infeasible` still sends it to its own row. |
| `K2` | `page_row == 0` **and** frame A has no heads and no persons | 1 | Establishing-shot logic: an empty first panel sets the place before characters appear.  *Concern:* "no detections" is a proxy for "establishing"; a busy texture or clock with no detections would also get a full row.  Tunable by requiring `page_row == 0` **and** A being the first scene of the clip, not just the page. |
| `K3` | `σ_A + σ_B < 0.07` (sum of carve fractions; 0.07 = "7%") **and** neither frame is a square panorama | 1 | If neither frame has much safe carve budget, both are data-rich, so forcing two panels would require heavily cropping both.  A full row each loses less.  *Tuned down from 0.15:* 0.15 fired on nearly every pair on a real clip (most `σ` are a few percent), dropping pages to ~3 panels; 0.07 keeps genuinely rigid pairs solo only. |
| default | — | 2 | Two frames unless an exception fires. |

**Safety rules (always on, even in Set K).**

| ID | Condition | Result |
| --- | --- | --- |
| `K_odd` | only one frame remains | 1 |
| `K_infeasible` | `u_a·k_min_a + u_b·k_min_b > R` with the **capped** floor | 1 |

`K_infeasible` was firing far too often in the first trial because the subject
floor `k_min` grew to ~1.0 for wide unions and oversized boxes, so almost any
close-up pair looked impossible.  Two changes make it rare:

* **`max_keep` cap** (`0.75`): `k_min` is clamped so two regular 16:9 frames
  always fit a row (`2 × 0.75 = 1.5`).  When the cap binds the frame is simply
  **hard cropped** through the wide subject union instead of being sent solo.
* **Conservative feasibility:** `_min_units` uses crop-only (no carve) at the
  capped floor, so the count decision is consistent with a placement policy that
  gates carving off.

The net effect: `K_infeasible` now only fires for **panoramas** (untouchable) or
frames already wider than 16:9 — not for ordinary close-up pairs.

### 1.2 Other count rules worth evaluating (reusable, optional)

| ID | Condition | Rationale |
| --- | --- | --- |
| `K_both_rigid` | both `σ < 0.05` **and** both `cover ≥ 50%` | Neither carve nor crop can safely shrink them; stronger than K3. |
| `K_sliver` | the split `plan_two` would give one frame `< keep_floor` | Prevents a degenerate sliver even when technically feasible. |
| `K_scene_gap` | A and B are more than `Δt` seconds apart or cross a scene-pack boundary | Avoids pairing unrelated beats just because they are adjacent; optional. |
| `K_tall` | a panorama is **taller than it is wide** (`height > width`) | The pano is never cropped or width-resized, so it gets **two page rows** of height instead of being squashed into one. Applied directly in `plan_rows` (like `K_odd`), not through `frames_per_row`. |
| ~~`K_hero`~~ | **removed** | Giving a single hero a full row is the wrong call: one subject is *easier* to crop around, so it should be a good shrink target (see `C8`), not a solo panel. |

### 1.3 Variants of K3 to test

K3 uses the **sum**; these alternatives change when a row goes solo:

- `K3_sum` (= K3): `σ_A + σ_B < T`.
- `K3_min`: `min(σ_A, σ_B) < T` — fires when *either* frame is rigid (stricter).
- `K3_max`: `max(σ_A, σ_B) < T` — fires only when *both* are rigid (looser).
- `K3_crop`: predicted required crop `1 − R/(u_A+u_B)` exceeds a threshold when
  both `σ` are low — directly models "how much cropping would this pair force?",
  which is the actual thing K3 is proxying.

K3_sum with `T=0.07` fires only when both frames are genuinely rigid; the first
draft of `T=0.15` fired on nearly every pair on a real clip because most `σ` are
a few percent, dropping pages to ~3 panels.  `K3_crop` and the `min`/`max`
variants are still worth comparing.

---

## 2. `plan_two` — placing exactly two frames

Guaranteed by the count step: exactly two frames, and (with `K_infeasible`) a
feasible pair.  Measure in **16:9-at-row-height units** (`u = (W/H)/(16/9)`, row
`= R ≈ 1.5`), and reduce both frames to satisfy `w_a + w_b ≤ R`.

### 2.1 Per-frame geometry

```
cover_i = body_percent + head_percent - overlap_percent
span_i  = (max box x2 - min box x1) / W
center_i= (min box x1 + max box x2) / 2
pad     = max(pad_frac·H_src, 0.35·max head width)
k_min_i = min(max((span_i·W + 2·pad)/W, keep_floor), max_keep)   # subject floor
κ_i     = 1 - k_min_i                                  # max crop fraction
σ_i     = seam_carve_shrink_i or 0                     # max safe carve fraction
```

`max_keep = 0.75` is the key addition: without it a wide union pushes `k_min`
towards `1.0`, i.e. "this frame cannot lose width", which is what made
`K_infeasible` fire constantly.  Capping it means such a frame is hard cropped
instead, and two 16:9 frames always fit.

### 2.2 The allocation (both frames shrink together)

```
D = max(0, u_a + u_b - R)                       # width units to remove
S_i = score(frame_i, rules);  ω_i = max(0, S_i) + base_weight
d_i = D · ω_i/(ω_a+ω_b)                         # share of the deficit
clip d_i to capacity (carve only if σ_i >= carve_min, then crop)
apply(frame_i, d_i):  carve min(σ_i, d_i/u_i) if σ_i >= carve_min, else crop
```

`base_weight` guarantees both frames participate (your "resize both so both lose
less"); `max_share` optionally caps one frame's share when the other still has
room.  A **panorama in a count-2 row** is handled by giving it `σ=κ=0`, so the
allocator naturally gives it zero deficit and the regular partner absorbs
everything — no special branch.

**Carve gate (`carve_min`).**  Seam carving used to run on almost every shrunk
frame because it was always tried first.  A placement policy now sets the
minimum `σ` worth carving; below it the frame is **hard cropped** instead.  The
gate also tightens the frame's capacity (a gated frame can only lose width by
cropping), so the count and placement steps stay consistent.  `carve_min=1.0`
disables carving entirely (`Place-G`).

### 2.3 The placement rule catalog

Score rules (summed):

| ID | Condition | Points | Validation |
| --- | --- | --- | --- |
| `C1` | 1 subject, `cover < 50%` | +1 | Margin to spare; centred crop keeps it. (Your rule.) |
| `C2` | 1 subject, `cover < 25%` | +1 | Even more margin; crop nearly free. |
| `C3` | ≥2 subjects, `cover ≥ 50%` | 0 | Bad crop target, not penalised. (Your rule.) |
| `C4` | ≥2 subjects, `span ≥ 60%` | −1 | Spread subjects raise `k_min`; crop risks cutting one. |
| `C5` | `cover ≥ 70%` | −1 | Any horizontal crop clips the subject. *(deprecated: a single large subject is now a good crop target; see C8)* |
| `C6` | `heads_in_body` and `head_percent ≥ 8%` | −1 | Foreground character; protect. *(deprecated: hero protection is the wrong call)* |
| `C7` | no heads and no persons | +2 | No anchor to damage; `κ = 1−keep_floor`. |
| `C8` | single subject and `k_min ≤ 0.60` | +1 | If the one subject already fits a tight crop, cropping around it is easy — the frame is a *good* shrink target. This is the deliberate inverse of the removed `K_hero`. |
| `T1` | `subtitle_count ≥ 4` | −1 | A text-heavy panel wants width so bubbles do not cover the art. |
| `S1` | `σ ≥ 0.25` | +2 | Engine found ≥25% low-energy width; two such ⇒ no crop. (Your tier.) |
| `S2` | `0.15 ≤ σ < 0.25` | +1 | Solid budget. (Your >15% example.) |
| `S3` | `σ < 0.15` | 0 | Rigid. (Your <15% example.) |
| `P1` | `center` within `0.12·W` of frame centre | +1 | Crop can be symmetric. |
| `P2` | `center` in outer 15% | −1 | Edge subject limits crop. |
| `H1` | `head_percent ≥ 8%` | −2 | *(deprecated)* |
| `H2` | `body_percent ≥ 50%` | −1 | *(deprecated)* |
| `H3` | `C7` holds | +3 (replaces C7) | Scene-setting frames are cheapest to trim. |

Method / placement rules (choose *how*, no points):

| ID | Action |
| --- | --- |
| `M1` | carve first, crop the residue (only when `σ ≥ carve_min`) |
| `M2` | crop centered on the subject bbox, clamped to `[0, W−crop_w]` |
| `M3` | crop must contain the union of all boxes + `pad` |
| `M4` | no subject: crop from the outer edge, else centre |
| `M5` | gaze bias for a single subject: open space into the gutter, pull back if facing out; `δ ≤ 0.08·W`, clamped by `k_min` |
| `M6` | recompute subject centres after carving |
| `M7` (implicit) | **hard crop**: when the cap binds, the kept width drops below the subject-safe `k_min`; the panel caption notes "hard crop below subject floor" |

### 2.4 Balance (placement-only fallback)

| ID | Action | Validation |
| --- | --- | --- |
| `F4` | if resulting widths leave the band `[0.45, 0.55]`, rebalance toward 50:50 within legal crops | Lopsided rows read as accidental. |

(Feasibility and panorama fallbacks moved to the count step, so `plan_two` has
no solo branch.)

---

## 3. `plan_solo` — placing one frame

| Frame | Placement |
| --- | --- |
| Regular | Scaled isotropically to row height `H`, natural width `u·(16/9)·H`, **centred**; leftover row width is margin.  No crop, no upscale (`H3`). |
| Wide panorama | Uniformly downscaled to the full row width `R` (height < H), vertically centred. |
| Narrow/square panorama | Natural width if it fits; else uniform downscale to `R`. |

---

## 4. Rule sets = count policy × placement policy

Because the two functions are independent, a "rule set" is now a **pair**.  Your
Set K is a count policy; it composes with any placement policy.

### 4.1 Count policies

| Policy | Rules | Character |
| --- | --- | --- |
| **`Count-K`** | `K1`, `K2`, `K3_sum` (T=0.07), default 2 (+ mandatory `K_odd`, `K_infeasible`) | Your proposal, retuned. |
| **`Count-Kp`** | `K1`, `K2`, `K3_crop`, default 2 | Your idea, proxied by predicted crop instead of raw σ sum. |
| **`Count-Establish`** | `K1`, `K2`, default 2 | Only the establishing-shot exception. |
| **`Count-Pano`** | `K1`, `K_both_rigid`, `K_sliver`, default 2 | Prefer pairs, solo only when a pair would be bad. |
| **`Count-Always2`** | `K1`, `K_odd`, `K_infeasible` only | Maximum density control. |

(`Count-Hero` and `Count-Conservative` are gone: hero soloing was removed and
`K_infeasible` is now rare, so the conservative policy is `Count-Pano`.)

### 4.2 Placement policies

`carve_min` is the gate below which a frame hard-crops instead of carving.

| Policy | Score rules | Methods | `carve_min` | Character |
| --- | --- | --- | --- | --- |
| `Place-A` | `C1,C2,C8,C3,C7,S1,S2,S3` | `M1–M4,M6` | 0.10 | your score, carving only when worth it |
| `Place-B` | same as A, `base=1.0`, `max_share=0.6` | `M1–M4,M6` | 0.00 | density-first |
| `Place-C` | `S1,S2,S3,C4,C8,C7`, `base=0.25` | `M1,M2,M3,M6` | 0.00 | carve-heavy, crop last |
| `Place-G` | `C1,C2,C8,C3,C7,P1,P2` | `M2,M3,M4,M6` | 1.00 | **crop-only** (never carves) |
| `Place-E` | `C1,C2,C8,C7,S1,S2,P1,P2,T1` | `M1–M6`, `F4` | 0.10 | gaze + balance + text width |
| `Place-F` | `S2,C7` only | `M1,M2,M6` | 0.00 | control, minimal |

### 4.3 Named combinations to run first

| Set | Count | Place | Question it answers |
| --- | --- | --- | --- |
| **K-B** | `Count-K` | `Place-B` | **Pipeline default.** Your count rules + score, carving whenever there is budget. |
| **K-G** | `Count-K` | `Place-G` | Crop-only: does hard cropping beat seam carving here? |
| **2-G** | `Count-Always2` | `Place-G` | The two "less carving / fewer singles" changes together. |
| **F-F** | `Count-Always2` | `Place-F` | Control: measures the value of all the extra rules. |

Removed after review: `K-A` (79% redundant with K-B), `K-C` (carve-heavy),
`2-A` (too-aggressive carving → distorted panels) and `K'-E` (too much
whitespace).  The `Place-A`/`Place-C`/`Place-E` policies remain in the registry
for future experiments but are no longer in `DEFAULT_SETS`.

The full matrix is still available (5 × 6 = 30 deterministic combos over the
same `scenes.json`) by selecting the policies directly: `--count Count-K
Count-Kp --place Place-A Place-C` builds the cross product (reaching policies
not in `DEFAULT_SETS`), or pass `--sets` to pick named sets.  Keep the pairs that score
best and mix their best individual rules afterwards.

### 4.4 What changed in this revision

* `K3` threshold `0.15 → 0.07`; it no longer fires on nearly every pair.
* `max_keep = 0.75` caps the subject floor, so `K_infeasible` no longer fires for
  ordinary close-up pairs — they hard-crop and share a row instead.  A first
  real run of the same 12-scene clip dropped from 8–10 single rows to **2** (the
  wide panorama and the odd tail).
* `K_hero`, `Count-Hero`, `Place-D` removed; `C8` now *rewards* a single
  crop-friendly subject, the inverse of the old hero protection.
* `carve_min` gate added: `Place-A` carves 7 of 12 panels in that clip, `Place-G`
  carves 0, `Place-B/C` carve 10 — a real spectrum instead of always carving.
* New rules: `C8` (single subject fits a tight crop), `T1` (subtitle width), and
  the implicit `M7` hard-crop note.  `C5`/`C6`/`H1`/`H2` are deprecated.
* `K_tall` added: a panorama taller than it is wide spans **two page rows**
  (rendered at its natural aspect, not squashed).  `Row.row_span` and
  span-aware pagination carry this; it is expected to be rare (one frame in a
  full episode).
* Sets removed after review: `K-C` (`Place-C` carve-heavy) and `2-A` (its
  carving distorted images), `K'-E` (`Place-E`, too much whitespace), and `K-A`
  (79% redundant with `K-B`).  The shipped `DEFAULT_SETS` are now `K-B`, `K-G`,
  `2-G`, `F-F`.

---

## 5. Panorama handling under the split

- **Count:** `K1` fires for a **wide** pano (aspect ratio > 16:9, i.e. unit
  `> 1.0`) ⇒ row of 1.  The pano is placed by `plan_solo` (uniform downscale to
  `R`); the regular frame becomes the `a` of the next row and is placed solo too,
  preserving order.
- **Count:** a **square/narrow** pano (aspect ratio ≤ 16:9) does not fire `K1` and
  stays at 2 frames; it shares when `u_p + u_regular·k_min ≤ R`, otherwise
  `K_infeasible` sends it, and its would-be partner, to separate rows.
- **Count:** a **tall** pano (`height > width`) fires `K_tall`: it is placed by
  `plan_solo` at its natural aspect but given **two page rows** of height
  (rendered `2 ×` the row height, never squashed).  It consumes two of the three
  page slots.
- **Place:** in a count-2 row with a pano, the pano gets `σ=κ=0`, so `plan_two`
  gives it zero deficit and the regular frame absorbs the whole `D`, matching the
  confirmed rule.
- Two adjacent wide panos: `K1` fires, each is soloed in turn.

---

## 6. Worked trace (Count-K × Place-A)

Frames in order:

| # | Frame | `u` | `σ` | subjects |
| --- | --- | --- | --- | --- |
| 0 | S1 scenery | 1.0 | 0.04 | none |
| 1 | S2 close-up | 1.0 | 0.30 | 1 head, 9%, centred |
| 2 | S3 two-shot | 1.0 | 0.03 | 2 persons, cover 60%, span 0.70 |
| 3 | S4 wide pano | 2.4 | — | — |
| 4 | S5 scenery | 1.0 | 0.05 | none |

- Row 0: A=S1, B=S2, `page_row=0`, A has no detections ⇒ **K2 fires** ⇒ 1.
  S1 solo, centred.  `i=1`.
- Row 1: A=S2, B=S3, `page_row=1`.  K3: `0.30+0.03=0.33 ≥ 0.07` ⇒ no.
  Count **2**.  `Place-A`: `S(S2)=C1+C2+S1(≥0.25)+P1 ≈ 5`, `S(S3)=C3+S3=0`.
  `base=0.5` ⇒ `ω=5.5:0.5`; with `D=0.5`, S2 takes ~0.46 and S3 ~0.04,
  clipped and spilled within the caps.  Both shrink; the two-shot is barely
  touched.  `i=3`.
- Row 2: A=S4 wide pano, B=S5.  `K1` ⇒ **1**.  S4 solo (uniformly downscaled
  to the full row).  `i=4`.
- Row 3 (page 2): A=S5, tail ⇒ `K_odd` ⇒ 1.  S5 solo.

Result: 1 + 2 + 1 + 1 = a 5-panel page-area over 4 rows, with the establishing
shot, the two-shot, and the pano all given room; only the close-up/two-shot row
was paired.  Changing `Count-Always2` would pair S1+S2 and S2+S3 instead and test
whether that density is worth the extra crop.

---

## 7. Evaluating count × placement on `input.mkv`

The harness is implemented:

* `src/anime2manga/layout.py` — the pure planner: `FrameMeta`, `LayoutConfig`,
  `frames_per_row`, `plan_two`, `plan_solo`, `plan_rows`, `paginate`, the
  `CountPolicy`/`PlacePolicy` registries (`COUNT_POLICIES`, `PLACE_POLICIES`,
  `DEFAULT_SETS`), and the evaluation metrics `head_cut` /
  `subject_center_error`.
* `src/anime2manga/layout_report.py` — `PanelRenderer` (carve then crop an
  actual frame) and the HTML writers (`render_set_html`, `render_index_html`).
* `scripts/layout_trials.py` — the driver.

Run it after a normal pipeline pass (`scenes.json` must exist):

```bash
PYTHONPATH=src python scripts/layout_trials.py output --source input.mkv -o output/layout_trials
# or: just layout
```

It writes `output/layout_trials/index.html` (metrics table + rule legend + links
to both
views), one `<set>.html` per rule set (3×2 pages; each panel lists the rules and
methods it fired with a short description of each), a **`<set>.clean.html`
rules-free preview** (pages of panels only, no captions or row labels, equal row
heights, a small white comic-style gutter between panels — a semi-finished page
for comparing compositions), a `<set>.json` plan dump, and `summary.csv/json`.
Pass `--sets K-B K-G` to run a subset, `--frames-dir` to reuse `output/frames`
instead of re-extracting from the video, and `--keep-floor`/`--max-keep`/
`--pad-frac` to tune the geometry; `--page-width` sets the preview page width and
`--gutter` the white space between preview panels (default 8px).  On a first real run the `0.15` sum
threshold for `K3` fired on almost every pair (most carve budgets are a few
percent), which the `single reasons` column made obvious; the shipped default is
now `0.07`.

The metrics are the table from §7 of the design above: `head_cut` first, then
`subject_center_error`, subject to `panels_per_page >= 5.5`, then distortion.
Sweep `K3`'s threshold and formulation (`sum`/`min`/`max`/`crop`), `T_hero`,
`keep_floor`, `pad_frac`, `base_weight`, `max_share`.

## 8. Comparing sets by image hash

`scripts/layout_compare.py <trials-dir>` hashes every panel image the sets
produced and reports, for each pair, the fraction of shared scenes whose image
hash is identical (per-scene agreement) plus the shared/different distinct
hashes.  It writes `comparison.md`/`comparison.json` in the trials dir.  This is
a cheap way to see which rule sets actually behave differently without
re-planning: two sets that agree on 100% of scenes are redundant.

On a full 292-scene run (seam budgets taken from `output/retarget/summary.json`),
the closest pairs were **K-A vs K-B (79% of scenes identical)**, **K-A vs F-F
(53%)**, and **K-G vs 2-G (48%)**; K-A vs K-G was only 31% (carving changes most
panels even under the same count policy).  K-A has since been removed (it was
79% redundant with K-B, the current default).
