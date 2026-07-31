# Mascot & logo — "Sabre"

A proposal for the sabrewing mark, plus the generation prompts to make it. The
current `web/public/sabrewing.svg` is a placeholder scratch shape; this is what
should replace it.

## The concept

**A hummingbird holding a hover, its swept wing built from a lattice of small
feather-tiles — most of them dark, a handful lit.**

Three things about this project are true at once, and one drawing can carry all
three:

1. **The name is literal.** A sabrewing (*Campylopterus*) is a large hummingbird
   whose adult males have thickened, flattened, sabre-curved shafts on the outer
   primary feathers. The wing *is* a blade. That is the silhouette: a compact
   body and a single long swept arc that reads as a sabre before it reads as a
   wing.
2. **The wing is the MoE.** The lit-feather lattice is the routing picture: 256
   experts, ten active. Most tiles sit dark in the plumage; a small top-k glows
   violet, one of them warm gorget-orange (the shared expert). A reader who
   knows nothing about MoE sees iridescence. A reader who does sees top-k.
3. **Hover, not sprint.** A hummingbird at a feeder is doing enormous sustained
   work while appearing still — which is exactly the thesis of this runtime:
   aggregate throughput over hours, not peak single-stream. So the mark is a
   *hover*, not a dive. Wings back and blurred into arc-trails, body level, beak
   forward. The stillness is the point; the figure-eight wingbeat trail behind
   it is the batch loop.

The bird faces **right** (forward, into the token stream) and is drawn as one
weight — no outline-plus-fill. It should survive being stamped in a single
color at 16 px.

Family it belongs to: colibrì is the small hummingbird this forks from, so
sabrewing's bird should read as the *same species of drawing, one size up* —
heavier body, longer blade, same iridescence. Not a rival animal.

### Rejected alternatives, and why

- **A literal sabre / crossed swords.** Loses the bird, gains a
  vaguely-military vibe this project doesn't want.
- **A brain / neural-net node graph.** Generic AI-slop iconography; every
  inference project has one.
- **A hummingbird in a full dive.** Fast-looking, but it argues for peak
  single-stream speed — the opposite of what sabrewing optimizes.
- **A chip / GPU die with wings.** Dates instantly and reads as hardware
  marketing.

## Palette

Straight from the stylesheet tokens in `web/src/style.css` (oklch there, hex
here for tools that need it):

| Role | Token | Hex |
|---|---|---|
| Deepest background | `--night-0` | `#0a0c1a` |
| Panel / plate | `--night-1` | `#111526` |
| Raised, unlit feather-tiles | `--night-2` | `#1c2035` |
| Hairline / tile borders | `--night-3` | `#2b314a` |
| **Iridescent wing (primary)** | `--violet` | `#a287fe` |
| Wing, shaded | `--violet-dim` | `#7d6ac0` |
| **Gorget ember (accent)** | `--gorget` | `#ff8244` |
| Ember, shaded | `--gorget-dim` | `#db703b` |
| Type / high-contrast fill | `--mist` | `#f0f1f9` |

Ratio discipline: violet dominates, ember is a *spark* — the throat patch, one
lit tile, the eye highlight. If more than ~10% of the mark is orange, it's
wrong. Gradient, where used, is the existing `--wing`: violet → ember at 135°,
with the ember landing off-canvas so only its warm edge shows.

## Construction rules

- **24 × 24 grid**, mark occupying 22 × 22 with 1 unit of optical padding; scale
  to 64, 512, 1024 from there.
- **One weight.** Filled shapes, no strokes, no outlines around fills.
- **Minimum feature 1/24 of the frame.** Anything thinner disappears in a
  favicon.
- **Three legibility gates**, all mandatory: 16 px single-color, 32 px
  two-color, and 1024 px full detail. If the lattice muddies at 32 px, the
  32 px variant drops to a solid wing with three lit tiles.
- **Flat.** No bevels, no drop shadows, no gloss, no 3D render, no photographic
  depth of field. The iridescence is *color*, not lighting.

---

## The master prompt

Use this for the primary mark (image model or human illustrator). It is written
to be read start-to-finish; don't shuffle the clauses, the ordering is
subject → silhouette → wing → color → medium → constraints.

> A flat vector logo mark of a single hummingbird — a sabrewing — seen in
> sharp profile from the left side, facing right, holding a stationary hover.
> The body is a compact, confident wedge: a rounded chest tapering to a short
> tail of three straight feather blades, a small round head, and a long,
> slightly downcurved needle beak pointing forward and just barely upward, so
> the bird reads as poised and attentive rather than aggressive. One visible
> eye, a simple dark almond with a single tiny warm-orange catchlight; no
> pupil detail, no eyelashes, no expression drawn beyond the eye. The throat
> carries a small gorget patch of warm ember orange, the only warm mass in
> the mark.
>
> The wing is the subject of the drawing. It sweeps up and back from the
> shoulder in one long, continuous, sabre-shaped arc — a scimitar curve,
> thick and confident at the shoulder, tapering to a fine point that
> overshoots the tail — so that at a glance the shape reads as a curved blade
> and only on second look as a wing. The leading edge of that arc is a single
> unbroken line; the trailing edge is subdivided into roughly twenty small
> quadrilateral feather tiles, arranged in two staggered rows that follow the
> curve, each tile separated from its neighbors by a thin dark hairline, like
> a mosaic or a stained-glass lattice. Most of those tiles are dark, cool,
> near-invisible slate — barely lifted off the background. Exactly five tiles
> are lit bright iridescent violet, scattered irregularly along the arc, not
> in a row and not evenly spaced, and exactly one tile near the shoulder is
> lit warm ember orange. The lit tiles should feel selected rather than
> decorative — as if a handful of cells in a large grid were switched on.
> A second, smaller wing is implied behind the body as a plain darker silhouette
> of the same curve, offset slightly, with no tiles, purely to give depth.
>
> Behind and below the bird, a thin single-weight violet line traces a
> horizontal figure-eight — a lemniscate — the path of the hovering wingbeat.
> It passes behind the body, is broken where the body and tail cross it so
> the bird stays clearly in front, and fades toward its outer ends. It is
> delicate: roughly one-third the visual weight of the wing's leading edge.
> No motion-blur streaks, no speed lines, no particles, no sparkles.
>
> Color: a very dark indigo-black background (#0a0c1a). Unlit feather tiles in
> #1c2035 with #2b314a hairlines. Wing body, lit tiles, beak, and the
> lemniscate in iridescent violet #a287fe, with #7d6ac0 for the shaded
> under-wing and the implied second wing. Gorget patch, the single lit
> shoulder tile, and the eye catchlight in warm ember #ff8244. No other hues
> anywhere — no cyan, no green, no pink, no gold. The overall impression is a
> violet bird at dusk with one warm spark at its throat.
>
> Medium and finish: flat vector illustration, geometric and precise, built
> from clean bezier curves with crisp edges and no texture. Solid fills only —
> no gradients except optionally one subtle violet-to-ember sweep across the
> wing arc that never fully reaches orange. No outlines around filled shapes,
> no drop shadows, no glow, no bevel, no gloss, no 3D rendering, no
> photorealism, no painterly brushwork, no halftone, no grain. The style sits
> between a modern open-source project mark and a field-guide plate: accurate
> bird anatomy, radically simplified.
>
> Composition: centered in a square frame, the bird occupying about 85% of the
> canvas with even optical padding, the wing arc's tip and the beak tip both
> comfortably inside the margins. The mark must remain legible when reduced to
> 16 by 16 pixels and must still be recognizable if flattened to a single
> flat color with no shading whatsoever. No text, no letters, no numerals, no
> wordmark, no border, no circle badge, no frame, no background scenery, no
> flowers, no feeder, no perch, no other animals, no human hands, no gradient
> mesh backgrounds, no watermark, no signature.

### If your tool takes a separate negative prompt

> text, letters, watermark, signature, photorealism, 3D render, glossy,
> bevel, drop shadow, outer glow, lens flare, sparkles, particles, motion
> blur, speed lines, gradient mesh, noise, grain, halftone, sketchy lines,
> outlined fills, mascot cartoon eyes, cute chibi proportions, two facing
> birds, flock, flowers, feeder, branch, circle badge, shield, ribbon banner,
> rainbow colors, cyan, teal, green, pink, gold, neon glow, cluttered
> background, isometric, low-poly triangulation of the whole body

---

## Variant prompts

### App icon / favicon (16–32 px)

> The same sabrewing hummingbird mark, radically reduced for a favicon. Keep
> only: the wedge body, the head, the beak, and the single sabre wing arc as
> one solid shape. Delete the feather-tile lattice entirely and replace it with
> exactly three violet notches cut into the wing's trailing edge. Delete the
> figure-eight trail. Keep the ember gorget as a single small triangle at the
> throat. Flat violet #a287fe on dark indigo #0a0c1a, one accent in ember
> #ff8244. Chunky, generous negative space, minimum feature thickness roughly
> 1/16 of the frame, must remain readable at 16 by 16 pixels. Square frame, no
> rounded-rectangle plate, no text.

### Horizontal wordmark lockup

> A horizontal logo lockup: the sabrewing hummingbird mark on the left, then
> optical whitespace equal to the height of the bird's body, then the single
> lowercase word "sabrewing" in a clean geometric sans-serif with generous
> letter-spacing, in near-white #f0f1f9 on dark indigo #0a0c1a. The word is
> set so its cap-height matches the height of the bird's body — not the wing
> tip — so the wing arc rises above the text line and the whole lockup reads
> as one shape with a rising diagonal. The bird's beak points toward the first
> letter. Optionally the crossing point of the "w" picks up a single violet
> tint; nothing else in the type is colored. No tagline, no border, no
> registered-trademark symbol.

### README hero banner (wide)

> A wide banner image, roughly 3:1, very dark indigo background. The sabrewing
> hummingbird mark sits left of center, large. Extending to the right from the
> tip of its wing, a sparse field of small dark quadrilateral tiles in the
> same lattice geometry drifts apart and thins out toward the right edge, like
> the wing dissolving into a grid — most tiles dark slate, a scattered few lit
> violet, one or two ember. The density falls off smoothly so the right third
> is nearly empty, leaving clean space for a title. Flat vector, no gradients
> in the background, no glow, no text.

### Sticker / friendly mascot (for the "Sabre" character, not the logo)

> The same sabrewing hummingbird, redrawn as a friendly die-cut sticker
> character: slightly larger head, rounder chest, the same sabre wing arc but
> softened, a small confident tilt to the head, still in strict profile facing
> right. Add a thick flat off-white keyline around the whole silhouette as a
> sticker border. Same palette — violet body, ember gorget, dark indigo
> details. Cheerful and competent, not cutesy: no huge sparkly anime eyes, no
> blush marks, no tiny arms, no accessories, no speech bubble. Flat vector,
> solid fills, no shading.

### Single-color / monochrome stamp

> The sabrewing mark rendered as one solid flat silhouette in a single color
> on a transparent background — body, head, beak, tail, and sabre wing arc
> fused into one connected shape, no internal detail except a single
> negative-space notch for the eye and three negative-space slits in the
> wing's trailing edge. Nothing else. Must read at 16 px and survive being
> stamped, embroidered, or laser-etched.

---

## After generation

Whatever comes out of an image model is a *reference*, not a shippable logo.
The shipping path:

1. Pick the best raster, then **redraw it as clean SVG paths by hand** (or
   trace and then rebuild the beziers). The final `web/public/sabrewing.svg`
   should be a handful of paths, no embedded raster, no filters.
2. Recolor with `currentColor` where possible so the mark inherits light/dark
   theme like the rest of the UI.
3. Run the three legibility gates (16 px mono, 32 px two-color, 1024 px full).
   The 32 px variant is a separate file if the lattice muddies.
4. Drop the results in `docs/media/` and replace both `web/public/sabrewing.svg`
   and `web/dist/sabrewing.svg`.
5. Add the wordmark lockup to the top of `README.md` only if it survives at
   GitHub's rendered width in both GitHub themes.
