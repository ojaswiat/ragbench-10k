# Brand Palette — "Oxford Ink"

A premium, academic, blue-monochrome identity for all proposal, dissertation,
presentation, and diagram work in this project. The system is built on a
single blue hue (varied by lightness), a matching blue-tinted neutral gray
scale, and one reserved gold accent for emphasis only.

Use this document as the single source of truth — when building a Word/LaTeX
template, a PowerPoint master, or a draw.io style library, pull colors from
here rather than re-deriving them.

## 1. Core scales

### 1.1 Primary Blue

| Token | Hex | Use |
|---|---|---|
| Blue 950 | `#06121F` | deepest shadow, rarely used |
| Blue 900 — Ink | `#0A1F33` | H1, title-slide background, darkest emphasis |
| Blue 800 | `#0F2D4D` | intermediate shade |
| Blue 700 — Primary | `#15406B` | H2, section-divider slides, primary brand color |
| Blue 600 | `#1C5485` | intermediate shade |
| Blue 500 — Mid | `#2B6CA3` | links, citations, accent icons |
| Blue 400 | `#5089B8` | intermediate tint |
| Blue 300 — Soft | `#82AAC9` | subtitle text on dark backgrounds, light chart fills |
| Blue 200 | `#B4CBDE` | intermediate tint |
| Blue 100 — Tint | `#DCE6EE` | callout box backgrounds, table tint |
| Blue 50 — Paper | `#F2F6FA` | page/slide background, alternative to white |

### 1.2 Neutral (blue-gray)

Desaturated and blue-tinted so it harmonizes with the primary scale rather
than reading as a generic gray.

| Token | Hex | Use |
|---|---|---|
| Neutral 900 | `#1A1D21` | body text |
| Neutral 700 | `#44494F` | captions, footnotes, connector/arrow lines |
| Neutral 500 | `#767C84` | muted text, axis labels, footers, page numbers |
| Neutral 300 | `#B7BCC2` | borders, grid lines, dividers |
| Neutral 100 | `#E4E6E9` | subtle fills, disabled states, code block background |
| Neutral 50 | `#F7F8F9` | off-white paper alternative |
| White | `#FFFFFF` | base background |

### 1.3 Gold accent — reserved

Used **only** for deliberate emphasis (a key finding, a premium section
divider, a critical-path highlight). Never used for routine data, body
content, or as a general-purpose color. If gold appears more than a
handful of times on a page, it has stopped being an accent.

| Token | Hex | Use |
|---|---|---|
| Gold 700 | `#8C6D00` | text-on-light emphasis |
| Gold 500 — Primary | `#D4A017` | primary accent — key callouts, dividers |
| Gold 300 | `#E8C667` | light accent fill |
| Gold 100 | `#FBF0D6` | highlight-box background wash |

### 1.4 Auxiliary evaluation colors

Outside the core brand identity. Use **only** for pass/fail or
correct/incorrect indicators in evaluation tables and result figures —
never as decorative or categorical colors.

| Token | Hex | Use |
|---|---|---|
| Success | `#0B8457` | correct / pass indicator |
| Error | `#C73E35` | incorrect / fail indicator |

Pair these with a redundant cue (icon, label, or pattern) rather than color
alone, since red/green-adjacent hues are not colorblind-safe on their own.

## 2. Charting

### 2.1 Categorical (5+ series)

Blue-dominant with a subtle hue drift, so each series stays in the cool
family but remains clearly distinguishable in bar, pie, and scatter charts.
Assign in this order:

| # | Name | Hex |
|---|---|---|
| 1 | Primary Blue | `#15406B` |
| 2 | Slate Indigo | `#3B4F7A` |
| 3 | Teal Blue | `#2B7A8C` |
| 4 | Cool Slate | `#5C6670` |
| 5 | Soft Blue | `#7FA6C9` |
| 6 (last resort) | Ink | `#0A1F33` |
| 7 (last resort) | Gold | `#D4A017` |

Categories 6 and 7 should only be used when a 6th or 7th series is
unavoidable — they exist for maximum contrast and reserved emphasis,
respectively, not as default palette members.

### 2.2 Sequential (ordinal data, heatmaps)

Single-hue ramp, light to dark:

```
#DCE6EE → #B4CBDE → #82AAC9 → #2B6CA3 → #15406B → #0A1F33
```

### 2.3 Diverging (delta / baseline-relative data)

```
#D4A017 → #E8C667 → #E4E6E9 (midpoint) → #82AAC9 → #15406B
```

Gold marks the negative pole, blue the positive pole — keeping the warm
accent purposeful even when used in a chart.

## 3. Component mapping

### 3.1 Dissertation / Proposal (Word, LaTeX, PDF)

| Element | Spec |
|---|---|
| Page background | White `#FFFFFF` (print) or Blue 50 `#F2F6FA` (digital/PDF) |
| H1 / chapter title | Blue 900 `#0A1F33`, 1pt rule beneath in Blue 700 |
| H2 / section heading | Blue 700 `#15406B` |
| H3 / subsection heading | Blue 500 `#2B6CA3` |
| H4 / minor heading | Neutral 900 `#1A1D21`, bold weight only |
| Body text | Neutral 900 `#1A1D21` |
| Captions / footnotes | Neutral 700 `#44494F` |
| Links, citations, cross-refs | Blue 500 `#2B6CA3` |
| Table header row | fill Blue 700 `#15406B`, text White |
| Table zebra striping | White / Blue 50 `#F2F6FA` |
| Table borders / rules | Neutral 300 `#B7BCC2` |
| Blockquote / callout box | bg Blue 100 `#DCE6EE`, left border 4px Blue 700 |
| Key-finding / highlight box | bg Gold 100 `#FBF0D6`, left border 3.5pt Gold 500 |
| Code / monospace block | bg Neutral 100 `#E4E6E9`, border 1px Neutral 300 |
| Page footer / page numbers | Neutral 500 `#767C84` |
| Pass / correct (eval tables only) | Success `#0B8457` |
| Fail / incorrect (eval tables only) | Error `#C73E35` |

### 3.2 PowerPoint

| Element | Spec |
|---|---|
| Title slide background | Blue 900 (Ink) `#0A1F33`, title text White |
| Title slide subtitle/byline | Blue 300 `#82AAC9` |
| Section-divider slide background | Blue 700 (Primary) `#15406B`, text White |
| Content slide background | White or Blue 50 `#F2F6FA` |
| Content slide heading | Blue 700 `#15406B` |
| Content slide body text | Neutral 900 `#1A1D21` |
| Bullet markers / accent dashes | Blue 500 `#2B6CA3` |
| Footer / slide number — light slides | Neutral 500 `#767C84` |
| Footer / slide number — dark slides | Blue 300 `#82AAC9` |
| Key-metric callout | Gold 500 text or Gold 100 background wash |
| Charts / figures | per Section 2 |

### 3.3 Flowcharts / diagrams

| Shape | Fill | Border | Text |
|---|---|---|---|
| Terminator (start/end) | Blue 700 `#15406B` | — | White |
| Process step | Blue 100 `#DCE6EE` | 2px Blue 700 | Blue 900 |
| Decision (diamond) | Blue 500 `#2B6CA3` | — | White |
| Input/Output, data store | Blue 300 `#82AAC9` | Blue 500 | Blue 900 |
| Subprocess / module | Slate Indigo `#3B4F7A` | — | White |
| Annotation / note | Neutral 100 `#E4E6E9` | Neutral 300 | Neutral 700 |

Connectors and arrows: Neutral 700 `#44494F`. Critical-path or single-point
emphasis: Gold 500 `#D4A017` stroke, 2–3px, used sparingly.

### 3.4 Tables (general — specs, READMEs, standalone documents)

| Element | Spec |
|---|---|
| Header row | fill Blue 700, text White |
| Zebra striping | White / Blue 50 |
| Borders / grid | Neutral 300 |
| Highlighted / key cell | bg Gold 100, text Blue 900, optional Gold 500 border |

## 4. Usage discipline

- **Roughly 60/30/10**: Neutral + Ink for text and structure (~60%),
  Primary/Mid blue for headings and accents (~30%), Gold for emphasis
  (<10%, and only where it earns attention).
- Never introduce a new hue outside this system (no new reds, greens, or
  purples) except the two auxiliary evaluation colors, which are scoped
  strictly to pass/fail indicators.
- Gold is an accent, not a category — don't assign it to a routine 3rd or
  4th data series; it's reserved for the categorical palette's last-resort
  slot (Section 2.1) and for emphasis elements (Section 1.3).
- Keep dark backgrounds (Ink `#0A1F33`, Primary `#15406B`) limited to title
  slides and section dividers, so the bulk of the document/deck stays light
  and print-consistent.
