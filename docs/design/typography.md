# Typography — COMP702 Proposal / Dissertation / Slides

Type system for Word/PDF/PowerPoint deliverables in this project, per `palette.md`'s
"Oxford Ink" identity. This reflects the scheme actually built into
`artifacts/Proposal_v1.0.0.docx`: a classic academic serif for reading text,
with a clean sans reserved for tabular and structural elements so data stays
easy to scan against the prose.

## Fonts

| Role | Font | Rationale |
|---|---|---|
| Body text, references, callout body | **Garamond** | Serif font for high readability in dense academic prose. |
| All headings (H1/H2), title page, tables, diagrams, captions, header, footer | **Calibri** | Clean sans-serif for clear structural hierarchy and scannable data. |
| Code / monospace | Consolas | For technical strings and code snippets. |

## Sizes (Word / PDF, A4, 2.5cm margins)

| Element | Font | Size | Weight / Style | Colour |
|---|---|---|---|---|
| Title (cover page) | Calibri | 32pt | Bold | Blue 900 `#0A1F33` |
| Subtitle (cover page) | Calibri | 13pt | Regular | Blue 700 `#15406B` |
| Cover page labels | Calibri | 11.5pt | Regular | Neutral 700 `#44494F` |
| Cover page names | Calibri | 13.5pt | Bold | Neutral 900 `#1A1D21` |
| Institution lines | Calibri | 11.5pt | Bold, all caps | Blue 900 `#0A1F33` |
| H1 (numbered & unnumbered) | Calibri | 20pt | Bold | Blue 900 `#0A1F33`, 1pt Blue 700 bottom rule |
| H2 / sub-section | Calibri | 15pt | Bold | Blue 700 `#15406B` |
| Body text & bullet lists | Garamond | 12pt | Regular | Neutral 900 `#1A1D21` |
| Callout box | Garamond | 11.5pt | Regular, lead-in bold | Blue 900 on Gold 100 wash |
| Reference list entries | Garamond | 12pt | Regular | Neutral 900, 0.18cm hanging indent |
| Table header row | Calibri | 10pt | Bold | White on Blue 700 |
| Table body | Calibri | 9.5pt | Regular | Neutral 900 |
| Diagram / flow-cell labels | Calibri | 9.5pt | Bold | White or Blue 900, per fill |
| Figure / table caption | Calibri | 10pt | Regular | Neutral 700 `#44494F` |
| Header / Footer | Calibri | 9pt | Regular (Header all caps) | Neutral 500 `#767C84` |

## Slide deck (PowerPoint)

| Element | Font | Size | Weight |
|---|---|---|---|
| Title slide title | Garamond | 36pt | Bold |
| Section divider title | Garamond | 28pt | Bold |
| Slide title | Garamond | 24pt | Bold |
| Slide body | Calibri | 16pt | Regular |
| Chart/diagram label | Calibri | 12pt | Regular |
| Footer / slide number | Calibri | 10pt | Regular |

## Paragraph and layout rules

- **Spacing & Alignment:** Body paragraphs use 1.5 line spacing, justified alignment with document-wide auto-hyphenation enabled (to prevent white-space rivers), 9pt space-after, block spacing (no first-line indent). 
- **Italics:** Strictly restricted to required academic conventions (e.g., book titles, *in vivo*, *et al.*). Do not use for general emphasis, headings, subheadings or captions.
- **Punctuation:** Use em-dashes (–) strictly for number/date ranges (e.g., pp. 157–173). Reserve hyphens solely for compound words. **No em-dashes**.
- **H1:** 22pt space-before, 11pt space-after, 1pt Blue 700 bottom-border rule. Keep-with-next enabled.
- **H2:** 13pt space-before, 7pt space-after.
- **Bullets:** 5.5pt space-after, 1.5 line spacing, 0.4cm hanging indent.
- **Cover page:** 1.25pt Blue 700 rule beneath the title, 0.5pt Gold 500 rule above the institution block. Fit to a single A4 page.
- **Callout boxes:** Gold 100 fill, 3.5pt Gold 500 left border, used sparingly for critical facts.
- **Table of Contents:** Left `dirty` with `updateFields` enabled in `settings.xml` for auto-population.
- **Language:** British English spelling throughout. Spoken punctuations. NO COMPLICATED PUNCTUATION (unless required). Reserve the ";", ":" to use minimally.
- **Paragraph Breaks**: DO NOT break the paragraphs/diagrams/tables/etc into two pages. Either reduce the paragraph size or expand the previous paragraphs to force the section to go on a new page. THIS RULE MUST BE STRICTLY FOLLOWED FOR DIAGRAMS/TABLES/FIGURES/CHARTS/ANY OTHER VISUAL ELEMENTS.