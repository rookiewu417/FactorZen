# FactorZen Brand Assets

## Logo Concept

The FactorZen mark is a **stacked Z-form** built from two interlocking right-angle strokes inside a rounded square: an upper graphite stroke and a lower violet stroke that fold back on each other. It reads as the `Z` of *Zen* and as an iterative research loop — signal routed through, refined, and returned.

The mark is designed for two-tone rendering and adapts to background: on light surfaces the upper stroke is graphite and the lower stroke is iris violet; on dark surfaces the strokes lighten to off-white and a softer violet so the mark keeps contrast without going flat.

## Files

- `logo-horizontal-light.svg`: README and light backgrounds.
- `logo-horizontal-dark.svg`: dark report covers, slides, and dark UI surfaces.
- `logo-mark-light.svg`: compact square mark for light backgrounds.
- `logo-mark-dark.svg`: compact square mark for dark backgrounds.
- `logo-icon-light-512.png`: 512×512 app icon on a light field (stores, launchers, social avatars).
- `logo-icon-dark-512.png`: 512×512 app icon on a dark field.
- `logo.svg`: compatibility alias for the light horizontal logo.
- `logo-mark.svg`: compatibility alias for the light mark.
- `favicon.svg`: square favicon; auto-adapts to light/dark via `prefers-color-scheme`.
- `favicon-32.png`, `favicon-16.png`: raster favicon fallbacks for legacy browsers.

## Brand Colors

| Role | Hex | Usage |
| --- | --- | --- |
| Graphite Ink | `#16181D` | Upper stroke and wordmark on light surfaces; background on dark surfaces |
| Field White | `#EDEFF2` | Upper stroke and wordmark on dark surfaces; clean report backgrounds |
| Iris Violet | `#6B4EFF` | Lower stroke — the accent — on light backgrounds |
| Lift Violet | `#8E7BFF` | Lower stroke on dark backgrounds and in dark-mode favicon |

## Font Recommendations

- Latin UI / wordmark: `Aptos Display`, `Aptos`, `Segoe UI`.
- Chinese UI: `Noto Sans SC`, `Microsoft YaHei`, `Source Han Sans SC`.
- Code, run IDs, and manifest metadata: `JetBrains Mono`, `Consolas`, monospace.

Use a heavy wordmark for the project name and regular or medium weights for interface copy. Keep letter spacing at `0` for normal text.

## Usage Notes

- Use the light logo on white or neutral light backgrounds.
- Use the dark logo on graphite, dark report covers, or slide backgrounds.
- Prefer `favicon.svg` for browser tabs and small app icons; ship `favicon-32.png` / `favicon-16.png` for browsers that reject SVG favicons.
- Use `logo-icon-*-512.png` for app-store listings, launcher tiles, and social avatars that require a raster square.
- Keep the two strokes as a single locked unit; do not recolor, separate, or rotate them.
- Do not pair the logo with candlestick charts, trading arrows, coin icons, bull/bear symbols, glossy gradients, brush circles, ornamental Zen motifs, generic chart bars, or stamp-style badges.
