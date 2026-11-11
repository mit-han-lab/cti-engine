# Brand assets

Vector exports of the cti design, with text outlined so no font is needed. The mark has a
transparent background; the overview sits on its own light-grey panel.

| File | Size | Used by |
|---|---|---|
| `mark.svg` | 120 × 120 | The mark: the README header, avatars, icons and favicons. |
| `concept.svg` | 217 × 83 | The README's overview: one model serving requests and training on new data. |

## Colour

| | Hex | Used for |
|---|---|---|
| Blue | `#2E64F8` | The mark's upper stroke. |
| Green | `#5BD99E` | The mark's lower stroke. |

## Exports

Raster, for anywhere SVG isn't accepted (a GitHub avatar upload, for one):

```sh
cairosvg assets/mark.svg -o mark-512.png -W 512 -H 512
```
