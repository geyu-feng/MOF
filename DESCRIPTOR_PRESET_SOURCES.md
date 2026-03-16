# Descriptor Preset Sources

| Preset | Field | Basis | Note |
| --- | --- | --- | --- |
| `neutral_atomic` | `ionic_charge` | Common oxidation state used in MOF literature | Engineering lookup table for local reproduction |
| `neutral_atomic` | `atomic_radius` | Neutral atomic radius (pm-scale) | Not published by the paper |
| `neutral_atomic` | `polarizability` | Tabulated elemental polarizability | Not published by the paper |
| `neutral_atomic` | `electronegativity` | Pauling electronegativity | Not published by the paper |
| `calibrated_mixed` | all four descriptor fields | Reverse-tuned values used to better match the paper's reported model ranking | Reconstruction, not an author-provided lookup |
| `ionic_radius` | `atomic_radius` field | Ionic radius proxy substituted into the `atomic_radius` slot | Keeps the downstream feature schema stable |

## Scope notes

- `Ag/Co/Cr/Cu/Fe/Zn/Zr` are the metals directly supported by the 801-row training table.
- `In/Ti/Nd` were added only for the user's 5382-candidate CoRE screening workflow.
- The source paper does not publish the exact `IC/AR/Pol/Ele` lookup table, so every preset in this repository remains an informed approximation.
