# Paper Reproduction

This workspace reproduces:

`Machine-learning-driven discovery of metal-organic framework adsorbents for hexavalent chromium removal from aqueous environments`

## Main Entry

- `scripts/reproduce_paper.py`
  Compatible CLI entry. It now stays thin and delegates implementation to internal modules.

## Module Map

- `scripts/repro_modules/common.py`
  Constants, schemas, preprocessing, data loading, CoRE table preparation, shared helpers.
- `scripts/repro_modules/modeling.py`
  Model registry, grouped CV, split logic, holdout fitting, screening-stage prediction helpers.
- `scripts/repro_modules/plots.py`
  Fig. 2, Fig. 3, Fig. 5, supplementary figures, PDF/page extraction helpers.
- `scripts/repro_modules/fig4.py`
  Fig. 4-specific model context, PDP computation, rug marks, uncertainty bundle, export logic.
- `scripts/repro_modules/reporting.py`
  Summary markdown generation.
- `scripts/repro_modules/workflow.py`
  Top-level reproduction workflow orchestration.

## Run

```powershell
& 'F:\ProgramFiles\Anaconda\python.exe' scripts\reproduce_paper.py
```

## Debugging Shortcuts

- Load training data only:
  `from scripts.reproduce_paper import load_training_table`
- Run model selection only:
  `from scripts.reproduce_paper import run_model_grid_search_cv`
- Render Fig. 4 only:
  `from scripts.reproduce_paper import render_fig4_artifacts`
