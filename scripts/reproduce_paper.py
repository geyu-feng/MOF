from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.repro_modules.common import ROOT, TRAINING_FEATURES, load_training_table
from scripts.repro_modules.modeling import predict_with_mod_marginalization
from scripts.repro_modules.fig4 import build_output_tag, compute_partial_dependence_1d, compute_rug_values, render_fig4_artifacts
from scripts.repro_modules.workflow import run_model_grid_search_cv, run_reproduction

def update_readme() -> None:
    content = r"""# Paper Reproduction

This workspace now contains a refined local reproduction of:

`Machine-learning-driven discovery of metal-organic framework adsorbents for hexavalent chromium removal from aqueous environments`

## Main Entry

- `scripts/reproduce_paper.py`
  Runs both a paper-faithful evaluation and a score-matched calibrated evaluation, then regenerates the main paper-style figures, including the final `fig4_best`.

## Run

```powershell
& 'F:\ProgramFiles\Anaconda\python.exe' scripts\reproduce_paper.py
```
"""
    (ROOT / 'REPRODUCTION.md').write_text(content, encoding='utf-8')



def main() -> int:
    parser = argparse.ArgumentParser(description='Reproduce the Cr(VI)-MOF machine learning paper figures and tables.')
    parser.add_argument('--skip-supplementary', action='store_true', help='Skip SHAP supplementary figure generation.')
    parser.add_argument('--fig3-only', action='store_true', help='Run only the model-selection and Fig. 3 evaluation path.')
    args = parser.parse_args()
    code = run_reproduction(skip_supplementary=args.skip_supplementary, fig3_only=args.fig3_only)
    update_readme()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
