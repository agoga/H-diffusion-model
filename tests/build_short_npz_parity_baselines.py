"""Build short-matrix legacy NPZ baselines for standard NPZ parity tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from tests.parity_framework import run_legacy_runtime_trace, MeasureSpec
    from tests.parity_short_matrix import LAYERS, SHORT_PARAM_SETS, SHORT_SCHEDULE_CASES, STAGES, matrix_key
except ModuleNotFoundError:
    from parity_framework import run_legacy_runtime_trace, MeasureSpec
    from parity_short_matrix import LAYERS, SHORT_PARAM_SETS, SHORT_SCHEDULE_CASES, STAGES, matrix_key


def main() -> None:
    out_root = Path(__file__).resolve().parent / "golden" / "legacy_series_short"
    out_root.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}

    for param_id, base_params in SHORT_PARAM_SETS.items():
        for case in SHORT_SCHEDULE_CASES:
            schedule_spec = f"{case.fire_s}:{case.fire_C}C, {case.anneal_s}:{case.anneal_C}C"
            for stage in STAGES:
                for layer in LAYERS:
                    measure = MeasureSpec(layer=layer, kind="total", stage=stage, units="cm^-3")
                    trace = run_legacy_runtime_trace(
                        schedule_spec=schedule_spec,
                        stage_names=list(STAGES),
                        measure=measure,
                        base_params=base_params,
                    )
                    key = matrix_key(param_id=param_id, case_id=case.case_id, stage=stage, layer=layer)
                    rel = f"{key}.npz"
                    out = out_root / rel
                    meta = {
                        "param_id": param_id,
                        "case_id": case.case_id,
                        "stage": stage,
                        "layer": layer,
                        "schedule_spec": schedule_spec,
                        "source": trace.source,
                    }
                    np.savez_compressed(
                        out,
                        t_s=np.asarray(trace.t_s, dtype=float),
                        y=np.asarray(trace.y, dtype=float),
                        metadata_json=json.dumps(meta, sort_keys=True),
                    )
                    index[key] = rel
                    print(f"wrote {out}")

    (out_root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    print(f"wrote {out_root / 'index.json'}")


if __name__ == "__main__":
    main()
