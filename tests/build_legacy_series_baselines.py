"""Generate frozen legacy-series NPZ baselines used by NPZ parity tests."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from tests.parity_framework import (
        LegacyRuntimeBaselineSource,
        MeasureSpec,
        ParityCase,
        save_legacy_series_baseline,
    )
except ModuleNotFoundError:
    from parity_framework import (
        LegacyRuntimeBaselineSource,
        MeasureSpec,
        ParityCase,
        save_legacy_series_baseline,
    )


def main() -> None:
    cases = [
        ParityCase(name="anneal_225", fire_C=650.0, anneal_C=225.0, fire_s=10.0, anneal_s=8_000_000.0),
        ParityCase(name="anneal_250", fire_C=650.0, anneal_C=250.0, fire_s=10.0, anneal_s=8_000_000.0),
        ParityCase(name="anneal_350", fire_C=650.0, anneal_C=350.0, fire_s=10.0, anneal_s=8_000_000.0),
        ParityCase(name="fire_750", fire_C=750.0, anneal_C=250.0, fire_s=10.0, anneal_s=8_000_000.0),
    ]
    measure = MeasureSpec(layer="C", kind="trapped", stage="annealing", units="cm^-3")
    baseline_source = LegacyRuntimeBaselineSource()

    out_root = Path(__file__).resolve().parent / "golden" / "legacy_series"
    out_root.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}

    for case in cases:
        trace = baseline_source.get_trace(case, measure)
        rel = f"{case.name}.npz"
        out_file = out_root / rel
        save_legacy_series_baseline(
            out_file=out_file,
            case=case,
            measure=measure,
            trace=trace,
        )
        index[case.name] = rel
        print(f"wrote {out_file}")

    (out_root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    print(f"wrote {out_root / 'index.json'}")


if __name__ == "__main__":
    main()
