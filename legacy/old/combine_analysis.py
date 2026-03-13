#!/usr/bin/env python3
"""
Combine shard CSVs → filtered, sorted, and neatly ordered.

- Include only rows where pretest == "OK" and overall is finite.
- Sort by overall (ascending, lower is better).
- Column order:
    1) overall
    2) trap/detrap energies: A_detrap, B_detrap, C_detrap, D_detrap,
                             A_trap,   B_trap,   C_trap,   D_trap
    3) attempt freqs (formatted like 1e13): <layer>_<trap|detrap>_attemptfreq
    4) T250_* columns (T250_score, T250_shift, T250_corr, T250_overlap)
    5) everything else (e.g., sig, other Txxx_* metrics, params not listed)

Usage:
    python combine_analysis.py --csv-dir /path/to/shard_csv --out /path/to/all_scores_curated.csv
"""

from __future__ import annotations
import argparse, csv, math
from pathlib import Path

# ----- helpers -----
def is_finite(x: str) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False

def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def fmt_scientific_intlike(x: str) -> str:
    """
    Format numeric string as '1e13' style (no + sign, no .0 mantissa).
    If not numeric, return original.
    """
    try:
        v = float(x)
    except Exception:
        return x
    # format as scientific with no mantissa decimals when close to integer power-of-ten
    s = f"{v:.0e}".replace("+", "")
    # collapse '1e013' -> '1e13'
    parts = s.split("e")
    if len(parts) == 2:
        mant, exp = parts
        # strip any trailing '.0' just in case (though .0e handled)
        if mant.endswith(".0"):
            mant = mant[:-2]
        if exp.startswith("0") and len(exp) > 1:
            exp = exp.lstrip("0") or "0"
        s = f"{mant}e{exp}"
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="shard_csv", help="Directory with scores_shard_*.csv")
    ap.add_argument("--out", default="all_scores_curated.csv", help="Output CSV path")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir).resolve()
    out_path = Path(args.out).resolve()

    files = sorted(csv_dir.glob("scores_shard_*.csv"))
    if not files:
        print(f"[combine] No shard CSVs found in {csv_dir}")
        return

    # 1) Read and collect rows + union of columns
    rows = []
    all_cols = set()
    for fp in files:
        with open(fp, "r", newline="") as f:
            r = csv.DictReader(f)
            if r.fieldnames is None:
                continue
            for row in r:
                # Filter: pretest == OK and finite overall
                if row.get("pretest", "").upper() != "OK":
                    continue
                ov = row.get("overall", "")
                if not is_finite(ov):
                    continue
                rows.append(row)
                all_cols.update(r.fieldnames)

    if not rows:
        print("[combine] No passing rows (pretest==OK with finite overall). Nothing to write.")
        return

    # 2) Sort by overall ascending
    rows.sort(key=lambda d: to_float(d.get("overall", "nan")))

    # 3) Build the desired column order
    # Param names you use in shard CSV: 'param_<name>'
    energies_order = [
        "A_detrap","B_detrap","C_detrap","D_detrap",
        "A_trap",  "B_trap",  "C_trap",  "D_trap",
    ]
    # Attempt freqs: both trap and detrap frequencies (if present)
    # We'll discover them dynamically but keep a stable order by layer, then trap/detrap
    layers = ["A","B","C","D"]
    freq_kinds = ["trap_attemptfreq","detrap_attemptfreq"]

    overall_col = ["overall"]

    energy_cols = [f"param_{k}" for k in energies_order if f"param_{k}" in all_cols]

    freq_cols = []
    for L in layers:
        for kind in freq_kinds:
            col = f"param_{L}_{kind}"
            if col in all_cols:
                freq_cols.append(col)

    # T250 block (put right after attempts)
    t250_cols = []
    for suffix in ["score","shift","overlap","mae_plateau","mae_rise","mae_return"]:
        col = f"T250_{suffix}"
        if col in all_cols:
            t250_cols.append(col)

    # Minimal always-include set:
    ordered = overall_col + energy_cols + freq_cols + t250_cols

    # 4) Append any remaining columns (stable-ish: alphabetical of leftovers)
    remaining = [c for c in sorted(all_cols) if c not in ordered]
    fieldnames = ordered + remaining

    # 5) Write output with formatting for attempt freqs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = {}
            for col in fieldnames:
                val = row.get(col, "")
                if col in freq_cols and val != "":
                    val = fmt_scientific_intlike(val)  # render as 1e13 style
                out[col] = val
            w.writerow(out)

    print(f"[combine] Wrote {len(rows)} rows → {out_path}")
    print(f"[combine] Columns ({len(fieldnames)}):")
    print(", ".join(fieldnames))

if __name__ == "__main__":
    main()
