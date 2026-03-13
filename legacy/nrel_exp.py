# nrel_exp.py
# Utilities for loading and plotting NREL anneal-sweep (J0 vs time) data.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import helpers as h
import vizkit as viz  # for colors & legend handles
# add these two lines below your existing imports
_np = np
_h  = h

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})



def _pick_sim_for_anneal_T(mgr, *, firing_temp_C: int, anneal_temp_C: int, layer: str, kind: str):
    """
    Match a schedule by (fire_C, anneal_C) and return (t_hours, y) for the ANNEALING stage.
    kind ∈ {"mobile","trapped","total"}.
    """
    fire_K   = float(firing_temp_C) + 273.15
    anneal_K = float(anneal_temp_C) + 273.15

    for sched in getattr(mgr, "schedules", []):
        _, cum = _h.parse_temp_schedule(sched)
        if not cum:
            continue
        first_TK = float(cum[0][1])
        last_TK  = float(cum[-1][1])
        if not (_np.isclose(first_TK, fire_K, rtol=0, atol=1e-6) and _np.isclose(last_TK, anneal_K, rtol=0, atol=1e-6)):
            continue

        # sim = mgr.ensure(sched)
        stage = "annealing"
        if   kind == "mobile":  t_s, y = sim.mobile(layer, stage=stage)
        elif kind == "trapped": t_s, y = sim.trapped(layer, stage=stage)
        elif kind == "total":   t_s, y = sim.total(layer, stage=stage)
        else: return None, None
        return (_np.asarray(t_s, float)/3600.0, _np.asarray(y, float))  # hours, series

    return None, None


# ---------- tiny summarizer used by both printouts ----------
def _summarize_sim_series(t_h, y, *, return_tol=0.05):
    """
    Return RAW and NORMALIZED metrics using the same normalization as helpers.normalize_baseline_peak:
      yN = (y - y0) / (ypeak - y0), with y0 from baseline window (0.1–0.3 hr).
    """
    import numpy as np
    t_h = np.asarray(t_h, float); y = np.asarray(y, float)
    if y.size == 0:
        return {"raw": {}, "norm": {}}

    # use the same routine your plots use
    tN, yN, y0, ypk, i_pk = _h.normalize_baseline_peak(t_h, y, exp=False)#t0_window=(0.1, 0.3))
    t_peak = float(tN[i_pk]) if (0 <= i_pk < tN.size) else float("nan")

    # time to return within tol of baseline after the peak (units: hours)
    t_return = _h.time_to_return_baseline(tN, yN, i_pk, tol=return_tol)

    # deepest dip after normalization (can be very negative if below baseline a lot)
    i_min = int(np.nanargmin(yN)) if yN.size else -1
    y_min = float(yN[i_min]) if (0 <= i_min < yN.size) else float("nan")
    t_min = float(tN[i_min]) if (0 <= i_min < tN.size) else float("nan")

    return {
        "raw":  {"y_baseline": float(y0), "y_peak": float(ypk), "delta": float(ypk - y0), "t_peak_hr": t_peak},
        "norm": {"t_peak_hr": t_peak, "t_return_hr": float(t_return), "y_min_norm": y_min, "t_min_hr": t_min},}
# ------------------------- Loaders -------------------------

def load_nrel_anneal_csv(path: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Read the NREL annealing sweep CSV (Google Sheets export).
    Skips first 4 metadata rows and removes pre-anneal points (t < 0.1 hr).
    Returns {temp_C: (time_hours, J0_values)}.
    """
    df = pd.read_csv(path, skiprows=4)
    df = df.dropna(how="all")
    df.columns = [c.strip() for c in df.columns]

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cols = list(df.columns)

    # Pair up Time(hr)_XXXC with J0_XXXC
    for i in range(0, len(cols), 2):
        if i + 1 >= len(cols):
            break
        time_col, j_col = cols[i], cols[i + 1]
        if "Time" not in time_col or "J0" not in j_col:
            continue

        try:
            T = int(time_col.split("_")[-1].replace("C", ""))
        except Exception:
            continue

        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
        j = pd.to_numeric(df[j_col], errors="coerce").to_numpy()
        mask = np.isfinite(t) & np.isfinite(j) & (t >= 0.1)  # skip early pre-anneal points
        if np.count_nonzero(mask):
            out[T] = (t[mask], j[mask])

    return out




# ------------------------- Plotters -------------------------

def plot_expt_anneal_raw(
    expt: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    ylog: bool = False,
    x_left: float | None = None,
    ax=None,
):
    """Plot raw J₀ vs time (hours), colored by anneal temperature."""
    temps = sorted(expt.keys())
    if not temps:
        print("No experimental data.")
        return None

    cmap = viz.map_colors_low_to_high(temps)
    fig, ax = (plt.subplots(figsize=(12, 4)) if ax is None else (ax.figure, ax))

    for T in temps:
        t, j = expt[T]
        ax.plot(t, j, lw=1.6, color=cmap[T], label=f"{T}°C")
        if j.size:
            ipk = int(np.nanargmax(j))
            ax.scatter([t[ipk]], [j[ipk]], s=25, marker="x", linewidths=1.2, color=cmap[T])

    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if x_left is not None:
        ax.set_xlim(left=x_left)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("J₀ (arb. units)")
    ax.set_title("NREL Experimental — anneal sweep (raw)")

    handles = viz.legend_handles(cmap, "{}°C")
    if handles:
        leg = ax.legend(
            handles=handles,
            title="Anneal temp (°C)",
            bbox_to_anchor=(1.05, 0.5),   # legend off-plot on the right
            loc="center left",
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.8,
        )
        leg.get_title().set_fontsize(11)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return ax

def plot_exp_vs_sim_normalized_overlay_hours(
    mgr,
    expt_path,
    *,
    firing_temp_C: int,
    anneal_temps_C=(200, 225, 250, 275, 300, 350),
    layer="C",
    kind="trapped",
    ylog=False,
    x_left=1e-3,
    legend_where="right",
    ax=None,
    print_data: bool = False,
    return_data: bool = False,
    y_clip: tuple[float,float] | None = None,
    plot: bool = True,
    # NEW:
    shift_decades_per_step: float = -1.0,   # -1.0 → each higher temp is 10× earlier (left)
    shift_on_plot_only: bool = True,        # only shift what we draw; keep metrics unshifted
):
    expt = load_nrel_anneal_csv(expt_path)
    temps = [t for t in anneal_temps_C if t in expt]
    if not temps:
        if print_data:
            print("No experimental temps available.")
        return None

    # consistent order
    temps = sorted(temps)
    # decade shift multiplier for rank i (0,1,2,...) → 10^(i*shift_decades_per_step)
    def _shift_factor(i: int) -> float:
        return 10.0 ** (i * shift_decades_per_step)

    # Only make a figure if plotting
    if plot:
        cmap = viz.map_colors_low_to_high(temps)
        fig, ax = (plt.subplots(figsize=(7.5, 4.5)) if ax is None else (ax.figure, ax))

    sim_nums = {}
    for i, T in enumerate(temps):
        # experiment (normalize)
        t_exp_h, j_exp = expt[T]
        tE, yN_E, y0E, ypkE, iE = h.normalize_baseline_peak(t_exp_h, j_exp,exp=True)

        # simulation (normalize)
        t_sim_h, y_sim = _pick_sim_for_anneal_T(mgr, firing_temp_C=firing_temp_C, anneal_temp_C=T, layer=layer, kind=kind)
        if t_sim_h is None:
            continue
        tS, yN_S, y0S, ypkS, iS = h.normalize_baseline_peak(t_sim_h, y_sim, exp=False)

        # --- metrics & scoring (UNSHIFTED) ---
        sim_nums.setdefault(T, {})
        sim_nums[T].update(_summarize_sim_series(t_sim_h, y_sim, return_tol=0.05))

        # Peak-alignment shift (decades) from normalized series
        s_align = 0.0
        if (0 <= iE < len(tE)) and (0 <= iS < len(tS)):
            tE_pk = float(tE[iE]); tS_pk = float(tS[iS])
            if np.isfinite(tE_pk) and np.isfinite(tS_pk) and (tE_pk > 0) and (tS_pk > 0):
                s_align = np.log10(tE_pk) - np.log10(tS_pk)

        try:
            # Force the scan to live around the peak-aligned shift and
            # REQUIRE meaningful coverage of plateau & return
            s = shape_score(
                tE, yN_E, tS, yN_S,
                half_window_dec=12.0,
                n_shifts=1201,
                min_decades=0.3,
                min_pts=8,
            )
            sim_nums[T]["shape_score"] = s
            
        except Exception as e:
            sim_nums[T]["shape_score_error"] = str(e)
            if print_data: print(f"[shape_score fail @ {T}C] {e}")

        # --- plotting (SHIFTED by decades; also peak-align SIM to EXP) ---
        if plot:
            f_temp = _shift_factor(i) if shift_on_plot_only else 1.0

            # apply peak alignment to SIM for visualization
            f_align = 10.0 ** s_align if np.isfinite(s_align) else 1.0

            tE_plot = tE * f_temp
            tS_plot = tS * f_temp * f_align

            # EXP: markers only
            ax.plot(
                tE_plot, yN_E,
                linestyle="none",
                marker="o", markersize=4,
                markerfacecolor=cmap[T],
                markeredgewidth=1.2, markeredgecolor=cmap[T],
                label=f"{T}°C exp",
            )
            # SIM: solid line
            ax.plot(tS_plot, yN_S, ls="-", lw=1.8, color=cmap[T], label=f"{T}°C sim")


    # Plot cosmetics only if plotting
    if plot:
        ax.set_xscale("log")
        if ylog: ax.set_yscale("log")
        # if x_left is not None: ax.set_xlim(left=x_left)
        if y_clip is not None: ax.set_ylim(*y_clip)
        ax.set_xlabel("Time (hours)")
        if legend_where == "right":
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=0.85, ncol=1)
            fig.tight_layout()
        else:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, ncol=min(6, 2*len(temps)))
            fig.tight_layout(rect=[0, 0.08, 1, 1])

    overall = overall_shape_score(sim_nums)
    print(f"score new:{overall}")

    return (ax, sim_nums, overall) if return_data else ax



def plot_exp_vs_sim_tpeak_treturn(
    expt,
    mgr,
    *,
    firing_temp_C: int,
    anneal_temps_C=(200, 250, 300, 350),
    layer="C",
    kind="trapped",
    return_tol=0.05,  # 5% from baseline in normalized space
    ax=None,
    print_data: bool = False,
    return_data: bool = False,
):
    temps = [t for t in anneal_temps_C if t in expt]
    if not temps:
        print("No experimental temps available.")
        return None

    # ---- collect exp/sim numbers using your helpers ----
    tpk_E, tret_E, tpk_S, tret_S = [], [], [], []
    sim_nums = {}
    for T in temps:
        # experiment
        t_exp_h, j_exp = expt[T]
        tE, yN_E, y0E, ypkE, iE = h.normalize_baseline_peak(t_exp_h, j_exp, t0_window=(0.1, 0.3))
        tpk_E.append(h.peak_time_hours(tE, iE))
        tret_E.append(h.time_to_return_baseline(tE, yN_E, iE, tol=return_tol))

        # simulation
        t_sim_h, y_sim = _pick_sim_for_anneal_T(mgr, firing_temp_C=firing_temp_C, anneal_temp_C=T, layer=layer, kind=kind)
        if t_sim_h is None:
            tpk_S.append(np.nan); tret_S.append(np.nan)
        else:
            tS, yN_S, y0S, ypkS, iS = h.normalize_baseline_peak(t_sim_h, y_sim, t0_window=(0.1, 0.3))
            tpk_S.append(h.peak_time_hours(tS, iS))
            tret_S.append(h.time_to_return_baseline(tS, yN_S, iS, tol=return_tol))
            sim_nums[T] = _summarize_sim_series(t_sim_h, y_sim, return_tol=return_tol)

    temps_arr = np.array(temps, float)
    fig, axes = (plt.subplots(1, 2, figsize=(10, 4)) if ax is None else (ax.figure, ax))
    ax_pk, ax_rt = axes if isinstance(axes, np.ndarray) else (axes, axes)

    # Panel 1: t_peak
    ax_pk.plot(temps_arr, tpk_E, "o--", lw=1.6, label="exp", ms=5)
    ax_pk.plot(temps_arr, tpk_S, "s-",  lw=1.6, label="sim", ms=5)
    ax_pk.set_xlabel("Anneal temperature (°C)")
    ax_pk.set_ylabel("t_peak (hours)")
    ax_pk.set_title("Peak time vs anneal T")
    ax_pk.grid(True, alpha=0.3)
    ax_pk.legend(frameon=True, framealpha=0.85, loc="best")

    # Panel 2: t_return
    ax_rt.plot(temps_arr, tret_E, "o--", lw=1.6, label="exp", ms=5)
    ax_rt.plot(temps_arr, tret_S, "s-",  lw=1.6, label="sim", ms=5)
    ax_rt.set_xlabel("Anneal temperature (°C)")
    ax_rt.set_ylabel(f"t_return to {int(return_tol*100)}% band (hours)")
    ax_rt.set_title("Return-to-baseline vs anneal T")
    ax_rt.grid(True, alpha=0.3)
    ax_rt.legend(frameon=True, framealpha=0.85, loc="best")

    fig.suptitle(f"Anneal sweep @ firing {firing_temp_C}°C — t_peak & t_return ({layer}, {kind})", y=1.02, fontsize=12)
    fig.tight_layout()

    if print_data and sim_nums:
        print("\n=== SIM t_peak & t_return (layer=%s, %s) ===" % (layer, kind))
        print("T(°C)   t_peak(hr)   t_return(hr) @ %.0f%%   min_norm   t_at_min(hr)" % (return_tol*100))
        for T in temps:
            n = sim_nums.get(T, {}).get("norm")
            if not n: continue
            print(f"{T:>4}    {n['t_peak_hr']:>8.2f}      {n['t_return_hr']:>9.2f}      {n['y_min_norm']:>8.3f}    {n['t_min_hr']:>9.2f}")

    return (ax_pk, ax_rt, sim_nums) if return_data else (ax_pk, ax_rt)




#fitting functions


def _resample_on_grid(t, y, grid_logt):
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y) & (t > 0)
    if m.sum() < 2:
        return np.full_like(grid_logt, np.nan, dtype=float)
    logt = np.log10(t[m]); y = y[m]
    order = np.argsort(logt)
    return np.interp(grid_logt, logt[order], y[order], left=np.nan, right=np.nan)

def overall_shape_score(sim_nums):
    vals = []
    for T, d in sim_nums.items():
        s = d.get("shape_score")
        if s and np.isfinite(s.get("score", np.nan)):
            vals.append(float(s["score"]))
    return np.mean(vals) if vals else float('nan')

 

def shape_score(
    t_exp_h, y_exp, t_sim_h, y_sim,
    *,
    half_window_dec=6.0,   # scan around peak alignment ± this many decades
    n_shifts=401,          # resolution of the scan
    n_grid=600,            # resample grid size on EXP log-time range
    plateau_thresh=0.03,   # plateau ends when EXP first exceeds this
    return_tol=0.05,       # "returned" when EXP ≤ this after peak
    min_decades=0.3,       # <- slightly looser (was 0.5)
    min_pts=10,            # <- slightly looser (was 12)
    rise_weight=0.5, plateau_weight=0.2, return_weight=0.5,
    _resample=_resample_on_grid,
):
    """
    Simple scorer (peak-centered shift scan; weighted MAE over plateau/rise/return).
    Uses FINITE-point coverage checks to avoid NaN/inf scores.
    """
    t_exp_h = np.asarray(t_exp_h, float); y_exp = np.asarray(y_exp, float)
    t_sim_h = np.asarray(t_sim_h, float); y_sim = np.asarray(y_sim, float)

    logt_exp = np.log10(t_exp_h[t_exp_h > 0])
    logt_sim = np.log10(t_sim_h[t_sim_h > 0])
    if logt_exp.size < 2 or logt_sim.size < 2:
        raise ValueError("Not enough positive time samples.")

    base_min, base_max = float(np.min(logt_exp)), float(np.max(logt_exp))
    base_grid = np.linspace(base_min, base_max, int(n_grid))
    yE_grid = _resample(t_exp_h, y_exp, base_grid)
    if np.all(~np.isfinite(yE_grid)):
        raise ValueError("Experimental series has no finite samples on the grid.")

    # EXP landmarks
    i_pk = int(np.nanargmax(yE_grid))
    i_plat_end = None
    for j in range(0, i_pk + 1):
        v = yE_grid[j]
        if np.isfinite(v) and (v > plateau_thresh):
            i_plat_end = max(0, j - 1); break
    if i_plat_end is None:
        i_plat_end = max(0, i_pk // 6)
    i_ret = None
    for j in range(i_pk + 1, yE_grid.size):
        v = yE_grid[j]
        if np.isfinite(v) and (v <= return_tol):
            i_ret = j; break
    if i_ret is None:
        i_ret = yE_grid.size - 1

    logt_pk, logt_plat_end, logt_ret = base_grid[i_pk], base_grid[i_plat_end], base_grid[i_ret]

    # Peak-align center (raw series)
    try:
        tE_pk = float(t_exp_h[int(np.nanargmax(y_exp))])
        tS_pk = float(t_sim_h[int(np.nanargmax(y_sim))])
        s_center = (np.log10(tE_pk) - np.log10(tS_pk)) if (tE_pk > 0 and tS_pk > 0) else 0.0
    except Exception:
        s_center = 0.0

    s_values = np.linspace(s_center - half_window_dec, s_center + half_window_dec, int(n_shifts))
    best = None

    for s in s_values:
        # overlap band (in decades)
        logt_sim_s = logt_sim + s
        lo = max(float(np.min(logt_sim_s)), base_min)
        hi = min(float(np.max(logt_sim_s)), base_max)
        if (hi - lo) < min_decades:
            continue

        sel = (base_grid >= lo) & (base_grid <= hi)
        if int(sel.sum()) < 3 * min_pts:
            continue

        grid   = base_grid[sel]
        yE_sub = yE_grid[sel]
        yS_sub = _resample(t_sim_h * (10.0 ** s), y_sim, grid)

        # finite-only mask
        m  = np.isfinite(yE_sub) & np.isfinite(yS_sub)
        if int(m.sum()) < 3 * min_pts:
            continue

        g  = grid[m]
        ye = yE_sub[m]
        ys = yS_sub[m]

        # region masks anchored to EXP landmarks, AND finite
        mask_plateau = (g <= logt_plat_end)
        mask_rise    = (g >  logt_plat_end) & (g <= logt_pk)
        mask_return  = (g >  logt_pk)       & (g <= logt_ret)

        # require coverage in plateau & return USING FINITE POINTS
        n_pl = int(np.sum(mask_plateau))
        n_rt = int(np.sum(mask_return))
        if n_pl < min_pts or n_rt < min_pts:
            continue

        # region MAEs
        def _mae(mask):
            if not np.any(mask):
                return np.inf
            dy = ye[mask] - ys[mask]
            return float(np.mean(np.abs(dy))) if dy.size else np.inf

        mae_plateau = _mae(mask_plateau)
        mae_rise    = _mae(mask_rise)
        mae_return  = _mae(mask_return)

        # if any region has no finite overlap (shouldn't, but guard), skip
        if not np.isfinite(mae_plateau) or not np.isfinite(mae_return):
            continue

        score = (plateau_weight * mae_plateau
                 + rise_weight    * mae_rise
                 + return_weight  * mae_return)

        cand = {
            "score": float(score),
            "shift_decades": float(s),
            "overlap_decades": float(hi - lo),
            "n_points": int(m.sum()),
            "n_plateau": n_pl,
            "n_rise": int(np.sum(mask_rise)),
            "n_return": n_rt,
            "mae_plateau": float(mae_plateau),
            "mae_rise": float(mae_rise),
            "mae_return": float(mae_return),
            "logt_peak_exp": float(logt_pk),
            "logt_return_exp": float(logt_ret),
        }
        if (best is None) or (score < best["score"]):
            best = cand

    if best is None:
        # last-resort relaxation: try once with looser guards
        if (min_decades > 0.2) or (min_pts > 6):
            return shape_score(
                t_exp_h, y_exp, t_sim_h, y_sim,
                half_window_dec=half_window_dec,
                n_shifts=n_shifts,
                n_grid=n_grid,
                plateau_thresh=plateau_thresh,
                return_tol=return_tol,
                min_decades=max(0.2, 0.5*min_decades),
                min_pts=max(6, min_pts // 2),
                rise_weight=rise_weight, plateau_weight=plateau_weight, return_weight=return_weight,
                _resample=_resample,
            )
        raise RuntimeError("No valid overlap to compare (even after relaxing).")
    return best

