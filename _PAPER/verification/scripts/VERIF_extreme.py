"""
tc_extreme_verif.py
===================
Self-contained extreme-event verification for CorrDiff TC downscaling.
Depends only on numpy / xarray / zarr / scipy / netCDF4.

DEFAULT CONFIG IS HOURLY:
    WRF_precip   hourly precipitation RATE  (mm/hr)
    WRF_SPD10    hourly mean 10-m wind speed (m/s)
The aggregation is a config switch (`agg`), so the same script also does
'sum' (daily accumulation) and 'max' (daily maximum) without modification.

Sources, all scored on ONE common valid mask so they see identical grid points:
    {ens}_members   each member as its OWN field; aggregated (mean, min-max)
                    across the M members. The ensemble mean is NEVER thresholded.
    {ens}_PMM       probability-matched ensemble mean (Ebert 2001).
    UNet            the deterministic baseline.

Metrics (from additive per-storm ingredients, pooled EXACTLY before scoring):

    SEDI   Symmetric Extremal Dependence Index (Ferro & Stephenson 2011).
           *** The reason it is here: TS/CSI and ETS are BASE-RATE DEPENDENT --
           they collapse toward 0 as the event gets rarer no matter how good the
           forecast is. That is the degeneracy that makes ETS = 0.06 at 30 m/s
           uninterpretable. SEDI is asymptotically base-rate INDEPENDENT and does
           not degenerate, so it is the score that still means something at your
           top thresholds. ***
           TWO CAVEATS, both mandatory to report:
             1. SEDI is NOT bias-corrected. Like POD it can be HEDGED by
                over-forecasting. ALWAYS show FBIAS beside it.
             2. SEDI is undefined with zero hits -> NaN here, never a fabricated
                floor. POD and FBIAS carry that story instead.
    ETS    Gilbert skill score (chance-corrected; base-rate dependent).
    FBIAS  (hits+fa)/(hits+miss). <1 under-produces events, >1 over-produces.
    FSS    Fractions Skill Score per neighborhood width.
    TS, POD, FAR, POFD   reported because they are what EXPLAIN the above.

AUC is deliberately absent: a value-binned AUC measures bin occupancy rather
than discrimination for smooth fields, and pooled over the full domain it is
dominated by trivially-easy non-events. Rank-AUC is used only in the diagnostics.

------------------------------------------------------------------------------
CHOOSING THRESHOLDS -- run recon_thresholds(CFG) FIRST
------------------------------------------------------------------------------
Pick thresholds by BASE RATE, not by round numbers. recon_thresholds() prints the
pooled target's quantiles and the base rate + event count at each candidate, so
you can see where the sample runs out before you commit. Rules of thumb:
    base rate > 1e-1   not an extreme; ETS is dominated by the chance correction
    base rate < 1e-5   too few events for a stable pooled score with 25 storms
    the top threshold cannot exceed what CONUS404 itself produces at 8 km on an
    hourly mean -- check the target's q99.9 in the recon output.

PRE-FLIGHT DIAGNOSTICS (CFG['diagnose']=True; runs on the first storm)
    D1  member-file coordinate consistency. xr.concat's default join='outer'
        silently ALIGNS mismatched coords -- shifting one member relative to
        another. Members then still verify fine individually while the ensemble
        MEAN (and hence the PMM) collapses. This script uses join='override'.
    D2  value distributions + negatives. Catches a wrong power/unit on any one
        source. Negatives matter because (-x)**4 is POSITIVE; they are clipped.
    D3  lag-1 temporal autocorrelation, truth vs members. For agg='none' this is
        informational (a temporally white ensemble is physically implausible);
        for agg='max' it is CRITICAL (daily maxima are inflated by noise
        accumulation) and for agg='sum' it biases the other way.
    D4  PMM sanity laws (rank-based, so binning cannot confound them).
    D5  inter-member vs member-truth correlation.
    D6  sample accounting.

Outputs
-------
  <outdir>/tc_extreme_ingredients.nc   per-storm additive ingredients
  <outdir>/tc_extreme_scores.csv       pooled long-format scorecard
  <outdir>/tc_extreme_diagnostics.txt  the pre-flight report

Pair with tc_extreme_plots.py.
"""

import os
import re
import csv
import glob
import warnings
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter
from scipy.stats import rankdata

EPS = 1e-12


# ======================================================================
# 1. Scoring primitives
# ======================================================================
def contingency(fcst_bin, obs_bin):
    fb = np.asarray(fcst_bin, bool)
    ob = np.asarray(obs_bin, bool)
    return np.array([np.count_nonzero(fb & ob), np.count_nonzero(~fb & ob),
                     np.count_nonzero(fb & ~ob), np.count_nonzero(~fb & ~ob)],
                    dtype=np.float64)


def sedi_from_table(t):
    """
    Symmetric Extremal Dependence Index (Ferro & Stephenson, 2011, Wea. Forecasting).

        SEDI = [lnF - lnH - ln(1-F) + ln(1-H)] / [lnF + lnH + ln(1-F) + ln(1-H)]
        H = POD  = hits/(hits+miss)      F = POFD = fa/(fa+cn)

    Range [-1, 1]; 1 perfect, 0 no skill (H == F gives exactly 0). Unlike
    TS/ETS it does NOT tend to 0 as the base rate tends to 0, which is why it is
    the score to trust at the rarest thresholds.

    H and F are bounded away from 0 and 1 by HALF A COUNT (standard continuity
    correction). Without it a single false alarm swings SEDI wildly. With zero
    hits SEDI is genuinely undefined and NaN is returned -- reporting a
    fabricated floor value there would flatter a model that simply cannot
    produce the event (POD and FBIAS already tell that story).
    """
    hits, miss, fa, cn = (float(x) for x in t)
    n1, n0 = hits + miss, fa + cn
    if n1 <= 0 or n0 <= 0 or hits <= 0:
        return np.nan
    H = min(max(hits / n1, 0.5 / n1), 1.0 - 0.5 / n1)
    F = min(max(fa / n0, 0.5 / n0), 1.0 - 0.5 / n0)
    num = np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)
    den = np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H)
    return float(num / den) if abs(den) > EPS else np.nan


def scores_from_table(t):
    """(hits, miss, fa, cn) -> SEDI, ETS, FBIAS, TS, POD, FAR, POFD, base_rate."""
    hits, miss, fa, cn = (float(x) for x in t)
    n = hits + miss + fa + cn
    hits_r = (hits + miss) * (hits + fa) / (n + EPS)
    den = hits + miss + fa - hits_r
    return dict(SEDI=sedi_from_table(t),
                ETS=(hits - hits_r) / den if den > EPS else np.nan,
                FBIAS=(hits + fa) / (hits + miss + EPS),
                TS=hits / (hits + miss + fa + EPS),
                POD=hits / (hits + miss + EPS),
                FAR=fa / (hits + fa + EPS),
                POFD=fa / (fa + cn + EPS),
                base_rate=(hits + miss) / (n + EPS), n=n)


def fss_components(obs_bin, fcst_bin, validf, scales):
    """Additive FSS components for one (T, Y, X) binary pair -> (nscales, 2)."""
    out = np.zeros((len(scales), 2))
    ob = (obs_bin * validf).astype(np.float32)
    fb = (fcst_bin * validf).astype(np.float32)
    for si, s in enumerate(scales):
        size = (1, int(s), int(s))
        Vf = uniform_filter(validf, size=size, mode="constant")
        good = Vf > 1e-6
        Po = np.where(good, uniform_filter(ob, size=size, mode="constant")
                      / np.maximum(Vf, 1e-6), 0.0)
        Pf = np.where(good, uniform_filter(fb, size=size, mode="constant")
                      / np.maximum(Vf, 1e-6), 0.0)
        out[si, 0] = float((((Pf - Po) ** 2) * validf).sum())
        out[si, 1] = float((((Pf ** 2) + (Po ** 2)) * validf).sum())
    return out


def fss_from_components(comp):
    comp = np.asarray(comp, float)
    num, den = comp[..., 0], comp[..., 1]
    return np.where(den > EPS, 1.0 - num / np.maximum(den, EPS), np.nan)


def rank_auc(values, obs_bin):
    """Exact rank AUC (no binning). Diagnostics only."""
    v = np.asarray(values, float).ravel()
    ob = np.asarray(obs_bin, bool).ravel()
    n1, n0 = ob.sum(), (~ob).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(v)
    return float((r[ob].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ======================================================================
# 2. Probability-matched ensemble mean (Ebert 2001)
# ======================================================================
def pmm_field(members2d, valid2d):
    M = members2d.shape[0]
    mean2d = members2d.mean(axis=0)
    v = valid2d & np.isfinite(mean2d)
    out = np.full(mean2d.shape, np.nan, dtype=np.float32)
    n = int(v.sum())
    if n == 0:
        return out
    rep = np.sort(members2d[:, v].ravel())[::-1][::M][:n]
    vals = np.empty(n, dtype=np.float32)
    vals[np.argsort(-mean2d[v], kind="stable")] = rep
    out[v] = vals
    return out


def pmm_stack(members4d, valid3d):
    """members4d: (M, T, Y, X); valid3d: (T, Y, X) -> (T, Y, X). One PMM per
    time step (or per aggregation period)."""
    out = np.full(members4d.shape[1:], np.nan, dtype=np.float32)
    for t in range(members4d.shape[1]):
        out[t] = pmm_field(members4d[:, t], valid3d[t])
    return out


# ======================================================================
# 3. Loading + time aggregation
# ======================================================================
def _natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def get_field(ds, var, role, var_config):
    cfg = var_config[var]
    name = cfg[f"{role}_var"]
    order = ("member", "time", ...) if role == "member" else ("time", ...)
    if name not in ds and {"WRF_U10", "WRF_V10"} <= set(ds.data_vars):
        da = np.sqrt(ds["WRF_U10"] ** 2 + ds["WRF_V10"] ** 2)
    else:
        da = ds[name]
    arr = np.asarray(da.transpose(*order).values, dtype=np.float32)
    # The ONLY reduced-space field is the target's fourth-root precipitation,
    # stored as WRF_precip_025; back-transform it to physical mm/hr with **4.
    # Every other field (member/UNet WRF_precip, all winds) is read AS-IS. This
    # is hard-coded to the specific variable name -- there is deliberately no
    # 'power' knob to set per role, because a shared power was what previously
    # double-transformed the already-physical member/UNet precip (15 -> 15**4).
    if name == "WRF_precip_025":
        arr = np.clip(arr, 0.0, None) ** 4      # clip: (-x)**4 would be POSITIVE
    return arr


def open_ensemble(pattern):
    """join='override': the default join='outer' silently aligns mismatched
    coordinates and can SHIFT one member relative to another, destroying the
    ensemble mean (and therefore the PMM). D1 checks whether it was needed."""
    files = sorted(glob.glob(pattern), key=_natkey)
    if not files:
        raise FileNotFoundError(f"no member files match {pattern}")
    ds = xr.concat([xr.open_zarr(f) for f in files], dim="member",
                   join="override", coords="minimal", compat="override")
    return ds.assign_coords(member=np.arange(ds.sizes["member"])), files


def load_window(ds, times, varnames):
    t = ds["time"].values
    pos = np.where(np.isin(t, times))[0]
    if pos.size == 0:
        return None
    return ds[[v for v in varnames if v in ds]].isel(time=pos).sortby("time")


def aggregate_time(arr, times, mode, min_hours=24):
    """
    mode='none'  -> HOURLY: every timestep is a sample, nothing is aggregated.
    mode='sum'   -> daily accumulation.  mode='max' -> daily maximum.
    For the daily modes, days with fewer than `min_hours` hours are DROPPED (a
    partial sum is not a daily accumulation).
    Returns (aggregated (..., N, Y, X), labels (N,), hours_per_sample (N,)).
    """
    if mode in (None, "none", "hourly"):
        t = np.asarray(times)
        return arr, t, np.ones(t.size, int)
    days = np.asarray(times).astype("datetime64[D]")
    uniq, inv = np.unique(days, return_inverse=True)
    counts = np.bincount(inv, minlength=uniq.size)
    keep = np.where(counts >= min_hours)[0]
    if keep.size == 0:
        return None, uniq[[]], counts[[]]
    out = [arr[..., inv == d, :, :].sum(axis=-3) if mode == "sum"
           else arr[..., inv == d, :, :].max(axis=-3) for d in keep]
    return np.stack(out, axis=-3), uniq[keep], counts[keep]


def cases_to_windows(catalog):
    by = {}
    for season, hid, name, cat, ta, tb in catalog:
        by.setdefault(season, []).append(
            (name, int(cat), np.datetime64(ta.replace(" ", "T")),
             np.datetime64(tb.replace(" ", "T"))))
    return by


def common_times(dsT, dsU, ens_ds, t0, t1):
    ct = dsT["time"].values
    for e in ens_ds:
        ct = np.intersect1d(ct, ens_ds[e]["time"].values)
    ct = np.intersect1d(ct, dsU["time"].values)
    return np.sort(ct[(ct >= t0) & (ct <= t1)])


# ======================================================================
# 4. Threshold reconnaissance -- RUN THIS BEFORE CHOOSING THRESHOLDS
# ======================================================================
def recon_thresholds(cfg, candidates):
    """
    Scan the pooled TC-window TARGET and print, per candidate threshold, the
    base rate and the pooled event count. Choose thresholds from this, not from
    round numbers: a threshold whose event count is a few hundred cannot support
    a stable pooled score, and a threshold above the target's own q99.9 is
    verifying something CONUS404 does not produce at 8 km.
    """
    vc = cfg["var_config"]
    LOAD = ["WRF_precip", "WRF_precip_025", "WRF_SPD10", "WRF_U10", "WRF_V10"]
    static = xr.open_zarr(f"{cfg['base']}/static/C404_TC_static_8km.zarr")
    lm = static["LANDMASK"].values
    print("\n" + "=" * 74)
    print(" THRESHOLD RECONNAISSANCE (pooled target over all TC windows)")
    print("=" * 74)
    for v in vc:
        smask = (lm <= 0) if vc[v].get("mask") == "ocean" \
            else np.ones(lm.shape, bool)
        cand = np.asarray(candidates[v], float)
        n_ev = np.zeros(cand.size, np.int64)
        n_tot = 0
        vals = []
        for year in cfg["years"]:
            dsT = xr.open_zarr(
                f"{cfg['base']}/C404_CorrDiff/TC_target_{year}.zarr")
            for name, cat, t0, t1 in cfg["windows"].get(year, []):
                t = dsT["time"].values
                ct = np.sort(t[(t >= t0) & (t <= t1)])
                if ct.size == 0:
                    continue
                w = load_window(dsT, ct, LOAD)
                a = get_field(w, v, "target", vc)
                a, _, _ = aggregate_time(a, w["time"].values, vc[v]["agg"],
                                         cfg.get("min_hours", 24))
                if a is None:
                    continue
                m = np.isfinite(a) & smask[None, :, :]
                x = a[m]
                n_tot += x.size
                n_ev += np.array([(x >= c).sum() for c in cand])
                if len(vals) < 40:
                    vals.append(x[::37])          # thin sample for quantiles
        x = np.concatenate(vals)
        q = np.nanpercentile(x, [50, 90, 99, 99.9, 99.99])
        print(f"\n {v}  (agg='{vc[v]['agg']}', {vc[v].get('unit','')})"
              f"   {n_tot:,} valid samples")
        print(f"   target quantiles  q50 {q[0]:.2f}   q90 {q[1]:.2f}   "
              f"q99 {q[2]:.2f}   q99.9 {q[3]:.2f}   q99.99 {q[4]:.2f}   "
              f"max {np.nanmax(x):.2f}")
        print(f"   {'threshold':>10s}{'base rate':>12s}{'events':>12s}   verdict")
        for c, e in zip(cand, n_ev):
            br = e / max(n_tot, 1)
            if br > 0.1:
                vd = "not extreme (chance correction dominates ETS)"
            elif e < 500:
                vd = "TOO FEW EVENTS -- score will be noise"
            elif br < 1e-5:
                vd = "marginal; check bootstrap width"
            else:
                vd = "usable"
            print(f"   {c:>10.4g}{br:>12.2e}{e:>12,d}   {vd}")
    print()


# ======================================================================
# 5. Diagnostics
# ======================================================================
class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        print(s)
        self.lines.append(s)

    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")
        print(f"\ndiagnostics -> {path}")


def d1_coords(files, rep):
    rep("D1  member-file coordinate consistency")
    ref = xr.open_zarr(files[0])
    bad = []
    for f in files[1:]:
        d = xr.open_zarr(f)
        for c in ref.coords:
            if c not in d.coords:
                bad.append(f"{os.path.basename(f)}: missing coord '{c}'")
            elif ref[c].shape != d[c].shape or \
                    not np.array_equal(ref[c].values, d[c].values):
                bad.append(f"{os.path.basename(f)}: coord '{c}' DIFFERS")
    if bad:
        rep("    *** MISALIGNED MEMBER FILES ***")
        for b in bad[:8]:
            rep(f"      {b}")
        rep("    join='override' forced member 0's coords on all members.")
        rep("    If the grids genuinely differ this is WRONG -- regrid first.")
    else:
        rep("    OK: all member files share identical coordinates.")
    rep()
    return not bad


def d2_distributions(fields, rep, unit, negfrac=None):
    rep(f"D2  value distributions ({unit})")
    rep(f"    {'source':<16s}{'median':>10s}{'q99':>10s}{'q99.9':>10s}{'max':>10s}")
    ref = None
    for name, v in fields.items():
        q = np.nanpercentile(v, [50, 99, 99.9])
        if ref is None:
            ref = q[2]
        flag = ""
        if name != "TARGET" and np.isfinite(ref) and ref > 0:
            r = q[2] / ref
            if r < 0.35:
                flag = f"  <-- q99.9 {1/max(r,1e-9):.0f}x TOO SMALL"
            elif r > 3.0:
                flag = f"  <-- q99.9 {r:.0f}x TOO LARGE"
        rep(f"    {name:<16s}{q[0]:>10.2f}{q[1]:>10.2f}{q[2]:>10.2f}"
            f"{np.nanmax(v):>10.2f}{flag}")
    if negfrac is not None:
        rep(f"    negatives in the member field: {negfrac:.2%}"
            + ("   <-- note: physical rate/speed should be >= 0"
               if negfrac > 0.001 else ""))
    rep()


def d3_autocorr(obs_h, mem_h, rep, var, agg):
    def lag1(w):
        a = w[:-1] - w[:-1].mean(0)
        b = w[1:] - w[1:].mean(0)
        den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        return np.where(den > EPS, (a * b).sum(0) / np.maximum(den, EPS), np.nan)
    r_t = float(np.nanmedian(lag1(obs_h)))
    r_m = float(np.nanmedian([np.nanmedian(lag1(mem_h[m]))
                              for m in range(mem_h.shape[0])]))
    rep(f"D3  lag-1 TEMPORAL autocorrelation ({var}, hourly; agg='{agg}')")
    rep(f"    CONUS404 truth  : {r_t:.3f}")
    rep(f"    members (median): {r_m:.3f}")
    low = r_m < r_t - 0.15
    if low and agg == "max":
        rep("    *** CRITICAL for agg='max': a temporally whiter field has an")
        rep("    INFLATED maximum (noise accumulates over the draws). Any FBIAS")
        rep("    gain is partly MANUFACTURED.")
    elif low and agg == "sum":
        rep("    *** For agg='sum': a temporally whiter field has an UNDER-")
        rep("    dispersed accumulation (noise averages out), so part of any")
        rep("    FBIAS deficit is a temporal artifact, not an amplitude deficit.")
    elif low:
        rep("    NOTE: with agg='none' this does not bias the scores (no")
        rep("    aggregation), but a temporally white ensemble is physically")
        rep("    implausible and is worth reporting as a model-fidelity result.")
    else:
        rep("    OK: members carry truth-like temporal coherence.")
    rep()
    return r_t, r_m


def d4_d5_pmm_laws(obs, mem, valid, thr, rep, ens):
    M = mem.shape[0]
    mean_f = mem.mean(0)
    pmm_f = pmm_stack(mem, valid)
    ob = (obs >= thr)[valid]
    a_mem = np.array([rank_auc(mem[m][valid], ob) for m in range(M)])
    a_mean = rank_auc(mean_f[valid], ob)
    a_pmm = rank_auc(pmm_f[valid], ob)
    rep(f"D4  PMM sanity laws ({ens}, thr {thr:g}, base rate {ob.mean():.3e})")
    rep(f"    rankAUC member (mean of {M}): {np.nanmean(a_mem):.4f}"
        f"   [{np.nanmin(a_mem):.3f}, {np.nanmax(a_mem):.3f}]")
    rep(f"    rankAUC ensemble mean       : {a_mean:.4f}")
    rep(f"    rankAUC PMM                 : {a_pmm:.4f}")
    gap = a_mean - a_pmm
    l1 = (a_pmm <= a_mean + 1e-2) and (gap < 0.10)
    rep(f"    LAW 1  rankAUC(PMM) <= rankAUC(ens mean), gap {gap:+.4f} (ties)"
        f" : {'OK' if l1 else 'VIOLATED -> PMM / valid-mask bug'}")
    l2 = a_mean >= np.nanmean(a_mem) - 1e-9
    rep(f"    LAW 2  rankAUC(ens mean) >= rankAUC(member)"
        f"          : {'OK' if l2 else 'VIOLATED -> members NOT mutually aligned (see D1)'}")
    ets_m = np.nanmean([scores_from_table(
        contingency(mem[m][valid] >= thr, ob))["ETS"] for m in range(M)])
    sp = scores_from_table(contingency(pmm_f[valid] >= thr, ob))
    l3 = sp["ETS"] >= ets_m - 0.02
    rep(f"    LAW 3  ETS(PMM) {sp['ETS']:.3f} >= ETS(member) {ets_m:.3f}"
        f"  (FBIAS {sp['FBIAS']:.2f}) : "
        f"{'OK' if l3 else 'VIOLATED -> ensemble mean is degraded'}")

    flat = mem.reshape(M, -1)[:, valid.ravel()]
    C = np.corrcoef(flat)
    iu = np.triu_indices(M, 1)
    ct = np.array([np.corrcoef(flat[m], obs[valid])[0, 1] for m in range(M)])
    rep(f"D5  ensemble coherence ({ens})")
    rep(f"    inter-member corr : {C[iu].mean():.3f}"
        f"   [{C[iu].min():.3f}, {C[iu].max():.3f}]")
    rep(f"    member vs truth   : {ct.mean():.3f}"
        f"   [{ct.min():.3f}, {ct.max():.3f}]")
    rep(f"    ens mean vs truth : "
        f"{np.corrcoef(mean_f[valid], obs[valid])[0,1]:.3f}"
        f"   (MUST exceed 'member vs truth')")
    if C[iu].mean() < 0.3 < ct.mean():
        rep("    *** members correlate with truth but NOT with each other:")
        rep("        impossible unless the averaged fields are misaligned.")
    rep()
    return l1 and l2 and l3


# ======================================================================
# 6. Per-storm ingredients
# ======================================================================
def verify_storm(obs, unet, mem, valid, thresholds, scales):
    validf = valid.astype(np.float32)
    idx = valid.ravel()
    obs_f = obs.ravel()[idx]
    un_f = unet.ravel()[idx]
    nt, ns = len(thresholds), len(scales)
    out = dict(unet=dict(tables=np.zeros((nt, 4)), fss=np.zeros((nt, ns, 2))),
               ens={})
    pmm = {e: pmm_stack(a, valid) for e, a in mem.items()}

    for ti, thr in enumerate(thresholds):
        ob = obs_f >= thr
        ob3 = (obs >= thr).astype(np.float32)
        out["unet"]["tables"][ti] = contingency(un_f >= thr, ob)
        out["unet"]["fss"][ti] = fss_components(
            ob3, (unet >= thr).astype(np.float32), validf, scales)

    for e, arr in mem.items():
        M = arr.shape[0]
        d = dict(member_tables=np.zeros((nt, M, 4)),
                 member_fss=np.zeros((nt, M, ns, 2)),
                 pmm_tables=np.zeros((nt, 4)), pmm_fss=np.zeros((nt, ns, 2)))
        mem_f = arr.reshape(M, -1)[:, idx]
        pmm_f = pmm[e].ravel()[idx]
        for ti, thr in enumerate(thresholds):
            ob = obs_f >= thr
            ob3 = (obs >= thr).astype(np.float32)
            for m in range(M):
                d["member_tables"][ti, m] = contingency(mem_f[m] >= thr, ob)
                d["member_fss"][ti, m] = fss_components(
                    ob3, (arr[m] >= thr).astype(np.float32), validf, scales)
            d["pmm_tables"][ti] = contingency(pmm_f >= thr, ob)
            pb3 = np.where(np.isfinite(pmm[e]), pmm[e] >= thr,
                           False).astype(np.float32)
            d["pmm_fss"][ti] = fss_components(ob3, pb3, validf, scales)
        out["ens"][e] = d
    return out


# ======================================================================
# 7. Driver
# ======================================================================
def run(cfg):
    os.makedirs(cfg["outdir"], exist_ok=True)
    rep = Report()
    vc, ens_names = cfg["var_config"], list(cfg["ensembles"])
    scales, thr = tuple(cfg["fss_scales"]), cfg["thresholds"]
    varnames = list(vc)
    LOAD = ["WRF_precip", "WRF_precip_025", "WRF_SPD10", "WRF_U10", "WRF_V10"]

    rep("=" * 74)
    rep(" TC EXTREME VERIFICATION")
    rep("=" * 74)
    for v in varnames:
        rep(f"  {v:<12s} agg='{vc[v]['agg']}'  {vc[v].get('unit','')}  "
            f"thresholds {thr[v]}  mask={vc[v].get('mask','none')}")
    rep(f"  FSS scales {scales} grid points\n")

    static = xr.open_zarr(f"{cfg['base']}/static/C404_TC_static_8km.zarr")
    lm = static["LANDMASK"].values
    smask = {v: (lm <= 0) if vc[v].get("mask") == "ocean"
             else np.ones(lm.shape, bool) for v in varnames}

    store = {v: dict(thr=np.asarray(thr[v], float),
                     scales=np.asarray(scales, int),
                     storms=[], categories=[], nsamp=[], per_storm=[])
             for v in varnames}
    done = not cfg.get("diagnose", True)

    for year in cfg["years"]:
        rep(f"===== {year} =====")
        dsT = xr.open_zarr(f"{cfg['base']}/C404_CorrDiff/TC_target_{year}.zarr")
        dsU = xr.open_zarr(cfg["unet_path_fmt"].format(base=cfg["base"],
                                                       year=year))
        ens_ds, ens_files = {}, {}
        for e in ens_names:
            ens_ds[e], ens_files[e] = open_ensemble(
                f"{cfg['base']}/TC_pred_corrdiff_final/"
                f"TC_{e}_corrdiff_pred_{year}_mem*.zarr")
        if not done:
            d1_coords(ens_files[ens_names[0]], rep)

        for name, cat, t0, t1 in cfg["windows"][year]:
            ct = common_times(dsT, dsU, ens_ds, t0, t1)
            if ct.size == 0:
                warnings.warn(f"{year} {name}: no common timesteps")
                continue
            wT, wU = load_window(dsT, ct, LOAD), load_window(dsU, ct, LOAD)
            wE = {e: load_window(ens_ds[e], ct, LOAD) for e in ens_names}
            tt = wT["time"].values

            for v in varnames:
                mode, mh = vc[v]["agg"], cfg.get("min_hours", 24)
                obs_h = get_field(wT, v, "target", vc)
                obs, labels, hrs = aggregate_time(obs_h, tt, mode, mh)
                if obs is None:
                    warnings.warn(f"{year} {name} {v}: nothing to verify")
                    continue
                unet = aggregate_time(get_field(wU, v, "unet", vc),
                                      tt, mode, mh)[0]
                mem_h = {e: get_field(wE[e], v, "member", vc) for e in ens_names}
                mem = {e: aggregate_time(mem_h[e], tt, mode, mh)[0]
                       for e in ens_names}

                valid = np.isfinite(obs) & np.isfinite(unet) \
                    & smask[v][None, :, :]
                for e in ens_names:
                    valid &= np.isfinite(mem[e]).all(0)

                if not done:
                    rep(f"--- pre-flight: {year} {name}, {v} "
                        f"({obs.shape[0]} samples from {ct.size} h) ---")
                    # member field is read as-is (never transformed), so the
                    # stored field IS mem[e]; report the fraction of negatives.
                    raw = mem[ens_names[0]]
                    d2_distributions(
                        {"TARGET": obs[valid], "UNet": unet[valid],
                         **{f"{e}_member0": mem[e][0][valid] for e in ens_names},
                         **{f"{e}_PMM": pmm_stack(mem[e], valid)[valid]
                            for e in ens_names}},
                        rep, vc[v].get("unit", ""),
                        negfrac=float((raw < 0).mean()))
                    d3_autocorr(obs_h, mem_h[ens_names[0]], rep, v, mode)
                    d4_d5_pmm_laws(obs, mem[ens_names[0]], valid,
                                   float(np.median(thr[v])), rep, ens_names[0])
                    rep(f"D6  {obs.shape[0]} samples kept "
                        f"(agg='{mode}'); valid points "
                        f"{int(valid.sum()):,}\n")

                ing = verify_storm(obs, unet, mem, valid, thr[v], scales)
                store[v]["storms"].append(name)
                store[v]["categories"].append(cat)
                store[v]["nsamp"].append(int(obs.shape[0]))
                store[v]["per_storm"].append(ing)
                M = mem[ens_names[0]].shape[0]
                s0 = scores_from_table(
                    ing["ens"][ens_names[0]]["member_tables"][-1].sum(0) / M)
                rep(f"  {year} {name:<9s} cat{cat} {v:<11s} "
                    f"base(top thr) {s0['base_rate']:.2e}  "
                    f"SEDI {s0['SEDI']:.3f}  ETS {s0['ETS']:.3f}  "
                    f"FBIAS {s0['FBIAS']:.2f}")
                del obs_h, mem_h
            done = True

    ds = ingredients_to_dataset(store, ens_names, vc)
    nc = f"{cfg['outdir']}/tc_extreme_ingredients.nc"
    ds.to_netcdf(nc)
    rep(f"\npooled ingredients -> {nc}")
    rows = score_all(store, ens_names)
    print_scores(rows, rep)
    write_csv(rows, f"{cfg['outdir']}/tc_extreme_scores.csv")
    rep.save(f"{cfg['outdir']}/tc_extreme_diagnostics.txt")
    return store, rows


# ======================================================================
# 8. Pooling / output
# ======================================================================
METRICS = ["SEDI", "ETS", "FBIAS", "TS", "POD", "FAR"]


def score_all(store, ens_names, categories=None):
    rows = []

    def add(v, t, src, met, val, spread=np.nan):
        rows.append(dict(var=v, threshold=float(t), source=src, metric=met,
                         value=float(val),
                         spread=float(spread) if np.isfinite(spread) else np.nan))

    for v, d in store.items():
        keep = [i for i, c in enumerate(d["categories"])
                if categories is None or c in categories]
        if not keep:
            continue
        P = [d["per_storm"][i] for i in keep]
        for ti, thr in enumerate(d["thr"]):
            ut = sum(p["unet"]["tables"][ti] for p in P)
            s = scores_from_table(ut)
            for k in METRICS:
                add(v, thr, "UNet", k, s[k])
            add(v, thr, "UNet", "base_rate", s["base_rate"])
            uf = fss_from_components(sum(p["unet"]["fss"][ti] for p in P))
            for si, sc in enumerate(d["scales"]):
                add(v, thr, "UNet", f"FSS_s{sc}", uf[si])
            for e in ens_names:
                mt = sum(p["ens"][e]["member_tables"][ti] for p in P)
                mf = fss_from_components(
                    sum(p["ens"][e]["member_fss"][ti] for p in P))
                per = [scores_from_table(mt[m]) for m in range(mt.shape[0])]
                for k in METRICS:
                    a = np.array([q[k] for q in per], float)
                    add(v, thr, f"{e}_members", k, np.nanmean(a), np.nanstd(a))
                for si, sc in enumerate(d["scales"]):
                    add(v, thr, f"{e}_members", f"FSS_s{sc}",
                        np.nanmean(mf[:, si]), np.nanstd(mf[:, si]))
                pt = sum(p["ens"][e]["pmm_tables"][ti] for p in P)
                sp = scores_from_table(pt)
                for k in METRICS:
                    add(v, thr, f"{e}_PMM", k, sp[k])
                pf = fss_from_components(
                    sum(p["ens"][e]["pmm_fss"][ti] for p in P))
                for si, sc in enumerate(d["scales"]):
                    add(v, thr, f"{e}_PMM", f"FSS_s{sc}", pf[si])
    return rows


def print_scores(rows, rep=print):
    fss = sorted({r["metric"] for r in rows if r["metric"].startswith("FSS_s")},
                 key=lambda k: int(k[5:]))
    cols = METRICS + fss
    for var, thr in sorted({(r["var"], r["threshold"]) for r in rows}):
        br = [r["value"] for r in rows if r["var"] == var
              and r["threshold"] == thr and r["metric"] == "base_rate"]
        rep(f"\n== {var} >= {thr:g}   (base rate {br[0]:.3e})" if br
            else f"\n== {var} >= {thr:g}")
        rep("  " + f"{'source':<16s}" + "".join(f"{c:>9s}" for c in cols))
        for s in sorted({r["source"] for r in rows if r["var"] == var
                         and r["threshold"] == thr}):
            line = "  " + f"{s:<16s}"
            for c in cols:
                m = [r["value"] for r in rows if r["var"] == var
                     and r["threshold"] == thr and r["source"] == s
                     and r["metric"] == c]
                line += (f"{m[0]:>9.3f}" if m and np.isfinite(m[0])
                         else f"{'--':>9s}")
            rep(line)
    rep("\n  SEDI '--' means ZERO HITS: the score is undefined, not zero.")
    rep("  SEDI is not bias-corrected -- read it beside FBIAS, never alone.")


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["var", "threshold", "source",
                                          "metric", "value", "spread"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"scores -> {path}")


def ingredients_to_dataset(store, ens_names, var_config=None):
    d, coords = {}, {}
    for v, dd in store.items():
        coords[f"{v}_thr"] = dd["thr"]
        coords[f"{v}_scale"] = dd["scales"]
        coords[f"{v}_storm"] = np.array(dd["storms"], dtype=str)
        d[f"{v}_category"] = ((f"{v}_storm",), np.array(dd["categories"], int))
        d[f"{v}_nsamp"] = ((f"{v}_storm",), np.array(dd["nsamp"], int))
        d[f"{v}_unet_tables"] = (
            (f"{v}_storm", f"{v}_thr", "cont"),
            np.stack([p["unet"]["tables"] for p in dd["per_storm"]]))
        d[f"{v}_unet_fss"] = (
            (f"{v}_storm", f"{v}_thr", f"{v}_scale", "fsscomp"),
            np.stack([p["unet"]["fss"] for p in dd["per_storm"]]))
        for e in ens_names:
            for key, dims in (
                    ("member_tables",
                     (f"{v}_storm", f"{v}_thr", "member", "cont")),
                    ("member_fss",
                     (f"{v}_storm", f"{v}_thr", "member", f"{v}_scale",
                      "fsscomp")),
                    ("pmm_tables", (f"{v}_storm", f"{v}_thr", "cont")),
                    ("pmm_fss",
                     (f"{v}_storm", f"{v}_thr", f"{v}_scale", "fsscomp"))):
                d[f"{v}_{e}_{key}"] = (
                    dims, np.stack([p["ens"][e][key] for p in dd["per_storm"]]))
    ds = xr.Dataset(d, coords=coords)
    ds.attrs["ensembles"] = list(ens_names)
    ds.attrs["variables"] = list(store)
    if var_config:                 # so the plots LABEL AXES FROM THE CONFIG
        for v in store:
            ds.attrs[f"{v}_agg"] = var_config[v]["agg"]
            ds.attrs[f"{v}_unit"] = var_config[v].get("unit", "")
    return ds


# ======================================================================
if __name__ == "__main__":
    base = "/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC"

    CATALOG = [
        # season, hurdat_id, storm, category, track_start, track_end
        (2020, "AL082020", "Hanna",    1, "2020-07-23 00:00", "2020-07-26 00:00"),
        (2020, "AL092020", "Isaias",   1, "2020-07-31 06:00", "2020-08-04 18:00"),
        (2020, "AL142020", "Marco",    1, "2020-08-22 12:00", "2020-08-25 00:00"),
        (2020, "AL132020", "Laura",    4, "2020-08-24 02:00", "2020-08-29 00:00"),
        (2020, "AL192020", "Sally",    2, "2020-09-11 18:00", "2020-09-17 06:00"),
        (2020, "AL252020", "Gamma",    1, "2020-10-03 16:45", "2020-10-06 12:00"),
        (2020, "AL262020", "Delta",    3, "2020-10-07 06:00", "2020-10-10 12:00"),
        (2020, "AL282020", "Zeta",     3, "2020-10-27 03:55", "2020-10-29 12:00"),
        (2020, "AL292020", "Eta",      1, "2020-11-08 00:00", "2020-11-13 06:00"),
    
        (2021, "AL052021", "Elsa",     1, "2021-07-05 00:00", "2021-07-09 16:30"),
        (2021, "AL082021", "Henri",    1, "2021-08-18 18:00", "2021-08-23 12:00"),
        #(2021, "AL072021", "Grace",    3, "2021-08-19 09:45", "2021-08-21 12:00"),
        (2021, "AL092021", "Ida",      4, "2021-08-27 12:00", "2021-09-01 06:00"),
        (2021, "AL142021", "Nicholas", 1, "2021-09-12 12:00", "2021-09-15 12:00"),
    
        #(2022, "AL072022", "Fiona",    4, "2022-09-20 00:00", "2022-09-23 06:00"),
        (2022, "AL092022", "Ian",      5, "2022-09-27 00:00", "2022-09-30 18:05"),
        (2022, "AL172022", "Nicole",   1, "2022-11-07 06:00", "2022-11-11 12:00"),
    
        #(2023, "AL082023", "Franklin", 4, "2023-08-24 00:00", "2023-08-30 12:00"),
        (2023, "AL102023", "Idalia",   4, "2023-08-26 12:00", "2023-08-31 06:00"),
        #(2023, "AL132023", "Lee",      2, "2023-09-13 12:00", "2023-09-15 12:00"),
    
        (2024, "AL022024", "Beryl",    1, "2024-07-05 11:00", "2024-07-09 06:00"),
        (2024, "AL042024", "Debby",    1, "2024-08-03 00:00", "2024-08-08 18:00"),
        #(2024, "AL052024", "Ernesto",  2, "2024-08-14 18:00", "2024-08-16 06:00"),
        (2024, "AL062024", "Francine",  2, "2024-09-09 12:00", "2024-09-12 12:00"),
        (2024, "AL092024", "Helene",   4, "2024-09-25 06:00", "2024-09-27 12:00"),
    ]

    CFG = dict(
        base=base,
        outdir=f"{base}/verif_hourly",
        years=[2020, 2021, 2022, 2023, 2024],
        ensembles=("ERA5", "GDAS"),
        windows=cases_to_windows(CATALOG),
        unet_path_fmt="{base}/TC_UNET/TC_UNET_pred_{year}_MSLP.zarr",
        diagnose=True,
        fss_scales=(3, 9, 27),          # grid points (x8 km)
        thresholds={
            # HOURLY PRECIPITATION RATE, mm/hr. NOT mm/day, and NOT mm/day/24:
            # 40/24 = 1.67 mm/hr is light rain (base rate ~2e-2), not an extreme.
            # These span light -> convective -> flash-flood scale. Run
            # recon_thresholds() FIRST and drop any whose event count is small.
            #   1   light rain
            #   2   light/moderate boundary
            #   5   moderate
            #  10   heavy; lower bound of convective rates
            #  20   heavy convective; where a smooth regression stops producing
            #  30   very heavy (~1.2 in/hr, flash-flood scale)
            "WRF_precip": [1., 2., 5., 10., 20., 30.],
            # HOURLY MEAN 10-m wind, m/s (NOT 1-min sustained -- label precisely)
            "WRF_SPD10":  [5., 10., 15., 20., 25., 30.],
        },
        var_config={
            # No 'power' key exists any more. get_field back-transforms ONLY the
            # target's fourth-root WRF_precip_025 (**4 -> physical mm/hr); every
            # other field (member/UNet WRF_precip, all winds) is read as-is.
            # target_var / member_var / unet_var just name the field per source.
            "WRF_precip": dict(target_var="WRF_precip_025",
                               member_var="WRF_precip", unet_var="WRF_precip",
                               agg="none", mask="none", unit="mm/hr"),
            "WRF_SPD10": dict(target_var="WRF_SPD10", member_var="WRF_SPD10",
                              unet_var="WRF_SPD10", agg="none",
                              mask="ocean", unit="m/s"),
        },
    )

    # STEP 1 -- choose thresholds from the data, not from round numbers:
    # recon_thresholds(CFG, candidates={
    #     "WRF_precip": [0.5, 1, 2, 5, 10, 15, 20, 30, 50],
    #     "WRF_SPD10":  [5, 10, 15, 20, 25, 30, 35]})
    #
    # STEP 2 -- verify:
    run(CFG)
