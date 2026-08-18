import os
import re
import sys
import time
import zarr
import numpy as np
import xarray as xr
from glob import glob
from datetime import datetime, timedelta

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year_verif = int(args['year'])

# ----------------------------------------------------------------------------- config
DATA_ROOT = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC'
OUT_ROOT  = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/verification'
 
YEARS = [2020, 2021, 2022, 2023, 2024]
 
VARNAMES = ['WRF_MSLP', 'WRF_PWAT', 'WRF_SP', 'WRF_T2', 'WRF_precip', 'WRF_SPD10']
SPEC_VARNAMES = ['WRF_MSLP', 'WRF_PWAT', 'WRF_SP', 'WRF_T2', 'WRF_precip', 'WRF_SPD10']   # spectra focus (FFT cost)
 
REGIONS    = ['domain', 'land', 'ocean']
MODELS     = ['era5', 'gdas', 'unet', 'target']         # superset for unified vars
ENS_MODELS = ['era5', 'gdas']                           # carry per-member detail
 
TIME_DIM = 'time'
DX_KM    = 8.0
 
TIME_BLOCK = 10        # per-variable block read; tune to grid/memory (IO dominates)
RANK_SEED  = 0         # reproducible random tie-breaking for rank histograms
 
F = np.float32         # stored dtype (halves disk vs float64)
 
 
def uv_to_spd(ds):
    ds['WRF_SPD10'] = np.sqrt(ds['WRF_U10'] ** 2 + ds['WRF_V10'] ** 2)
    ds = ds.drop_vars(('WRF_U10', 'WRF_V10'))
    return ds
 
 
# ------------------------------------------------------------------- scoring primitives
def fair_crps_grid(ens, obs):
    """Fair/unbiased ensemble CRPS (Ferro 2014), per grid point.
 
    ens : (M, ny, nx)   obs : (ny, nx)   ->  (ny, nx)
    O(M log M) sorted identity for the pairwise term.
    """
    M = ens.shape[0]
    term1 = np.abs(ens - obs[None]).mean(axis=0)
    ens_sorted = np.sort(ens, axis=0)
    k = np.arange(1, M + 1).reshape(M, 1, 1)
    w = (2 * k - M - 1)
    pair = (w * ens_sorted).sum(axis=0)          # = 0.5 * sum_{i,j}|x_i-x_j|
    return term1 - pair / (M * (M - 1))
 
 
def rank_grid(ens, obs, rng):
    """Observation rank among M members, per grid point, in [0, M].
    Random tie-breaking (Hamill 2001) -- essential for precip ties at zero.
    """
    below = (ens < obs[None]).sum(axis=0)
    equal = (ens == obs[None]).sum(axis=0)
    add = np.floor(rng.random(equal.shape) * (equal + 1)).astype(np.int64)
    add = np.minimum(add, equal)
    return below + add
 
 
# ------------------------------------------------------------------ spectra primitives
def make_k_bins(ny, nx, dx, nbins=None):
    ky = np.fft.fftfreq(ny, d=dx)
    kx = np.fft.fftfreq(nx, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    kmag = np.sqrt(KX ** 2 + KY ** 2)             # cycles per km
    knyq = min(np.abs(kx).max(), np.abs(ky).max())
    if nbins is None:
        nbins = min(ny, nx) // 2                   # <- reduce to shrink psd_member
    edges = np.linspace(0.0, knyq, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return kmag, edges, centers
 
 
def hann2d(ny, nx):
    return np.outer(np.hanning(ny), np.hanning(nx))
 
 
def radial_psd_stack(stack, w2d, wnorm, bin_idx, bin_sel, bin_cnts, nb):
    """Radially-binned 2D power spectrum for a stack of fields.
 
    stack : (S, ny, nx) -> (S, nb). FFT vectorized over S (members).
    De-mean, Hann-window, normalize by mean(w^2). Magnitudes are consistent-but-
    arbitrary: compare curves/ratios across models, not absolute levels.
    """
    f = stack - np.nanmean(stack, axis=(-2, -1), keepdims=True)
    f = np.nan_to_num(f, nan=0.0) * w2d[None]
    Ff = np.fft.fft2(f, axes=(-2, -1))
    psd2d = (np.abs(Ff) ** 2) / (stack.shape[-1] * stack.shape[-2]) / wnorm
    S = stack.shape[0]
    out = np.full((S, nb), np.nan)
    for s in range(S):
        ps = psd2d[s].ravel()[bin_sel]
        sums = np.bincount(bin_idx, weights=ps, minlength=nb)
        nz = bin_cnts > 0
        out[s, nz] = sums[nz] / bin_cnts[nz]
    return out
 
 
# ----------------------------------------------------------------------- block readers
def ens_block(ds, var, sl):
    da = ds[var].transpose('member', TIME_DIM, ...)
    return da.isel({TIME_DIM: sl}).values         # (M, Tb, ny, nx)
 
 
def det_block(ds, var, sl):
    da = ds[var].transpose(TIME_DIM, ...)
    return da.isel({TIME_DIM: sl}).values         # (Tb, ny, nx)
 
 
# ---------------------------------------------------------------------------- per year
def verify_year(year):
    print(f'[{year}] opening inputs', flush=True)
 
    ds_static = xr.open_zarr(f'{DATA_ROOT}/static/C404_TC_static_8km.zarr')
    land = np.squeeze(ds_static['LANDMASK'].values)
    assert land.ndim == 2, f'LANDMASK is not 2D: shape {land.shape}'
    base = np.isfinite(land)
    region_masks = {
        'domain': base,
        'land':   base & (land > 0),
        'ocean':  base & (land == 0),
    }
 
    ds_target = uv_to_spd(xr.open_zarr(f'{DATA_ROOT}/C404_CorrDiff/TC_target_{year}.zarr'))
    ds_target['WRF_PWAT'] = ds_target['WRF_PWAT_05']**2
    ds_target['WRF_precip'] = ds_target['WRF_precip_025']**4
    ds_target = ds_target.drop_vars(('WRF_PWAT_05', 'WRF_precip_025', 'WRF_Q_tot_05', 'WRF_T'))
    print(f'target vars {list(ds_target.keys())}')
    
    ds_unet   = uv_to_spd(xr.open_zarr(f'{DATA_ROOT}/TC_UNET/TC_UNET_pred_{year}_MSLP.zarr'))
    print(f'unet vars {list(ds_unet.keys())}')
    
    fn_e5 = sorted(glob(f'{DATA_ROOT}/TC_pred_corrdiff_final/TC_ERA5_corrdiff_pred_{year}_mem*.zarr'))
    fn_gd = sorted(glob(f'{DATA_ROOT}/TC_pred_corrdiff_final/TC_GDAS_corrdiff_pred_{year}_mem*.zarr'))
    assert fn_e5, f'no ERA5 members found for {year}'
    assert fn_gd, f'no GDAS members found for {year}'
    ds_era5 = uv_to_spd(xr.concat([xr.open_zarr(f) for f in fn_e5], dim='member'))
    print(f'ds_era5 vars {list(ds_era5.keys())}')
    
    ds_gdas = uv_to_spd(xr.concat([xr.open_zarr(f) for f in fn_gd], dim='member'))
    print(f'ds_gdas vars {list(ds_gdas.keys())}')
    
 
    # --- positional alignment ---
    t_common = ds_target[TIME_DIM].values
    for d in (ds_unet, ds_era5, ds_gdas):
        t_common = np.intersect1d(t_common, d[TIME_DIM].values)
    assert t_common.size > 0, f'no overlapping times for {year}'
    ds_target = ds_target.sel({TIME_DIM: t_common})
    ds_unet   = ds_unet.sel({TIME_DIM: t_common})
    ds_era5   = ds_era5.sel({TIME_DIM: t_common})
    ds_gdas   = ds_gdas.sel({TIME_DIM: t_common})
 
    T = t_common.size
    M = ds_era5.sizes['member']
    assert ds_gdas.sizes['member'] == M, 'ERA5/GDAS member counts differ (shared member dim)'
    print(f'[{year}] T={T} times, M={M} members', flush=True)
 
    ny, nx = ds_target[VARNAMES[0]].isel({TIME_DIM: 0}).values.shape
    R, V, MD, ME = len(REGIONS), len(VARNAMES), len(MODELS), len(ENS_MODELS)
    midx = {m: i for i, m in enumerate(MODELS)}
    eidx = {m: i for i, m in enumerate(ENS_MODELS)}
 
    # --- ensemble-level ingredients ---
    abs_err  = np.full((T, R, V, MD), np.nan, dtype=F)
    sq_err   = np.full((T, R, V, MD), np.nan, dtype=F)
    crps     = np.full((T, R, V, MD), np.nan, dtype=F)
    spr_var  = np.full((T, R, V, MD), np.nan, dtype=F)
    rank_h   = np.full((T, R, V, MD, M + 1), np.nan, dtype=F)
    n_valid  = np.zeros((T, R, V), dtype=np.int32)
 
    # --- per-member ingredients (ensembles only) ---
    abs_err_m = np.full((T, R, V, ME, M), np.nan, dtype=F)   # |member - obs|
    sq_err_m  = np.full((T, R, V, ME, M), np.nan, dtype=F)   # (member - obs)^2
 
    # --- spectra setup (domain only) ---
    kmag, edges, kcent = make_k_bins(ny, nx, DX_KM)
    nb = kcent.size
    w2d = hann2d(ny, nx)
    wnorm = float((w2d ** 2).mean())
    flat_idx = np.digitize(kmag.ravel(), edges) - 1
    bin_sel = (flat_idx >= 0) & (flat_idx < nb)
    bin_idx = flat_idx[bin_sel]
    bin_cnts = np.bincount(bin_idx, minlength=nb)
    SV = len(SPEC_VARNAMES)
    psd        = np.full((T, SV, MD, nb), np.nan, dtype=F)       # member-mean + refs
    psd_member = np.full((T, SV, ME, M, nb), np.nan, dtype=F)    # per-member
 
    rng = np.random.default_rng(RANK_SEED)
 
    # --- main loop: variable outer (caps memory), time-block inner ---
    for vi, v in enumerate(VARNAMES):
        is_spec = v in SPEC_VARNAMES
        svi = SPEC_VARNAMES.index(v) if is_spec else None
        for start in range(0, T, TIME_BLOCK):
            sl = slice(start, min(start + TIME_BLOCK, T))
            Tb = sl.stop - sl.start
            obs_b = det_block(ds_target, v, sl)       # (Tb,ny,nx)
            un_b  = det_block(ds_unet,   v, sl)
            e5_b  = ens_block(ds_era5,   v, sl)       # (M,Tb,ny,nx)
            gd_b  = ens_block(ds_gdas,   v, sl)
 
            for j in range(Tb):
                tt = start + j
                obs = obs_b[j]
                un  = un_b[j]
                e5m = e5_b[:, j]                       # (M,ny,nx)
                gdm = gd_b[:, j]
                e5_mean = e5m.mean(axis=0)
                gd_mean = gdm.mean(axis=0)
 
                # per-member error fields (computed once per timestep/ensemble)
                aem = {'era5': np.abs(e5m - obs[None]), 'gdas': np.abs(gdm - obs[None])}
                sem = {'era5': (e5m - obs[None]) ** 2, 'gdas': (gdm - obs[None]) ** 2}
 
                # common validity -> all models scored on identical points
                finite = (np.isfinite(obs) & np.isfinite(un)
                          & np.isfinite(e5_mean) & np.isfinite(gd_mean))
 
                ae = {'era5': np.abs(e5_mean - obs),
                      'gdas': np.abs(gd_mean - obs),
                      'unet': np.abs(un - obs)}
                se = {'era5': (e5_mean - obs) ** 2,
                      'gdas': (gd_mean - obs) ** 2,
                      'unet': (un - obs) ** 2}
                cr = {'era5': fair_crps_grid(e5m, obs),
                      'gdas': fair_crps_grid(gdm, obs),
                      'unet': ae['unet']}              # deterministic CRPS == MAE
                sv = {'era5': e5m.var(axis=0, ddof=1),
                      'gdas': gdm.var(axis=0, ddof=1)}
                rk = {'era5': rank_grid(e5m, obs, rng),
                      'gdas': rank_grid(gdm, obs, rng)}
 
                for ri, rname in enumerate(REGIONS):
                    mask = region_masks[rname] & finite
                    n = int(mask.sum())
                    n_valid[tt, ri, vi] = n
                    if n == 0:
                        continue
                    mb = mask[None]                    # (1,ny,nx) for member-vectorized means
                    for m in ('era5', 'gdas', 'unet'):
                        mi = midx[m]
                        abs_err[tt, ri, vi, mi] = ae[m][mask].mean()
                        sq_err[tt, ri, vi, mi]  = se[m][mask].mean()
                        crps[tt, ri, vi, mi]    = cr[m][mask].mean()
                    for m in ('era5', 'gdas'):
                        mi, ei = midx[m], eidx[m]
                        spr_var[tt, ri, vi, mi] = sv[m][mask].mean()
                        ranks = rk[m][mask]
                        rank_h[tt, ri, vi, mi, :] = np.bincount(ranks, minlength=M + 1)[:M + 1]
                        # per-member spatial means (vectorized over member)
                        abs_err_m[tt, ri, vi, ei, :] = (aem[m] * mb).sum(axis=(-2, -1)) / n
                        sq_err_m[tt, ri, vi, ei, :]  = (sem[m] * mb).sum(axis=(-2, -1)) / n
 
                if is_spec:
                    sp_e5 = radial_psd_stack(e5m, w2d, wnorm, bin_idx, bin_sel, bin_cnts, nb)  # (M,nb)
                    sp_gd = radial_psd_stack(gdm, w2d, wnorm, bin_idx, bin_sel, bin_cnts, nb)
                    p_un  = radial_psd_stack(un[None],  w2d, wnorm, bin_idx, bin_sel, bin_cnts, nb)[0]
                    p_tg  = radial_psd_stack(obs[None], w2d, wnorm, bin_idx, bin_sel, bin_cnts, nb)[0]
                    # per-member
                    psd_member[tt, svi, eidx['era5'], :, :] = sp_e5
                    psd_member[tt, svi, eidx['gdas'], :, :] = sp_gd
                    # unified convenience (member-mean + references)
                    psd[tt, svi, midx['era5'], :]   = np.nanmean(sp_e5, axis=0)
                    psd[tt, svi, midx['gdas'], :]   = np.nanmean(sp_gd, axis=0)
                    psd[tt, svi, midx['unet'], :]   = p_un
                    psd[tt, svi, midx['target'], :] = p_tg
 
        print(f'[{year}]   done {v}', flush=True)
 
    # --- assemble single Dataset ---
    wavelength = np.where(kcent > 0, 1.0 / kcent, np.inf)
    ds_out = xr.Dataset(
        data_vars=dict(
            abs_error       =(('time', 'region', 'variable', 'model'), abs_err),
            sq_error        =(('time', 'region', 'variable', 'model'), sq_err),
            crps            =(('time', 'region', 'variable', 'model'), crps),
            spread_var      =(('time', 'region', 'variable', 'model'), spr_var),
            rank_hist       =(('time', 'region', 'variable', 'model', 'rank'), rank_h),
            n_valid         =(('time', 'region', 'variable'), n_valid),
            abs_error_member=(('time', 'region', 'variable', 'ens_model', 'member'), abs_err_m),
            sq_error_member =(('time', 'region', 'variable', 'ens_model', 'member'), sq_err_m),
            psd             =(('time', 'spec_variable', 'model', 'wavenumber'), psd),
            psd_member      =(('time', 'spec_variable', 'ens_model', 'member', 'wavenumber'), psd_member),
        ),
        coords=dict(
            time=t_common,
            region=REGIONS,
            variable=VARNAMES,
            model=MODELS,
            ens_model=ENS_MODELS,
            member=np.arange(M),                                # = sorted mem-file order
            rank=np.arange(M + 1),
            spec_variable=SPEC_VARNAMES,
            wavenumber=kcent,                                   # cycles per km
            wavelength_km=('wavenumber', wavelength),
        ),
        attrs=dict(
            description='CorrDiff TC verification vs CONUS404 (per-timestep ingredients, '
                        'with per-member detail)',
            truth='CONUS404 (TC_target)',
            crps_estimator='fair / unbiased (Ferro 2014)',
            unet_crps='equals MAE (deterministic CRPS reduces to absolute error)',
            ensemble_mean_errors='abs_error/sq_error for era5,gdas are on the ENSEMBLE MEAN',
            per_member_errors='abs_error_member/sq_error_member treat each member as a '
                              'deterministic forecast (member-as-forecast skill + spread)',
            no_per_member='spread_var, rank_hist, crps are ensemble properties -- no '
                          'per-member form (a per-member CRPS == that member MAE == '
                          'abs_error_member)',
            spread_note='spread_var is raw ensemble variance (ddof=1); apply Fortin '
                        'sqrt(1+1/M) to RMS spread when forming the spread-skill ratio',
            rmse_note='rebuild RMSE as sqrt(time-mean MSE); do NOT average per-time RMSE',
            rank_tie_break=f'random uniform (Hamill 2001), seed={RANK_SEED}',
            spectra='radial PSD, Hann-windowed, DOMAIN ONLY; per-member spectra saved in '
                    'psd_member; psd holds the member-mean plus unet/target references',
            spectra_members=f'all {M} members used (no subsample)',
            spectra_units='arbitrary but consistent; compare ratios/curves, not levels',
            spatial_mean='unweighted over valid points (projected ~equal-area 8 km cells)',
            common_mask='all models scored on identical valid points per (time,region,var)',
            landmask_rule='land: LANDMASK>0 ; ocean: LANDMASK==0',
            dx_km=DX_KM, n_members=M,
            usage='collapse over season or per-storm time windows downstream '
                  '(see collapse_scores)',
        ),
    )
    ds_out = ds_out.chunk({'time': min(200, T)})
 
    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = f'{OUT_ROOT}/TC_verification_scores_{year}.zarr'
    ds_out.to_zarr(out_path, mode='w')
    print(f'[{year}] wrote {out_path}', flush=True)
    return out_path
 
 
# --------------------------------------------------------------- downstream aggregation
def collapse_scores(ds, time_sel=None):
    """Collapse the per-timestep ingredient Zarr into finished scores.
 
    time_sel : None (whole file) or slice/boolean/list of times. Use the SAME ds with
               different time_sel for season vs per-storm. Count-weighted in time.
 
    Returns finished scores on (region, variable, model[, ...]) plus per-member arrays
    on (..., ens_model, member) for member-skill spread and spectral envelopes.
    """
    d = ds if time_sel is None else ds.sel(time=time_sel)
    M = d.sizes['rank'] - 1
    w = d['n_valid']                                  # (time,region,variable)
 
    def tw(x):                                        # time, count-weighted mean
        return (x * w).sum('time') / w.sum('time')
 
    mae  = tw(d['abs_error'])
    mse  = tw(d['sq_error'])
    rmse = np.sqrt(mse)
    crps = tw(d['crps'])
 
    rms_spread = np.sqrt(tw(d['spread_var']))
    spread_skill = np.sqrt(1.0 + 1.0 / M) * rms_spread / rmse
 
    rh = d['rank_hist'].sum('time')
    rank_hist = rh / rh.sum('rank')
 
    crpss = 1.0 - crps.sel(model=['era5', 'gdas']) / crps.sel(model='unet')
 
    # per-member: member-as-forecast skill (use these for inter-member spread, e.g.
    # rmse_member.std('member') or the min/max envelope across members)
    mae_member  = tw(d['abs_error_member'])
    rmse_member = np.sqrt(tw(d['sq_error_member']))
 
    # spectra: member-mean curve, plus full per-member for the spectral envelope
    psd        = d['psd'].mean('time')
    psd_member = d['psd_member'].mean('time')         # (spec_variable,ens_model,member,wavenumber)
 
    return xr.Dataset(dict(
        mae=mae, rmse=rmse, crps=crps,
        spread_skill_ratio=spread_skill,
        rank_hist=rank_hist,
        crpss=crpss,
        mae_member=mae_member,
        rmse_member=rmse_member,
        psd=psd,
        psd_member=psd_member,
    ))
 
 
def smoke_test(path):
    assert os.path.exists(os.path.join(path, '.zmetadata')), \
        'missing .zmetadata -> interrupted/incomplete write'
    ds = xr.open_zarr(path)
    out = collapse_scores(ds)
    print('--- season collapse (domain) ---')
    print('CRPS:\n', out['crps'].sel(region='domain').to_pandas())
    print('CRPSS vs UNet:\n', out['crpss'].sel(region='domain').to_pandas())
    # show per-member detail is populated: spread of member RMSE for precip
    rm = out['rmse_member'].sel(region='domain', variable='WRF_precip')
    print('precip per-member RMSE (era5) min/mean/max:',
          float(rm.sel(ens_model='era5').min()),
          float(rm.sel(ens_model='era5').mean()),
          float(rm.sel(ens_model='era5').max()))
    return out


p = verify_year(year_verif)
print(p)

# /glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/verification/TC_verification_scores_2020.zarr
# /glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/verification/TC_verification_scores_2021.zarr
# /glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/verification/TC_verification_scores_2022.zarr
# /glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/verification/TC_verification_scores_2023.zarr
# /glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/verification/TC_verification_scores_2024.zarr
