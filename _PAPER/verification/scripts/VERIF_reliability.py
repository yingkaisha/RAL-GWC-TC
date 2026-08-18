import re
import os
import sys
import zarr
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob

varnames = ['WRF_precip', 'WRF_SPD10']


raw_cases = [
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

def uv_to_spd(ds):
    ds['WRF_SPD10'] = np.sqrt(ds['WRF_U10'] ** 2 + ds['WRF_V10'] ** 2)
    return ds.drop_vars(('WRF_U10', 'WRF_V10'))

def natkey(s):                       # natural sort so mem2 < mem10
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def load_ensemble(pattern):
    files = sorted(glob(pattern), key=natkey)
    members = [uv_to_spd(xr.open_zarr(f))[varnames] for f in files]
    ds = xr.concat(members, dim='member')
    return ds.assign_coords(member=np.arange(ds.sizes['member']))

# def load_ensemble(pattern):
#     files = sorted(glob(pattern), key=natkey)
#     members = [xr.open_zarr(f)[varnames] for f in files]
#     ds = xr.concat(members, dim='member')
#     return ds.assign_coords(member=np.arange(ds.sizes['member']))

def cases_to_windows(raw_cases):
    """
    Adapt a flat TC catalog of tuples
        (season, hurdat_id, storm, category, track_start, track_end)
    into per-year windows for run_variable / TC_WINDOWS and a name->category map.

    Returns (windows_by_year, category_by_name):
        windows_by_year[year] = [(storm, start_dt64, end_dt64, category), ...]
        category_by_name[storm] = category
    NOTE: track_start/track_end are best-track lifetimes; for CONUS extreme
    verification the signal concentrates near approach/landfall. Offshore-only
    storms (e.g. Lee, Franklin) contribute little land precip but are relevant
    for offshore wind/surge -- per-storm scores let you drop them if desired.
    """
    windows_by_year, category_by_name = {}, {}
    for season, hurdat_id, storm, category, t0, t1 in raw_cases:
        windows_by_year.setdefault(season, []).append(
            (storm, _to_dt64(t0), _to_dt64(t1), int(category)))
        category_by_name[storm] = int(category)
    return windows_by_year, category_by_name

def _to_dt64(x):
    """Parse a datetime string/obj to numpy datetime64 (accepts space or 'T')."""
    if isinstance(x, np.datetime64):
        return x
    return np.datetime64(str(x).strip().replace(' ', 'T'))

def storm_labels_from_windows(time_values, tc_windows, warn_empty=True):
    """
    tc_windows: list of (name, start, end) or (name, start, end, category) with
    inclusive datetime bounds (e.g. from HURDAT2 best-track over each storm's
    CONUS-relevant life). Returns (labels[str], keep_mask).

    Times in no window are dropped; times in a window get that storm's NAME.
    Overlapping windows (coincident storms) are resolved deterministically by
    list order (later entry wins) -- harmless for the COMBINED metric since the
    union is taken and each timestep is counted once, but per-storm attribution
    of overlap hours is ambiguous on a full domain (use a storm-following box to
    separate coincident storms cleanly).
    """
    t = np.asarray(time_values)
    labels = np.full(t.size, '', dtype='<U40')
    keep = np.zeros(t.size, bool)
    empties = []
    for entry in tc_windows:
        name, t0, t1 = entry[0], _to_dt64(entry[1]), _to_dt64(entry[2])
        sel = (t >= t0) & (t <= t1)
        if not sel.any():
            empties.append(name)
        labels[sel] = name
        keep |= sel
    if warn_empty and empties:
        warnings.warn("TC windows with zero matching timesteps in this "
                      f"extraction: {empties}. Check the storm is in this year's "
                      "TC_*_{year}.zarr and the dates are correct.")
    return labels, keep

def common_time_intersection(*dsets):
    """Intersect the time axes of all datasets; return the shared values."""
    t = dsets[0]['time'].values
    for ds in dsets[1:]:
        t = np.intersect1d(t, ds['time'].values)
    if t.size == 0:
        raise ValueError("Empty time intersection across sources — check that "
                         "target/unet/era5/gdas cover the same TC timesteps.")
    return t

base = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC'
TC_WINDOWS, _ = cases_to_windows(raw_cases)

list_target = []
list_era5 = []
list_gdas = []

for year in range(2020, 2025):
    ds_target = uv_to_spd(xr.open_zarr(f'{base}/C404_CorrDiff/TC_target_{year}.zarr'))
    ds_target['WRF_precip'] = ds_target['WRF_precip_025'] ** 4   # -> physical
    ds_target = ds_target[varnames]

    # ds_unet = uv_to_spd(xr.open_zarr(f'{base}/TC_UNET/TC_UNET_pred_{year}_MSLP.zarr'))[varnames]
    ds_era5 = load_ensemble(f'{base}/TC_pred_corrdiff_final/TC_ERA5_corrdiff_pred_{year}_mem*.zarr')
    ds_gdas = load_ensemble(f'{base}/TC_pred_corrdiff_final/TC_GDAS_corrdiff_pred_{year}_mem*.zarr')
    
    tc_window = TC_WINDOWS.get(year)
    ct = common_time_intersection(ds_target, ds_era5, ds_gdas)
    labels_t, keep_t = storm_labels_from_windows(ct, tc_window)

    ds_era5_tc = ds_era5.sel(time=ct).isel(time=keep_t)
    ds_gdas_tc = ds_gdas.sel(time=ct).isel(time=keep_t)
    ds_target_tc = ds_target.sel(time=ct).isel(time=keep_t)

    list_era5.append(ds_era5_tc)
    list_gdas.append(ds_gdas_tc)
    list_target.append(ds_target_tc)

ds_era5_all = xr.concat(list_era5, dim='time').sortby('time')
ds_gdas_all = xr.concat(list_gdas, dim='time').sortby('time')
ds_target_all = xr.concat(list_target, dim='time').sortby('time')

n_bins = 12
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

THRESHOLDS = {
    'WRF_precip': [1., 10., 30.],
    'WRF_SPD10': [15., 20., 30.],
}

for var_ in varnames:
    print(var_)
    da_era5 = ds_era5_all[var_]
    da_gdas = ds_gdas_all[var_]
    da_target = ds_target_all[var_]
    
    dict_fcst = {'era5': da_era5, 'gdas': da_gdas}
    dict_result = {}
    
    for i_thres, threshold in enumerate(THRESHOLDS[var_]):
        print(threshold)
        obs_binary = (da_target >= threshold).astype(float)
        o_flat_full = obs_binary.values.ravel()
        valid_obs_mask = ~np.isnan(o_flat_full)
        
        # Precompute base rate once per threshold
        o_valid = o_flat_full[valid_obs_mask]
        base_rate = float(o_valid.mean())
        
        for key_name, da_fcst in dict_fcst.items():
            # This computes fraction of members >= threshold directly
            prob_forecast = (da_fcst >= threshold).mean(dim='member')
            
            p_flat_full = prob_forecast.values.ravel()
            
            valid_mask = valid_obs_mask & ~np.isnan(p_flat_full)
            p_flat = p_flat_full[valid_mask]
            o_flat = o_flat_full[valid_mask]
            N = len(p_flat)
            
            bin_indices = np.clip(np.searchsorted(bin_edges, p_flat, side='right') - 1, 0, n_bins - 1)
            
            # Single-pass vectorized aggregation
            counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
            sum_prob = np.bincount(bin_indices, weights=p_flat, minlength=n_bins)
            sum_obs = np.bincount(bin_indices, weights=o_flat, minlength=n_bins)
            
            # Safe division (avoid divide-by-zero warnings)
            with np.errstate(invalid='ignore'):
                mean_prob = np.where(counts > 0, sum_prob / counts, np.nan)
                obs_freq = np.where(counts > 0, sum_obs / counts, np.nan)
                
            brier = float(np.mean((p_flat - o_flat) ** 2))
            
            dict_result[f'{key_name}_thresID_{i_thres}'] = {
                'base_rate': base_rate,
                'mean_prob': mean_prob,
                'obs_freq': obs_freq,
                'counts': counts,
                'brier': brier,
            }
            
    dict_result['thres'] = THRESHOLDS[var_]
    print(dict_result['thres'])
    save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/verif_hourly/reliability_{var_}.npy'
    np.save(save_name, dict_result, allow_pickle=True)