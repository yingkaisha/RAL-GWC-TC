
import os
import sys
import zarr
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob

# 2020-2024 for each experiment
fn_ERA5 = sorted(glob('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/TC_pred/pred_ERA5_*.zarr'))
fn_GDAS = sorted(glob('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/TC_pred/pred_GDAS_*.zarr'))
fn_unet = sorted(glob('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/TC_UNET/TC_UNET_pred_*_MSLP.zarr'))
fn_target = sorted(glob('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/C404_8km/C404_8km_*.zarr'))[-5:]

list_ERA5 = []
list_GDAS = []
list_unet = []
list_target = []
for i in range(5):
    # ERA5
    ds_ERA5_ = xr.open_zarr(fn_ERA5[i])
    ds_ERA5_['WRF_PWAT'] = ds_ERA5_['WRF_PWAT_05']**2
    ds_ERA5_['WRF_Q_tot'] = ds_ERA5_['WRF_Q_tot_05']**2
    ds_ERA5_['WRF_precip'] = ds_ERA5_['WRF_precip_025']**4
    ds_ERA5_ = ds_ERA5_.drop_vars(['WRF_PWAT_05', 'WRF_Q_tot_05', 'WRF_precip_025', 'forecast_hour'])
    if i == 0:
        varnames = list(ds_ERA5_.keys())
    list_ERA5.append(ds_ERA5_)
    
    # GDAS
    ds_GDAS_ = xr.open_zarr(fn_GDAS[i])
    ds_GDAS_['WRF_PWAT'] = ds_GDAS_['WRF_PWAT_05']**2
    ds_GDAS_['WRF_Q_tot'] = ds_GDAS_['WRF_Q_tot_05']**2
    ds_GDAS_['WRF_precip'] = ds_GDAS_['WRF_precip_025']**4
    ds_GDAS_ = ds_GDAS_.drop_vars(['WRF_PWAT_05', 'WRF_Q_tot_05', 'WRF_precip_025', 'forecast_hour'])
    ds_GDAS_ = ds_GDAS_[varnames]
    list_GDAS.append(ds_GDAS_)
    
    # unet
    ds_unet_ = xr.open_zarr(fn_unet[i])
    list_unet.append(ds_unet_)

    # target
    ds_target_ = xr.open_zarr(fn_target[i])
    ds_target_ = ds_target_[varnames]
    list_target.append(ds_target_)

ds_ERA5 = xr.concat(list_ERA5, dim='time')
ds_GDAS = xr.concat(list_GDAS, dim='time')
ds_unet = xr.concat(list_unet, dim='time')
ds_target = xr.concat(list_target, dim='time')

ds_unet = ds_unet.rename_vars({'level': 'bottom_top'})
# match `time` coords for all
# compute MAE for all variables found in ds_ERA5, ds_GDAS, and ds_unet
# save outputs as ds_MAE_ERA5, ds_MAE_GDAS, ds_MAE_unet

# %%
# ---------- match time coords across all experiments ---------- #
time_common = np.intersect1d(ds_ERA5['time'].values, ds_GDAS['time'].values)
time_common = np.intersect1d(time_common, ds_unet['time'].values)
time_common = np.intersect1d(time_common, ds_target['time'].values)
print('matched time steps: {}'.format(len(time_common)))

ds_ERA5 = ds_ERA5.sel(time=time_common)
ds_GDAS = ds_GDAS.sel(time=time_common)
ds_unet = ds_unet.sel(time=time_common)
ds_target = ds_target.sel(time=time_common)

# %%
# ---------- domain-averaged MAE per time step ---------- #
# target values are wrapped with the prediction coords so the
# subtraction is strictly positional, no silent xarray alignment
dict_pred = {'ERA5': ds_ERA5, 'GDAS': ds_GDAS, 'unet': ds_unet}
dict_MAE = {}

for expname, ds_pred in dict_pred.items():
    print(expname)
    dict_var = {}
    for varname in list(ds_pred.data_vars):
        print(varname)
        # skip static fields and vars missing from the target
        if 'time' not in ds_pred[varname].dims:
            continue
        if varname not in ds_target.data_vars:
            print('{}: {} not in target, skipped'.format(expname, varname))
            continue

        da_pred = ds_pred[varname]
        da_target = ds_target[varname]

        # reorder target dims if names match but order differs
        if (set(da_target.dims) == set(da_pred.dims)) and (da_target.dims != da_pred.dims):
            da_target = da_target.transpose(*da_pred.dims)

        assert da_pred.shape == da_target.shape, \
            '{} {}: shape mismatch {} vs {}'.format(expname, varname, da_pred.shape, da_target.shape)

        da_target_pos = xr.DataArray(da_target.data, dims=da_pred.dims, coords=da_pred.coords)
        dims_space = da_pred.dims[-2:]
        da_mae = np.abs(da_pred - da_target_pos).mean(dim=dims_space, skipna=True)

        # compute per variable to keep peak memory low
        dict_var[varname] = da_mae.compute()

    dict_MAE[expname] = xr.Dataset(dict_var)

ds_MAE_ERA5 = dict_MAE['ERA5']
ds_MAE_GDAS = dict_MAE['GDAS']
ds_MAE_unet = dict_MAE['unet']

# %%
# ---------- summary: MAE aggregated over all time steps ---------- #
for expname in dict_MAE.keys():
    print('========== {} =========='.format(expname))
    ds_mean = dict_MAE[expname].mean(dim='time')
    for varname in ds_mean.data_vars:
        val = ds_mean[varname].values
        if val.ndim == 0:
            print('{}: {:.6f}'.format(varname, float(val)))
        else:
            # 3d vars keep their vertical dim, values printed per level
            print('{}: {}'.format(varname, np.round(val, 6)))


ds_MAE_ERA5.to_netcdf('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/verif_hourly/MAE_LAM_ERA5_2020_2024.nc')
ds_MAE_GDAS.to_netcdf('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/verif_hourly/MAE_LAM_GDAS_2020_2024.nc')
ds_MAE_unet.to_netcdf('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/verif_hourly/MAE_LAM_unet_2020_2024.nc')



