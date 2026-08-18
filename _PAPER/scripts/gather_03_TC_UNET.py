import os
import re
import sys
import time
import dask
import zarr
import yaml
import numpy as np
import xarray as xr
from glob import glob
from datetime import datetime, timedelta

sys.path.insert(0, os.path.realpath('../libs/'))
import verif_utils as vu
import plevel_utils as pu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])
GRAVITY = 9.80665

# list_fn = sorted(glob(f'/glade/derecho/scratch/ksha/DWC/RAW_OUTPUT/CONUS_TC_UNET/*{year}*/*{year}*'))

# ds_year = xr.open_mfdataset(
#     list_fn,
#     engine='netcdf4',            # or 'h5netcdf' if the files are HDF5-based (usually faster)
#     combine='nested',
#     concat_dim='time',
#     parallel=False,               # parallelizes only the metadata-open step
#     drop_variables=['forecast_hour'],
#     chunks={'time': 1, 'level': -1, 'latitude': -1, 'longitude': -1},  # set here, skip .chunk() later
#     data_vars='minimal',         # don't broadcast non-time vars across the concat dim
#     coords='minimal',
#     compat='override',           # skip cross-file coord equality checks — big win for many files
# )

# compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)
# dict_encoding = {var: {'compressor': compress} for var in ds_year.data_vars}

# save_dir = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/TC_UNET/TC_UNET_pred_{year}.zarr'

# ds_year.to_zarr(
#     save_dir,
#     mode='w',
#     consolidated=True,
#     compute=True,
#     encoding=dict_encoding,
# )

# print(save_dir)


ds = xr.open_zarr(f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/TC_UNET/TC_UNET_pred_{year}.zarr')

ds = ds.rename({'latitude': "south_north", 'longitude': "west_east"})

ds_static = xr.open_zarr('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/static/C404_TC_static_8km.zarr')
surface_gp = (ds_static["HGT_M"].values * GRAVITY).astype(np.float64)

ds['WRF_PWAT'] = ds['WRF_PWAT_05']**2
ds['WRF_precip'] = ds['WRF_precip_025']**4
ds = ds.drop_vars(('WRF_PWAT_05', 'WRF_precip_025', 'WRF_Q_tot_05',))

# ====================================== #
# MSLP
# ====================================== #
sp  = ds["WRF_SP"].values.astype(np.float32)   # (time, south_north, west_east)
t2m = ds["WRF_T2"].values.astype(np.float32)

mslp = np.zeros_like(sp, dtype=np.float32)
for t in range(sp.shape[0]):
    mslp[t] = pu.mean_sea_level_pressure_simple(sp[t], t2m[t], surface_gp)
    
ds["WRF_MSLP"] = xr.DataArray(
    mslp,
    dims=("time", "south_north", "west_east"),
    coords={"time": ds["time"], "south_north": ds["south_north"], "west_east": ds["west_east"]},
    name="MSLP",
    attrs={"units": "Pa", "long_name": "Mean sea level pressure"},
)

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)
dict_encoding = {var: {'compressor': compress} for var in ds.data_vars}

save_dir = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/TC_UNET/TC_UNET_pred_{year}_MSLP.zarr'

ds.to_zarr(
    save_dir,
    mode='w',
    consolidated=True,
    compute=True,
    encoding=dict_encoding,
)

print(save_dir)





