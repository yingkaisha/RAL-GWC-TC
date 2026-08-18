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

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

varnames = ['WRF_SP', 'WRF_T2', 'WRF_U10', 'WRF_V10', 'WRF_PWAT_05', 'WRF_precip_025']

N_ens = 20
years = np.arange(2020, 2025)

list_fn = sorted(glob('/glade/derecho/scratch/ksha/corrdiff_ERA5_TC_*'))[:N_ens]

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_member, fn in enumerate(list_fn):
    list_nc = sorted(glob(fn + f'/pred_{year}*.nc'))

    list_ds = []
    for fn_nc in list_nc:
        ds = xr.open_dataset(fn_nc)[varnames]
        ds = ds.squeeze('member', drop=True)        # removes size-1 dim + stale coord
        ds['WRF_PWAT']   = ds['WRF_PWAT_05']**2
        ds['WRF_precip'] = ds['WRF_precip_025']**4
        ds = ds.drop_vars(['WRF_PWAT_05', 'WRF_precip_025'])

        list_ds.append(ds)

    ds_mem = xr.concat(list_ds, dim='time')
    ds_mem = ds_mem.assign_coords(member=i_member)   # optional: tag the member id

    ds_mem = ds_mem.chunk(
        {
            'time': 1,
            'south_north': -1,
            'west_east': -1,
        }
    )

    dict_encoding = {var: {'compressor': compress} for var in ds_mem.data_vars}

    save_dir = (
        '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/'
        f'TC_pred_corrdiff/TC_ERA5_corrdiff_pred_{year}_mem{i_member:02d}.zarr'
    )
    ds_mem.to_zarr(save_dir, mode='w', consolidated=True, compute=True,
                   encoding=dict_encoding)
