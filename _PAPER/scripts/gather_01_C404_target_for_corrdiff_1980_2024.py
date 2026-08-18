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

year_pred = int(args['year'])
var_pick = ['WRF_T', 'WRF_Q_tot_05', 'WRF_SP', 'WRF_T2', 'WRF_U10', 'WRF_V10', 'WRF_PWAT_05', 'WRF_precip_025']

fn = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/C404_8km/C404_8km_{year_pred}.zarr'
ds_year = xr.open_zarr(fn)[var_pick]
ds_year = ds_year.isel(bottom_top=slice(1))

# =================================================== #
# rechunk
varname_4d = ['WRF_T', 'WRF_Q_tot_05']

ds_year = ds_year.chunk(
    {
        'time': 1, 
        'bottom_top': 12, 
        'south_north': -1, 
        'west_east': -1
    }
)

varnames = list(ds_year.keys())
# zarr encodings
dict_encoding = {}

chunk_size_3d = dict(chunks=(1, -1, -1))
chunk_size_4d = dict(chunks=(1, 12, -1, -1))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4d:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

save_dir = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/C404_CorrDiff/C404_8km_{year_pred}.zarr'
ds_year.to_zarr(save_dir, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
print(save_dir)

