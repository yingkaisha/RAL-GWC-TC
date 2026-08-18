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

def sort_files(fn_list):
    """Sort .nc filenames by the trailing _<integer>.nc index."""
    return sorted(fn_list, key=lambda x: int(re.search(r'_(\d+)\.nc$', x).group(1)))

# ========= #
year_pred = int(args['year'])
fmt = '%Y-%m-%dT%H'
var_pick = ['WRF_T', 'WRF_Q_tot_05', 'WRF_SP', 'WRF_T2', 'WRF_U10', 'WRF_V10', 'WRF_PWAT_05', 'WRF_precip_025']
ds_example = xr.open_zarr('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/C404_8km/C404_8km_2000.zarr')
# ========= #

source_dir = f'/glade/derecho/scratch/ksha/DWC/RAW_OUTPUT/CONUS_TC_ERA5_HIST/{year_pred}*/*'

dt_start = f'{year_pred}-01-01T00'
dt_end = f'{year_pred}-12-31T23'

list_fn = sort_files(glob(source_dir))

list_ds = []
for fn in list_fn:
    ds = xr.open_dataset(fn, chunks={})[var_pick]
    ds = ds.isel(level=slice(1))
    list_ds.append(ds)

ds_year = xr.concat(list_ds, dim='time')
ds_year = ds_year.sel(time=slice(dt_start, dt_end))

rename_map = {old: new for old, new in (
    ('level',     'bottom_top'),
    ('latitude',  'south_north'),
    ('longitude', 'west_east'),
) if old in ds_year.dims}
ds_year = ds_year.rename(rename_map)

ds_year['south_north'] = ds_example['south_north'].values
ds_year['west_east'] = ds_example['west_east'].values
ds_year['bottom_top'] = ds_example['bottom_top'].values[:1]

# =================================================== #
# rechunk
varname_4d = ['WRF_P', 'WRF_U', 'WRF_V', 'WRF_T', 'WRF_Q_tot_05']

ds_year = ds_year.chunk(
    {
        'time': 1, 
        'bottom_top': -1, 
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

save_dir = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/pred_C404_for_corrdiff/pred_{year_pred}.zarr'
ds_year.to_zarr(save_dir, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
print(save_dir)



