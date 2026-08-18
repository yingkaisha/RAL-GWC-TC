
import os
import sys
import time
import dask
import zarr
import numpy as np
import xarray as xr
from glob import glob

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

# varname_4d = ['WRF_Q_tot_05', 'WRF_T']
# fn = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/C404_8km/C404_8km_{year}.zarr'
# ds_year = xr.open_zarr(fn)
# ds_year = ds_year[['WRF_Q_tot_05', 'WRF_T', 'WRF_SP', 'WRF_T2', 'WRF_U10', 'WRF_V10', 'WRF_PWAT_05', 'WRF_precip_025']].isel(bottom_top=slice(1))

# ds_year = ds_year.chunk(
#     {
#         'time': 1, 
#         'bottom_top': 1, 
#         'south_north': 336, 
#         'west_east': 336
#     }
# )

# varnames = list(ds_year.keys())
# # zarr encodings
# dict_encoding = {}

# chunk_size_3d = dict(chunks=(1, 336, 336))
# chunk_size_4d = dict(chunks=(1, 1, 336, 336))

# compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

# for i_var, var in enumerate(varnames):
#     if var in varname_4d:
#         dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
#     else:
#         dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

# save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/C404_CorrDiff/C404_8km_{year}.zarr'
# ds_year.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
# print(save_name)

# ========================================================================================================================== #
varname_4d = ['Q', 'T',]
fn = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/ERA5/ERA5_8km_{year}.zarr'
ds_year = xr.open_zarr(fn)
ds_year = ds_year[['SP', 'VAR_2T', 'VAR_10U', 'VAR_10V', 'PWAT_05', 'precip_025', 'T', 'Q']].isel(level=slice(1))

ds_year = ds_year.chunk(
    {
        'time': 1, 
        'level': 1,
        'south_north': 336, 
        'west_east': 336
    }
)


varnames = list(ds_year.keys())

# zarr encodings
dict_encoding = {}

chunk_size_3d = dict(chunks=(1, 336, 336))
chunk_size_4d = dict(chunks=(1, 1, 336, 336))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4d:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_TC/ERA5_CorrDiff/ERA5_8km_{year}.zarr'
ds_year.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
print(save_name)

