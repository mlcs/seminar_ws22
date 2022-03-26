# %%
import climnet.utils.spatial_utils as sput
import climnet.utils.general_utils as gut
import climnet.utils.time_utils as tu
from climnet.dataset import BaseRectDataset
import os
import xarray as xr
import numpy as np
from importlib import reload

grid_step = 2.5
output_folder = 'climate_data'
time_range = ['1980-01-01', '2020-12-31']

dirname_sp = "/home/strnad/data/era5/surface_pressure/"
dirname_t2m = "/home/strnad/data/era5/2m_temperature/"
output_dir = '/home/strnad/data/climnet/outputs/'
plot_dir = '/home/strnad/data/climnet/plots/'


# %%
# ERA 5 single pressure levels
fname_t2m = dirname_t2m + \
    '2m_temperature_sfc_1979_2020.nc'

var_name = 't2m'
name = 'era5'
fname = fname_t2m
dataset_file = output_dir + \
    f"/{output_folder}/{name}_{var_name}_{grid_step}_ds.nc"

if os.path.exists(dataset_file) is False:
    print(f'Create Dataset {dataset_file}', flush=True)
    ds = BaseRectDataset(data_nc=fname,
                         var_name=var_name,
                         grid_step=grid_step,
                         large_ds=True,
                         )
    ds.save(dataset_file)
else:
    print(f'File {fname} already exists!', flush=True)
    ds = BaseRectDataset(load_nc=dataset_file)
# %%
reload(tu)
reload(gut)
ds_monmean = tu.apply_timemean(ds=ds.ds, timemean='month')
dataset_file_monmean = output_dir + \
    f"/{output_folder}/{name}_{var_name}_{grid_step}_monmean_ds.nc"

gut.save_ds(ds=ds_monmean,
            filepath=dataset_file_monmean)


# %%
# MSWEP single pressure level precipitation

var_name = 'pr'
name = 'mswep'
grid_step = 1
dataset_file = output_dir + \
    f"/{output_folder}/{name}_{var_name}_{grid_step}_ds.nc"


ds = BaseRectDataset(load_nc=dataset_file)
# %%
lon_range_c = [65, 95]
lat_range_c = [7, 37]
ds_pr_cut = sput.cut_map(ds.ds['pr'], lon_range=lon_range_c,
                         lat_range=lat_range_c)
ds_pr_cut = tu.get_month_range_data(ds_pr_cut,
                                    start_month='Jun',
                                    end_month='Sep')
dataset_file_india = output_dir + \
    f"/{output_folder}/{name}_{var_name}_{grid_step}_india_jjas_ds.nc"

gut.save_ds(ds=ds_pr_cut,
            filepath=dataset_file_india)