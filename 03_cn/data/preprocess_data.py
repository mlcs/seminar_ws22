# %%
import climnet.utils.general_utils as gut
import climnet.utils.time_utils as tu
from climnet.dataset import BaseRectDataset
import os
import xarray as xr
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
