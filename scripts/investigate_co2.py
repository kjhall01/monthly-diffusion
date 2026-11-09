import xarray as xr 
import matplotlib.pyplot as plt 


co2 = xr.open_dataset(
    "/Users/kyle/Dev/monthly-diffusion-data/data/"
    "co2-forcing-v1/products/co2_monthly_e3sm_ssp370_185001-210012.nc"
).co2

co2.plot.line(x='time')
plt.savefig('co2.png')
plt.close()
