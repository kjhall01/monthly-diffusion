import xarray as xr
import os
from datetime import datetime
import pytz
import numpy as np
import xesmf as xe 
import metpy.calc as mpcalc
from metpy.units import units
import uuid 

cc_license = 'AIMIP model data produced by MD1 is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment (presuming AIMIP adheres to the same). Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.'


def regrid_to_template(ds_src, templatefile, method='conservative'):
    ds = xr.open_dataset(templatefile)

    regridder = xe.Regridder(
        ds_src,
        ds,
        method=method,
        periodic=True
    )

    ds_1p0_center = regridder(ds_src)

    dlat=1 # templatefile is 1 degree
    lon_bnds = np.stack([ds_1p0_center.lon - dlat/2, ds_1p0_center.lon + dlat/2], axis=1)
    lat_bnds = np.stack([ds_1p0_center.lat - dlat/2, ds_1p0_center.lat + dlat/2], axis=1)

    ds_1p0_center["lat_bnds"] = (("lat", "bnds"), lat_bnds)
    ds_1p0_center["lon_bnds"] = (("lon", "bnds"), lon_bnds)
    return ds_1p0_center

def extend_template_197810_202412(templatefile):
    month_starts = pd.date_range("1978-10-01", "2024-12-01", freq='MS') # changed for ERA5
    month_ends = pd.date_range("1978-10-01", "2025-01-01", freq='ME')
    month_centers = [ month_starts[i] + (month_ends[i] - month_starts[i]) /2 for i in range(len(month_ends)) ]     
    month_centers = np.asarray(month_centers) 
    month_starts = np.asarray(month_starts) 
    month_ends = np.asarray(month_ends) 
    month_bnds = np.hstack([month_starts.reshape(-1, 1), month_ends.reshape(-1,1)]) 
    
    ds = xr.open_dataset(templatefile)
    #ds = getattr(ds, varname)
    ds = ds.isel(time=0, drop=True)
    ds = xr.concat([ds for _ in range(month_centers.shape[0]) ], 'time')
    ds = ds.assign_coords({'time': month_centers })
    ds["time_bnds"] = (("time", "bnds"), month_bnds)
    ds.to_netcdf(templatefile + 'new')
    

def cmorize_data_with_template(user_data_array, template_path, output_path, metadata_overrides, overwrite=True):
    """
    Replaces the data of the main variable in a template NetCDF file with data from a user's
    in-memory NumPy array, updates metadata, and saves a new CMOR-compliant file.

    Args:
        user_data_array (np.ndarray): The user's n-dimensional data array.
        template_path (str): Path to the CMOR-compliant template NetCDF file.
        output_path (str): Path for the new output NetCDF file.
        metadata_overrides (dict): Dictionary of global attributes to update (e.g., {'source_id': 'NewModel'}).
    """
    print(f"--- Starting CMORization Process ---")
    print(f"User data: In-memory NumPy array of shape {user_data_array.shape}")
    print(f"Template: {template_path}")
    print(f"Output: {output_path}")
    new_template = Path(str(template_path) + 'new') 
    if not new_template.is_file() or overwrite:
        extend_template_197810_202412(template_path)
        
    with xr.open_dataset(new_template, decode_times=False) as template_ds:

        # 1. Determine the main variable name from the template
        main_var_name = template_ds.attrs.get('variable_id')
        if not main_var_name or main_var_name not in template_ds.data_vars:
            raise ValueError("Cannot determine main variable from 'variable_id' in template.")
        print(f"\nIdentified main data variable: '{main_var_name}'")

        # 2. Start with a deep copy of the template to preserve all metadata
        cmor_ds = template_ds.copy(deep=True)

        # 3. Replace the data array
        print("Replacing data with user-provided array.")
        # Ensure dimensions match before replacing data
        template_shape = cmor_ds[main_var_name].shape
        if user_data_array.shape != template_shape:
            raise ValueError(f"Shape of user data array {user_data_array.shape} does not match shape of template data {template_shape}.")
        cmor_ds[main_var_name].data = user_data_array

        # 4. Apply metadata overrides
        print("\nApplying metadata overrides...")
        for key, value in metadata_overrides.items():
            if key in cmor_ds.attrs:
                print(f"  - Overriding global attribute '{key}': '{cmor_ds.attrs.get(key)}' -> '{value}'")
                cmor_ds.attrs[key] = value
            else:
                print(f"  - Adding new global attribute '{key}': '{value}'")
                cmor_ds.attrs[key] = value

        # 5. Update history
        history_update = f"{datetime.now(pytz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}: Data replaced with user-provided in-memory data."
        cmor_ds.attrs['history'] = f"{history_update} ; {template_ds.attrs.get('history', '')}"
        print("\nUpdated history attribute.")

        # 6. Prepare encodings and perform final cleanup (lessons learned)
        print("\nPreparing encodings and performing final cleanup...")
        encoding = {}
        valid_keys = {'_FillValue', 'dtype', 'scale_factor', 'add_offset', 'units', 'calendar', 'zlib', 'complevel', 'shuffle', 'fletcher32', 'contiguous', 'chunksizes', 'least_significant_digit'}

        # Demote 'height' to prevent attribute propagation issues
        if 'height' in cmor_ds.coords:
            cmor_ds = cmor_ds.reset_coords(['height'])
            print("  - Demoted 'height' from coordinate to data variable.")

        for var_name in cmor_ds.variables:
            da = cmor_ds[var_name]
            var_encoding = template_ds[var_name].encoding.copy() if var_name in template_ds else {}

            # Disable _FillValue for coordinates, bounds, and our special 'height' case
            if var_name in cmor_ds.coords or '_bnds' in var_name or var_name == 'height':
                var_encoding['_FillValue'] = None
            
            # Clean up rogue 'coordinates' attributes
            if 'coordinates' in da.attrs and var_name != main_var_name:
                 del da.attrs['coordinates']

            encoding[var_name] = {k: v for k, v in var_encoding.items() if k in valid_keys}

        # 7. Save the final file
        print("\nSaving to NetCDF...")
        cmor_ds.to_netcdf(output_path, encoding=encoding, unlimited_dims=['time'])
        print(f"\nSuccessfully created {output_path}")

    return


def generate_tracking_id():
    """Generates a new CMIP-compliant tracking_id."""
    return f"hdl:21.14100/{uuid.uuid4()}"


def map_cmor_to_kjch(v):
    mappings = {
        "huss": ['VAR_2DSFC'], # we later convert to huss from tdas
        "pr": ['MTPRSFC'],
        "ps": ['SPSFC'],
        "tas": ['VAR_2TSFC'],
        "ts": ["SKTSFC"],
        "uas": ['VAR_10USFC'],
        "vas": ['VAR_10VSFC'],
        "hus": ['q1000.0', 'q850.0', 'q700.0', 'q500.0', 'q250.0', 'q100.0', 'q50.0'],
        "ua": ['u1000.0', 'u850.0', 'u700.0', 'u500.0', 'u250.0', 'u100.0', 'u50.0'],
        "va": ['v1000.0', 'v850.0', 'v700.0', 'v500.0', 'v250.0', 'v100.0', 'v50.0'],
        "ta": ['t1000.0', 't850.0', 't700.0', 't500.0', 't250.0', 't100.0', 't50.0'],
        "zg": ['z500.0'] #['z1000.0', 'z850.0', 'z700.0', 'z500.0', 'z250.0', 'z100.0', 'z50.0']
    }
    return mappings[v]


regrid_methods = {
    "ts": "conservative_normed",
    "huss": "conservative_normed",
    "pr": "conservative",
    "ps": "conservative_normed",
    "tas": "conservative_normed",
    "uas": "conservative_normed",
    "vas": "conservative_normed",
    "hus": "conservative_normed",
    "ua": "conservative_normed",
    "va": "conservative_normed",
    "ta": "conservative_normed",
    "zg": "conservative_normed"
}



if __name__ == "__main__":
    from pathlib import Path 
    import pandas as pd 
    model_path = "MD-1p5.pth"
    ds = xr.open_mfdataset(f"p2k_forcings-{model_path[:-4]}/*.nc")
    
    #ds = xr.open_dataset("/glade/work/khall/ERA5/AIMIP-Data/era5-flat-1p5x1p5.nc").sel(time=slice("1978-10-01", "2022-12-01"))
    institution = "UMD-PARETO" 
    email = "kylehall@umd.edu"
    model_name = model_path[:-4]
    experiment = "aimip-2k"           #aimip-2k, aimip-4k 
    title = "Monthly Diffusion model output prepared for AIMIP (Uniform +2K)"
    further_info_url = "https://github.com/kjhall01/monthly-diffusion"
    version = f"v{pd.Timestamp.today().strftime('%Y%m%d')}"

    for member in [1,2,3]: #ds.member.values:
        print(f'--------- starting member {member} ----------')
        for variable in ["ts", "huss", "pr", "ps",  "tas",   "uas",  "vas", "hus", "ta", "ua", "va", "zg"]:
            print(f"   variable: {variable}")
            cur_ds = ds.sel(varlev=map_cmor_to_kjch(variable), member=member)
            #cur_ds = ds.sel(varlev=map_cmor_to_kjch(variable))

            if variable in ['ta', 'va', 'ua', 'hus']:
                cur_ds = cur_ds.rename({'varlev': 'plev'})
                cur_ds = cur_ds.assign_coords({'plev': [float(i[1:]) for i in cur_ds.plev.values]}).sortby('plev')
                cur_ds = cur_ds.isel(plev=slice(None, None, -1))
                cur_ds = cur_ds.transpose('time', 'plev', 'lat', 'lon')
            elif variable == 'zg':
                cur_ds = cur_ds.rename({'varlev': 'plev'})
                cur_ds = cur_ds.assign_coords({'plev': [float(i[1:]) for i in cur_ds.plev.values]}).sortby('plev').sel(plev=[500.0])
                cur_ds = cur_ds.transpose('time', 'plev', 'lat', 'lon')
            else:
                cur_ds = cur_ds.mean('varlev')
                cur_ds = cur_ds.transpose('time', 'lat', 'lon')

            rxixpxfx =  f"r{member}i1p1f1"
            filename = f"{variable}_Amon_{model_name}_{experiment}_{rxixpxfx}_gr_197810-202412.nc"
            out_dir = Path(f"{institution}/{model_name}/{experiment}/{rxixpxfx}/Amon/{variable}/gr/{version}")
            out_dir.mkdir(exist_ok=True, parents=True) 
            out_path = out_dir / filename

            # Define file paths
            base_dir = f"templates_p2k/{variable}/gr/v20190815/"
            template_filename = f"{variable}_Amon_MPI-ESM1-2-LR_amip_r1i1p1f1_gr_197901-199812.nc"
            template_path = os.path.join(base_dir, template_filename)

            cur_ds = regrid_to_template(cur_ds, template_path, method=regrid_methods[variable]).da

            # Check if template file exists before we proceed
            if not os.path.exists(template_path):
                print(f"Error: Template file not found at {template_path}")
                assert False, "Please run cmorize_from_template.py to generate it first."

            print(f"--- Loading user data into memory from {template_path} ---")

            user_data_array = cur_ds.values 

            if variable == 'huss':
                sp = ds.sel(varlev=['SPSFC'], member=member).mean('varlev').transpose('time', 'lat', 'lon')
               # sp = ds.sel(varlev=['SPSFC']).mean('varlev').transpose('time', 'lat', 'lon')

                sp = regrid_to_template(sp, template_path).da.values # just spatial regridding should be fine 
                user_data_array = mpcalc.specific_humidity_from_dewpoint(sp * units("Pa"), user_data_array * units("kelvin")) # need to conver to celsius and hPa

            print(f"Successfully loaded '{variable}' data into array of shape {user_data_array.shape}")

            new_tracking_id = generate_tracking_id()
            print(f"\nGenerated new tracking_id: {new_tracking_id}")

            metadata_overrides = {
                'source_id': model_name,
                'institution': institution,
                'contact': email,
                'title': title,
                'tracking_id': new_tracking_id,
                'experiment': experiment.upper(),
                'experiment_id': experiment,
                'further_info_url': further_info_url,
                'institution_id': institution.lower(),
                'nominal_resolution': '111 km',
                'license': cc_license
            }

            # 3. Call the CMORization function with the in-memory data
            print("\n--- Running CMORization with In-Memory Data ---")
            try:
                cmorize_data_with_template(
                    user_data_array=user_data_array,
                    template_path=template_path,
                    output_path=out_path,
                    metadata_overrides=metadata_overrides
                )
                print("\nExample script finished successfully.")
                print(f"Check the new file at: {out_path}")
                print(f"You can inspect it with: ncdump -h {out_path}")
            except (ValueError, FileNotFoundError) as e:
                print(f"An error occurred: {e}")
