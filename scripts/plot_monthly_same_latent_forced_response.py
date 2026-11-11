"""Month-stratified same-latent decoder forced-response maps."""
from pathlib import Path
import sys,argparse
import numpy as np,pandas as pd,torch,xarray as xr,yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent))
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state,open_co2

def main():
 p=argparse.ArgumentParser(); p.add_argument("--run",type=Path,default=ROOT/"runs/mdv1-e3smv2-r1i1p1f1"); p.add_argument("--data",type=Path,default=ROOT/"examples/e3sm-local.yaml"); p.add_argument("--output",type=Path,default=ROOT/"analysis/monthly_same_latent_forced_response_epoch159"); p.add_argument("--device",default="cpu"); p.add_argument("--variables",nargs="+",default=["t2m","sst","mtpr"]); args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
 source=yaml.safe_load(args.data.read_text()); a,o,_=open_canonical_state(source["state_zarr"],expected_attrs=source.get("expected_attrs")); co2=open_co2(source["co2_netcdf"],expected_sha256=source.get("co2_sha256")); a,o=a.sel(time=slice("1966-01","2006-12")),o.sel(time=slice("1966-01","2006-12")); dates=pd.DatetimeIndex(a.time.values)
 model=MonthlyDiffusionV1.load(args.run/"model.pt",device=args.device); cache=torch.load(args.run/"latents.pt",map_location="cpu",weights_only=False)
 za=xr.DataArray(cache["training"]["atmosphere"].numpy(),dims=("time","node","channel"),coords={"time":dates,"node":np.arange(162),"channel":np.arange(4)}); zo=xr.DataArray(cache["training"]["ocean"].numpy(),dims=("time","node","channel"),coords={"time":dates,"node":np.arange(162),"channel":np.arange(4)})
 cearly=co2.sel(time=slice("1850-01","1880-12")).groupby("time.month").mean("time"); clate=co2.sel(time=slice("2070-01","2100-12")).groupby("time.month").mean("time")
 # Synthetic unique timestamps preserve each requested month while allowing
 # the public CO2 lookup to validate unique monthly coordinates.
 early_dates=pd.date_range("1966-01-01",periods=len(dates),freq="MS"); late_dates=pd.date_range("1966-01-01",periods=len(dates),freq="MS")
 early_co2=xr.DataArray(cearly.sel(month=early_dates.month).values,dims="time",coords={"time":early_dates},name="co2"); late_co2=xr.DataArray(clate.sel(month=late_dates.month).values,dims="time",coords={"time":late_dates},name="co2")
 zae=za.assign_coords(time=early_dates); zoe=zo.assign_coords(time=early_dates); zal=za.assign_coords(time=late_dates); zol=zo.assign_coords(time=late_dates)
 da,ea=model.decode(atmos=zae,ocean=zoe,co2=early_co2); dl,el=model.decode(atmos=zal,ocean=zol,co2=late_co2)
 fields={**{k:dl[k]-da[k] for k in dl.data_vars},
         **{k:el[k]-ea[k] for k in el.data_vars}}
 # The latents are identical; this is late-minus-early decoder conditioning.
 for var in args.variables:
  source_field=fields[var]; maps=[]
  for month in range(1,13):
   maps.append(source_field.isel(time=np.flatnonzero(dates.month==month)).mean("time",skipna=True))
  response=xr.concat(maps,dim=xr.IndexVariable("month",range(1,13))); response.to_netcdf(args.output/f"{var}_monthly_response.nc")
  annual=response.mean("month",skipna=True); annual.to_netcdf(args.output/f"{var}_annual_response.nc")
  vmax=float(np.nanpercentile(np.abs(response.values),98)); vmax=max(vmax,1e-8)
  with plt.rc_context({"font.size":8}):
   fig,ax=plt.subplots(4,3,figsize=(12,9),constrained_layout=True)
   for i,m in enumerate(range(1,13)):
    im=ax.flat[i].pcolormesh(response.lon,response.lat,response.sel(month=m),cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto"); ax.flat[i].set_title(pd.Timestamp(2000,m,1).strftime("%B")); ax.flat[i].set_xlabel("lon"); ax.flat[i].set_ylabel("lat")
   fig.colorbar(im,ax=ax.ravel().tolist(),label=f"Late − early same-latent response ({var})",shrink=.82); fig.suptitle(f"Monthly same-latent forced response · {var}")
   fig.savefig(args.output/f"{var}_monthly_response.png",dpi=250); fig.savefig(args.output/f"{var}_monthly_response.pdf"); plt.close(fig)
   fig,ax=plt.subplots(figsize=(7.5,3.8),constrained_layout=True)
   im=ax.pcolormesh(annual.lon,annual.lat,annual,cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto"); ax.set(xlabel="lon",ylabel="lat",title=f"Annual-mean same-latent forced response · {var}"); fig.colorbar(im,ax=ax,label=f"Late − early ({var})",shrink=.82)
   fig.savefig(args.output/f"{var}_annual_response.png",dpi=250); fig.savefig(args.output/f"{var}_annual_response.pdf"); plt.close(fig)
 print(f"Wrote {args.output}")
if __name__=="__main__": main()
