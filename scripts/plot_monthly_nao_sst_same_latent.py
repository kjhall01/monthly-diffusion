"""Monthly NAO–SST maps from same-latent normal decoder reconstructions."""
from pathlib import Path
import sys, argparse
import numpy as np, pandas as pd, torch, xarray as xr, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent))
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state, open_co2

def wm(x): return x.weighted(np.cos(np.deg2rad(x.lat))).mean(("lat","lon"))
def nao(a):
    v=wm(a.mslp.sel(lat=slice(35,45),lon=slice(330,360)))-wm(a.mslp.sel(lat=slice(60,70),lon=slice(330,350)))
    return v.groupby("time.month")-v.groupby("time.month").mean("time")
def reg(field,index):
    x=field-field.mean("time"); y=index-index.mean("time")
    return ((x*y).mean("time")/(y*y).mean("time")).rename(field.name or "sst")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run",type=Path,default=ROOT/"runs/mdv1-e3smv2-r1i1p1f1"); p.add_argument("--data",type=Path,default=ROOT/"examples/e3sm-local.yaml"); p.add_argument("--output",type=Path,default=ROOT/"analysis/monthly_nao_sst_same_latent_epoch159"); p.add_argument("--device",default="cpu"); args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    source=yaml.safe_load(args.data.read_text()); a,o,_=open_canonical_state(source["state_zarr"],expected_attrs=source.get("expected_attrs")); co2=open_co2(source["co2_netcdf"],expected_sha256=source.get("co2_sha256")); a,o=a.sel(time=slice("1966-01","2006-12")),o.sel(time=slice("1966-01","2006-12")); dates=pd.DatetimeIndex(a.time.values)
    model=MonthlyDiffusionV1.load(args.run/"model.pt",device=args.device); cache=torch.load(args.run/"latents.pt",map_location="cpu",weights_only=False)
    za=xr.DataArray(cache["training"]["atmosphere"].numpy(),dims=("time","node","channel"),coords={"time":dates,"node":np.arange(162),"channel":np.arange(4)}); zo=xr.DataArray(cache["training"]["ocean"].numpy(),dims=("time","node","channel"),coords={"time":dates,"node":np.arange(162),"channel":np.arange(4)})
    _,decoded=model.decode(ocean=zo,co2=co2.sel(time=dates)); decoded=decoded.assign_coords(time=dates)
    n=nao(a); maps=[]; ref=[]
    for month in range(1,13):
        keep=dates.month==month
        maps.append(reg(decoded.sst.isel(time=np.flatnonzero(keep)),n.isel(time=np.flatnonzero(keep))))
        ref.append(reg(o.sst.isel(time=np.flatnonzero(keep)),n.isel(time=np.flatnonzero(keep))))
    maps=xr.concat(maps,dim=xr.IndexVariable("month",range(1,13))); ref=xr.concat(ref,dim=xr.IndexVariable("month",range(1,13)))
    xr.Dataset({"decoded":maps,"e3sm":ref}).to_netcdf(args.output/"monthly_nao_sst_maps.nc")
    vmax=float(np.nanpercentile(np.abs(np.concatenate([maps.values.ravel(),ref.values.ravel()])),99)); vmax=max(vmax,1e-8)
    with plt.rc_context({"font.size":8}):
        fig,ax=plt.subplots(4,3,figsize=(12,9),constrained_layout=True)
        for i,m in enumerate(range(1,13)):
            im=ax.flat[i].pcolormesh(maps.lon,maps.lat,maps.sel(month=m),cmap="RdBu_r",vmin=-vmax,vmax=vmax,shading="auto"); ax.flat[i].set_title(pd.Timestamp(2000,m,1).strftime("%B")); ax.flat[i].set_xlabel("lon"); ax.flat[i].set_ylabel("lat")
        fig.colorbar(im,ax=ax.ravel().tolist(),label="SST regression on NAO (K per index unit)",shrink=.82); fig.suptitle("Same-latent decoder · month-stratified NAO–SST regression")
        fig.savefig(args.output/"monthly_nao_sst_same_latent.png",dpi=250); fig.savefig(args.output/"monthly_nao_sst_same_latent.pdf"); plt.close(fig)
    print(f"Wrote {args.output}")
if __name__=="__main__": main()
