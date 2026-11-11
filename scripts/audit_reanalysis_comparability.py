"""Read-only checkpoint/data/coastal-SST audit of the matched frozen models.

Distribution statistics use every finite training value, without latitude
weights, matching Preprocessor.fit. Plot KDEs use a fixed uniform sample over
grid cells and months; missing values are excluded, genuine zeros retained.
Nothing here fits or changes a model or its preprocessing.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.data.preprocessing import transform, inverse_transform
from monthly_diffusion.analysis.spatial import weighted_mean
from monthly_diffusion.analysis.long_rollout_plots import display
from monthly_diffusion.data.dataloader import month_features

RUNS = dict(e3sm=ROOT/"runs/mdv1-e3smv2-r1i1p1f1", era=ROOT/"runs/mdv1-era5-oras5")
ROLLOUTS = dict(e3sm=RUNS["e3sm"]/"long_rollout/ssp370_1850_2100_5member",
    era=RUNS["era"]/"long_rollout/ssp370_1850_2100_5member_gaussian_spinup1845")
DIAGNOSTICS = dict(e3sm=RUNS["e3sm"]/"long_rollout_ocean_mask_aware_epoch176/full_suite",
    era=ROLLOUTS["era"]/"diagnostics")
LABELS = dict(e3sm="E3SMv2", era="ERA5/ORAS5")
COLORS = dict(e3sm="#333333", era="#0072B2")
STAGES = ("raw", "transformed", "zonal_removed", "standardized")


def finish(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def architecture(models, sources, output):
    a, b = models["e3sm"], models["era"]
    ca, cb = a.config.to_dict(), b.config.to_dict()
    result = dict(config_differences={k:[ca[k],cb[k]] for k in ca if ca[k] != cb[k]},
        training_co2_identical=np.array_equal(a.preprocessor.training_co2, b.preprocessor.training_co2),
        training_dates_identical=np.array_equal(a.preprocessor.training_times, b.preprocessor.training_times),
        same_grid=bool(a.statics.lat.equals(b.statics.lat) and a.statics.lon.equals(b.statics.lon)),
        sources=sources, models={}, statics={}, source_hashes={})
    for k,m in models.items():
        record=json.loads((ROLLOUTS[k]/"forecast.json").read_text())
        result["models"][k]=dict(checkpoint_sha256=m.weight_digest(), stages=m.stages, config=m.config.to_dict(),
            rollout_checkpoint_matches=record["checkpoint_sha256"]==m.weight_digest(),
            rollout_config=record["config"], rollout_initialization=record.get("initialization_kind","encoded"),
            rollout_start=record["valid_times"][0], sampler=record["sampler"],
            grid=dict(m.statics.sizes), latent_nodes=m.geometry.graphs["mesh_l2"].n_nodes,
            autoencoder_parameters=sum(p.numel() for p in m.autoencoder.parameters()),
            transition_parameters={c:sum(p.numel() for p in getattr(m.transition,c).parameters())
                for c in ("atmosphere","ocean")},
            co2_mean=m.preprocessor.co2_mean, co2_std=m.preprocessor.co2_std,
            topo_mean=m.preprocessor.topo_mean, topo_std=m.preprocessor.topo_std,
            co2_levels=sorted(m.autoencoder.ocean.encoder.context.co2_film_levels),
            film_parameters={n:float(p.norm()) for n,p in m.autoencoder.named_parameters() if "co2_shift" in n})
    for v in ("lsm","phi_s"):
        x,y=a.statics[v].values,b.statics[v].values
        result["statics"][v]=dict(identical=np.array_equal(x,y),rms_difference=float(np.sqrt(np.mean((x-y)**2))),
            max_abs_difference=float(np.max(abs(x-y))))
    for p in [ROOT/"monthly_diffusion"/f for f in ("modules/film.py","modules/autoencoder.py",
        "modules/ecc.py","modules/_ecc_padded.py","components/coupled_latent_diffusion.py",
        "components/sampling.py","data/preprocessing.py","forecast.py")]:
        result["source_hashes"][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    # Geometry must agree in edge connectivity as well as in node counts.
    result["identical_geometry"] = all(
        torch.equal(g.edge_set.sender,b.geometry.maps[name].edge_set.sender) and
        torch.equal(g.edge_set.receiver,b.geometry.maps[name].edge_set.receiver) and
        torch.equal(g.edge_set.edge_attr,b.geometry.maps[name].edge_set.edge_attr)
        for name,g in a.geometry.maps.items())
    (output/"architecture.json").write_text(json.dumps(result,indent=2))
    print("architecture",json.dumps({k:result[k] for k in ("config_differences","training_co2_identical","identical_geometry","statics")}),flush=True)


def distributions(models, fields, output):
    samples=output/"samples";samples.mkdir(exist_ok=True)
    rows=[]
    for k,m in models.items():
        ds=fields[k].sel(time=m.preprocessor.training_times)
        # Same random spatial sample for both data sources, every variable/stage.
        picks=np.random.default_rng(572).integers(0,ds.sizes["lat"]*ds.sizes["lon"],size=(ds.sizes["time"],256))
        for spec in m.config.variables:
            name=spec.name; sample_path=samples/f"{k}_{name}.npz"
            row_path=samples/f"{k}_{name}.json"
            if sample_path.exists() and row_path.exists():
                rows.append(json.loads(row_path.read_text()));continue
            saved=m.preprocessor.stats[name];center=np.asarray(saved["center"])
            nlat,nlon=ds.sizes["lat"],ds.sizes["lon"]
            counts=np.zeros((nlat,nlon),dtype=int);zonal_sum=np.zeros(nlat);zonal_count=np.zeros(nlat)
            n=zero=negative=floor=ceiling=nonfinite=0
            raw_min,raw_max=np.inf,-np.inf
            sum_y=sum_y2=sum_r=sum_r2=0.
            inverse_error=0.;parts={s:[] for s in STAGES}
            for start in range(0,len(ds.time),12):
                raw=ds[name].isel(time=slice(start,start+12)).values.astype("float64")
                valid=np.isfinite(raw);count=int(valid.sum());n+=count;counts+=valid.sum(0)
                zero+=int((raw[valid]==0).sum());negative+=int((raw[valid]<0).sum())
                raw_min=min(raw_min,float(np.nanmin(raw)));raw_max=max(raw_max,float(np.nanmax(raw)))
                y=np.where(valid,transform(raw,spec),np.nan)
                r=y-center[None,:,None] if center.ndim else y-center
                nonfinite+=int((valid & ~np.isfinite(y)).sum())
                if spec.transform=="softplus_inverse":floor+=int((raw[valid]/spec.scale<=spec.eps).sum())
                if spec.transform=="logit":
                    floor+=int((raw[valid]<=spec.eps).sum());ceiling+=int((raw[valid]>=1-spec.eps).sum())
                # Measure the inverse against the explicitly guarded input, not
                # against unguarded zeros (which necessarily become epsilon).
                guarded=np.clip(raw,spec.eps,1-spec.eps) if spec.transform=="logit" else (
                    np.maximum(raw,spec.eps*spec.scale) if spec.transform=="softplus_inverse" else raw)
                inverse_error=max(inverse_error,float(np.nanmax(abs(inverse_transform(y,spec)-guarded))))
                yf=y[valid];rf=r[valid]
                sum_y+=yf.sum();sum_y2+=np.dot(yf,yf);sum_r+=rf.sum();sum_r2+=np.dot(rf,rf)
                zonal_sum+=np.nansum(y,axis=(0,2));zonal_count+=valid.sum((0,2))
                index=picks[start:start+len(raw)]
                for s,values in zip(STAGES,(raw,y,r,r/saved["std"])):
                    picked=np.take_along_axis(values.reshape(len(raw),-1),index,axis=1).ravel()
                    parts[s].append(picked[np.isfinite(picked)].astype("float32"))
            fitted=np.divide(zonal_sum,zonal_count,out=np.zeros(nlat),where=zonal_count>0)
            actual_std=float(np.sqrt(max(0.,sum_r2/n-(sum_r/n)**2)))
            row=dict(dataset=k,variable=name,component=spec.component,units=ds[name].attrs.get("units"),
                transform=spec.transform,transform_scale=spec.scale,epsilon=spec.eps,
                count=n,missing_fraction=1-n/(len(ds.time)*nlat*nlon),raw_min=raw_min,raw_max=raw_max,
                zero_fraction=zero/n,negative_fraction=negative/n,lower_guard_fraction=floor/n,upper_guard_fraction=ceiling/n,
                transformed_mean=sum_y/n,transformed_std=float(np.sqrt(max(0.,sum_y2/n-(sum_y/n)**2))),
                saved_residual_std=saved["std"],recomputed_residual_std=actual_std,
                relative_std_error=abs(actual_std/saved["std"]-1),
                center_max_abs_error=float(np.max(abs(fitted-center))),
                standardized_mean=sum_r/n/saved["std"],standardized_std=actual_std/saved["std"],
                finite_source_nonfinite_transform=nonfinite,inverse_max_abs_error=inverse_error,
                any_valid_cells=int((counts>0).sum()),changing_validity_cells=int(((counts>0)&(counts<len(ds.time))).sum()),
                saved_mask_matches=bool(np.array_equal(counts>0,np.asarray(saved["mask"]))))
            arrays={s:np.concatenate(v) for s,v in parts.items()}
            row.update({f"standardized_q{q:g}":float(np.percentile(arrays["standardized"],q)) for q in [.1,1,50,99,99.9]})
            np.savez_compressed(sample_path,**arrays);row_path.write_text(json.dumps(row,indent=2));rows.append(row)
            print(k,name,"raw",raw_min,raw_max,"norm std",row["standardized_std"],flush=True)
    pd.DataFrame(rows).to_csv(output/"training_data_statistics.csv",index=False)
    titles=dict(raw="Raw physical fields",transformed="After nonlinear transform",
                zonal_removed="Transformed − training annual zonal mean",standardized="Final model coordinates: zonal residual / global standard deviation")
    for stage in STAGES:
        fig,axes=plt.subplots(5,5,figsize=(17,15),constrained_layout=True)
        for spec,ax in zip(models["era"].config.variables,axes.flat):
            values={k:np.load(samples/f"{k}_{spec.name}.npz")[stage] for k in models}
            unit="model units" if stage=="standardized" else "transformed units"
            if stage=="raw":
                for k in values:
                    shown,unit=display(spec.name,xr.DataArray(values[k],attrs={"units":spec.units}));values[k]=shown.values
            pooled=np.concatenate(list(values.values()))
            lo,hi=np.quantile(pooled,[.001,.999]);span=max(float(hi-lo),1.e-8)
            grid=np.linspace(lo-.04*span,hi+.04*span,240)
            for k,y in values.items():
                if y.std()==0: ax.axvline(y[0],color=COLORS[k],label=LABELS[k]);continue
                take=np.random.default_rng(51).choice(len(y),size=min(12000,len(y)),replace=False)
                ax.plot(grid,gaussian_kde(y[take])(grid),color=COLORS[k],lw=1.3,label=LABELS[k])
            ax.set_title(f"{spec.name} · {'atmos' if spec.component=='atmosphere' else 'ocean'}",fontsize=10)
            ax.set_xlabel(unit,fontsize=8);ax.tick_params(labelsize=8);ax.grid(alpha=.14)
            ax.ticklabel_format(axis="both",style="sci",scilimits=(-3,4));ax.xaxis.get_offset_text().set_fontsize(7)
            ax.yaxis.get_offset_text().set_fontsize(7)
        axes[0,0].legend(frameon=False,fontsize=8)
        for ax in axes[:,0]:ax.set_ylabel("Density",fontsize=9)
        fig.suptitle(titles[stage]+"\n1966–2006 · finite cells only; true zeros retained · unweighted pooling matches normalization · central 99.8% display",fontsize=13)
        finish(fig,output/f"distributions_{stage}")


def coast_mask(wet):
    land=~wet
    neighbors=np.roll(land,1,axis=1)|np.roll(land,-1,axis=1)
    neighbors[1:]|=land[:-1];neighbors[:-1]|=land[1:]
    return wet & neighbors


def coastal_sst(models, fields, sources, output):
    rows=[];maps={}
    for k,m in models.items():
        path=output/f"{k}_sst_climatology.nc"
        if path.exists():
            maps[k]=xr.load_dataset(path)
        else:
            cache=torch.load(RUNS[k]/"latents.pt",map_location="cpu",weights_only=False)
            assert cache["ae_digest"]==m.weight_digest(autoencoder_only=True)
            dates=np.asarray(cache["training_time"])
            co2=open_co2(sources[k]["co2_netcdf"],expected_sha256=sources[k].get("co2_sha256"))
            total=None
            for start in range(0,len(dates),12):
                z=xr.DataArray(cache["training"]["ocean"][start:start+12].numpy(),dims=("time","node","channel"),
                               coords=dict(time=dates[start:start+12]))
                _,decoded=m.decode(ocean=z,co2=co2)
                part=decoded.sst.sum("time",skipna=False)
                total=part if total is None else total+part
            raw=fields[k].sst.sel(time=dates).mean("time").compute()
            # Recompute the reference from the TRAINING MEMBER, not E3SM's
            # four held-out reference members used by its long-run plots.
            rollout=xr.load_dataset(DIAGNOSTICS[k]/"model/sst.nc").climatology.sel(period="training",drop=True)
            maps[k]=xr.Dataset(dict(reference=raw,autoencoded=total/len(dates),rollout=rollout,
                lsm=m.statics.lsm,phi_s=m.statics.phi_s))
            maps[k].to_netcdf(path)
        ds=maps[k];wet=np.isfinite(ds.reference.values)
        coast=coast_mask(wet)
        regions=dict(all_ocean=wet,coastal_1cell=coast,interior=wet&~coast,
            nino34=wet&((ds.lat.values[:,None]>=-5)&(ds.lat.values[:,None]<=5))&
                ((ds.lon.values[None,:]>=190)&(ds.lon.values[None,:]<=240)))
        for name,mask in regions.items():
            select=xr.DataArray(mask,dims=("lat","lon"),coords=dict(lat=ds.lat,lon=ds.lon))
            for label in ("autoencoded","rollout"):
                bias=(ds[label]-ds.reference).where(select)
                rows.append(dict(dataset=k,comparison=label,region=name,cells=int(mask.sum()),
                    bias=float(weighted_mean(bias)),rmse=float(np.sqrt(weighted_mean(bias**2)))))
    pd.DataFrame(rows).to_csv(output/"coastal_sst_metrics.csv",index=False)
    import cartopy.crs as ccrs
    fig,axes=plt.subplots(2,3,figsize=(16,7),subplot_kw=dict(projection=ccrs.Robinson()),constrained_layout=True)
    for i,k in enumerate(models):
        ds=maps[k];wet=np.isfinite(ds.reference.values)
        for j,(values,title) in enumerate([(ds.autoencoded-ds.reference,"Autoencoder − source"),
                (ds.rollout-ds.reference,"Rollout − source"),
                (xr.DataArray(wet.astype(float),dims=("lat","lon"),coords=dict(lat=ds.lat,lon=ds.lon))- (1-ds.lsm),"SST-valid mask − (1−LSM)")]):
            limit=3. if j<2 else 1.
            im=axes[i,j].pcolormesh(ds.lon,ds.lat,values,transform=ccrs.PlateCarree(),cmap="RdBu_r",vmin=-limit,vmax=limit,rasterized=True)
            axes[i,j].coastlines(linewidth=.45);axes[i,j].set_global();axes[i,j].set_title(LABELS[k]+" · "+title,fontsize=10)
            fig.colorbar(im,ax=axes[i,j],orientation="horizontal",shrink=.8,pad=.035,label="°C" if j<2 else "fraction",extend="both" if j<2 else "neither")
    fig.suptitle("Training-period SST climatology and support · 1966–2006\nEach model compared with its own training source; coastal support is not identical between sources",fontsize=12)
    finish(fig,output/"sst_coastal_bias")


def spread_and_replay(models, output):
    """Check saved state continuity, replay selected months, and freeze CO2."""
    rows=[]
    for k,m in models.items():
        record=json.loads((ROLLOUTS[k]/"forecast.json").read_text())
        dates=np.array(record["valid_times"],dtype="datetime64[M]")
        for month in ("1895-01","1898-01","1900-01"):
            index=int(np.flatnonzero(dates==np.datetime64(month,"M"))[0])
            previous=torch.load(ROLLOUTS[k]/f"month-{index:06d}.zarr/forecast_state.pt",map_location="cpu",weights_only=False)
            current=torch.load(ROLLOUTS[k]/f"month-{index+1:06d}.zarr/forecast_state.pt",map_location="cpu",weights_only=False)
            times=[previous["times"][0],previous["times"][1],current["times"][0]]
            calendar=month_features(xr.DataArray(times,dims="time")).reshape(1,6).expand(record["members"],-1)
            ah,oh=[torch.cat((h[:,0],h[:,1]),-1) for h in previous["history"]]
            with torch.no_grad():
                a,o,_,_=m.transition.sample(calendar,ah,oh,noise=tuple(current["noise"]))
            for component,pred,old,new in zip(("atmosphere","ocean"),(a,o),previous["history"],current["history"]):
                rows.append(dict(dataset=k,month=month,check="transition_replay",variable=component,
                    max_abs_error=float((pred-new[:,0]).abs().max()),
                    rms_error=float((pred-new[:,0]).square().mean().sqrt()),
                    history_shift_exact=torch.equal(old[:,0],new[:,1]),step_continuous=current["step"]==previous["step"]+1))
            saved=xr.open_zarr(ROLLOUTS[k]/f"month-{index+1:06d}.zarr",chunks=None,consolidated=True)
            latents=[xr.DataArray(h[:,0].numpy()[None],dims=("time","member","node","channel"),
                coords=dict(time=saved.time,member=saved.member)) for h in current["history"]]
            decoded=xr.merge(m.decode(*latents,co2=float(record["co2_ppm"][index])))
            for v in decoded:
                x,y=decoded[v].values,saved[v].values
                rows.append(dict(dataset=k,month=month,check="decoder_replay",variable=v,
                    max_abs_error=float(np.nanmax(abs(x-y))),rms_error=float(np.sqrt(np.nanmean((x-y)**2))),
                    mask_identical=np.array_equal(np.isfinite(x),np.isfinite(y)),
                    rms_error_over_field_std=float(np.sqrt(np.nanmean((x-y)**2))/max(np.nanstd(y),1e-10))))
            saved.close()
        print(k,"saved transition + decoder replay complete",flush=True)
    pd.DataFrame(rows).to_csv(output/"archive_replay.csv",index=False)
    m=models["era"];archive=ROLLOUTS["era"]
    record=json.loads((archive/"forecast.json").read_text())
    dates=np.array(record["valid_times"],dtype="datetime64[ns]")
    fixed_ppm=float(record["co2_ppm"][np.flatnonzero(dates.astype("datetime64[M]")==np.datetime64("1900-01","M"))[0]])
    fixed_path=output/"spread_fixed_co2_decodes.nc"
    names=[v.name for v in m.config.variables if v.component=="ocean"]
    if not fixed_path.exists():
        selected=np.flatnonzero((dates>=np.datetime64("1888-01"))&(dates<np.datetime64("1909-01")))
        pieces=[]
        for offset in range(0,len(selected),6):
            block=selected[offset:offset+6]
            z=np.stack([torch.load(archive/f"month-{i+1:06d}.zarr/forecast_state.pt",map_location="cpu",weights_only=False)["history"][1][:,0].numpy() for i in block])
            latent=xr.DataArray(z,dims=("time","member","node","channel"),coords=dict(time=dates[block],member=range(record["members"])))
            _,decoded=m.decode(ocean=latent,co2=fixed_ppm)
            pieces.append(xr.Dataset({v:weighted_mean(decoded[v]) for v in names}))
            if offset%60==0:print("fixed-CO2 decode",offset,"/",len(selected),flush=True)
        xr.concat(pieces,"time").assign_attrs(co2_ppm=fixed_ppm,definition="archived ocean latents decoded at constant CO2, calendar unchanged").to_netcdf(fixed_path)
    fixed=xr.load_dataset(fixed_path)
    trace_path=output/"ocean_latent_trace_1850_1950.nc"
    if not trace_path.exists():
        selected=np.flatnonzero(dates<np.datetime64("1951-01"))
        values=[]
        for i in selected:
            s=torch.load(archive/f"month-{i+1:06d}.zarr/forecast_state.pt",map_location="cpu",weights_only=False)
            assert np.datetime64(s["times"][0],"M")==dates[i].astype("datetime64[M]")
            values.append(s["history"][1][:,0].numpy())
        z=np.stack(values)
        xr.Dataset(dict(latent=(("time","member","node","channel"),z)),
                   coords=dict(time=dates[selected],member=range(record["members"]))).to_netcdf(trace_path)
    z=xr.load_dataset(trace_path).latent
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    compare=[]
    for ax,v in zip(axes.flat[:4],["sst","mld","ssh","ohc300"]):
        raw=xr.load_dataset(DIAGNOSTICS["era"]/f"model/{v}.nc").global_mean
        raw,unit=display(v,raw);counter,_=display(v,fixed[v])
        annual=raw.groupby("time.year").mean("time").sel(year=slice(1888,1908))
        constant=counter.groupby("time.year").mean("time")
        for member in annual.member.values:
            ax.plot(annual.year,annual.sel(member=member),color=f"C{member}",label=f"Member {member}")
            ax.plot(constant.year,constant.sel(member=member),color=f"C{member}",ls="--",lw=1)
        ax.set(title=v,ylabel=unit)
        for kind,value in (("original",annual),("fixed_co2",constant)):
            for lo,hi in [(1888,1895),(1900,1908)]:
                period=value.sel(year=slice(lo,hi))
                compare.append(dict(variable=v,kind=kind,period=f"{lo}-{hi}",
                    mean_member_range=float((period.max("member")-period.min("member")).mean()),
                    mean_member_std=float(period.std("member").mean())))
    center=z.mean(("node","channel")).groupby("time.year").mean("time")
    for member in center.member.values:
        axes[2,0].plot(center.year,center.sel(member=member),color=f"C{member}")
    axes[2,0].set(title="Ocean latent mean · normalized coordinates",xlabel="Year",ylabel="Mean latent")
    window=(dates>=np.datetime64("1888-01"))&(dates<np.datetime64("1909-01"))
    axes[2,1].plot(dates[window],np.array(record["co2_ppm"])[window],color="black")
    axes[2,1].set(title="Prescribed CO₂ · no discontinuity near 1900",ylabel="ppm")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    axes[0,0].legend(ncol=3,frameon=False,fontsize=8)
    fig.suptitle(f"ERA5/ORAS5 rollout: spread cliff audit\nSolid: archived forcing; dashed: SAME latents decoded at constant {fixed_ppm:.2f} ppm",fontsize=12)
    finish(fig,output/"spread_cliff_1900")
    pd.DataFrame(compare).to_csv(output/"spread_cliff_summary.csv",index=False)


def support_audit(models, output):
    """Distinguish field-specific validity from the shared SST message mask."""
    rows=[]
    for k,m in models.items():
        wet=np.asarray(m.preprocessor.stats["sst"]["mask"])
        weights=np.broadcast_to(np.cos(np.deg2rad(m.statics.lat.values))[:,None],wet.shape)
        # Two real encoded states, decoded in identical January/CO2 context.
        # This probes whether finite non-SST outputs can receive latent signal
        # outside the common SST support, without fitting or changing anything.
        cache=torch.load(RUNS[k]/"latents.pt",map_location="cpu",weights_only=False)
        assert cache["ae_digest"]==m.weight_digest(autoencoder_only=True)
        latent=xr.DataArray(cache["training"]["ocean"][[0,120]].numpy()[None],
            dims=("time","member","node","channel"),
            coords=dict(time=[np.datetime64("1966-01-01")],member=[0,1]))
        _,decoded=m.decode(ocean=latent,co2=m.preprocessor.co2_mean)
        for spec in m.config.variables:
            if spec.component!="ocean":continue
            mask=np.asarray(m.preprocessor.stats[spec.name]["mask"])
            excluded=mask & ~wet
            difference=(decoded[spec.name].isel(member=1)-decoded[spec.name].isel(member=0)).values[0]
            outside_response=float(np.max(abs(difference[excluded]))) if excluded.any() else np.nan
            rows.append(dict(dataset=k,variable=spec.name,valid_cells=int(mask.sum()),
                valid_outside_sst_mask=int(excluded.sum()),
                fraction_variable_area_outside_sst_mask=float(weights[excluded].sum()/weights[mask].sum()),
                valid_on_lsm_majority_land=int((mask&(m.statics.lsm.values>.5)).sum()),
                max_abs_latent_response_outside_sst=outside_response,
                outside_response_over_physical_field_std=outside_response/max(float(decoded[spec.name].std()),1.e-12)))
    pd.DataFrame(rows).to_csv(output/"ocean_support.csv",index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"analysis/era5_e3sm_comparability")
    p.add_argument("--stage",choices=("all","data","coast","architecture","spread","support"),default="all")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    models={k:MonthlyDiffusionV1.load(v/"model.pt",device="cpu") for k,v in RUNS.items()}
    sources={k:yaml.safe_load((v/"data_sources.yaml").read_text()) for k,v in RUNS.items()}
    fields={}
    for k,s in sources.items():
        a,o,_=open_canonical_state(s["state_zarr"],expected_attrs=s.get("expected_attrs"))
        fields[k]=xr.merge((a,o),compat="no_conflicts")
    identity=dict(checkpoints={k:m.weight_digest() for k,m in models.items()},sources=sources,version=1)
    marker=args.output/"configuration.json"
    if marker.exists() and json.loads(marker.read_text())!=identity:raise ValueError("Audit identity changed; use a new output directory")
    marker.write_text(json.dumps(identity,indent=2))
    if args.stage in ("all","architecture"):architecture(models,sources,args.output)
    if args.stage in ("all","data"):distributions(models,fields,args.output)
    if args.stage in ("all","coast"):coastal_sst(models,fields,sources,args.output)
    if args.stage in ("all","spread"):spread_and_replay(models,args.output)
    if args.stage in ("all","support"):support_audit(models,args.output)
    print("Saved audit:",args.output,flush=True)


if __name__=="__main__":main()
