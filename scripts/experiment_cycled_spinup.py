"""Paired, resumable test of temporary AE cycling as climate initialization.

Four arms: encoded-noise / 30-month cycling then latent-only / always cycling /
observed encoded Nov-Dec 1958 histories. All use matched monthly innovations,
unchanged AR1 noise, fixed 1850 CO2 for decoding/cycling, and identical calendars.
Observation histories retain their actual-CO2 encoding from the checkpoint's
validation cache; they are coherent modern states, not preindustrial truth.
No retraining. Annual files save EVERY predicted/fed-back latent, physical
Nino3.4 and Tahiti-Darwin pressure indices, and full forecast restart states.
No physical maps are archived. All-field physical decode/re-encode remains
identical to the pilot; only the current history slot is replaced.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import torch
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.training.common import atomic_save
from monthly_diffusion.analysis.spectra import lag_autocorrelation, temporal_spectrum
from diagnose_cycled_coupling import physical_indices, fixed_anomalies, pair_metrics, lag_curve
from diagnose_latent_attractor import anomalies

COMPONENTS=("atmosphere","ocean")
METHODS=("latent_only","cycle30","continuous","observed")
LABELS={"latent_only":"No cycling","cycle30":"Cycle 30 months\nthen latent-only",
        "continuous":"Continuous cycling","observed":"Observed initialization"}
COLORS={"latent_only":"#0072B2","cycle30":"#D55E00","continuous":"#CC79A7","observed":"#009E73"}


@torch.inference_mode()
def run_experiment(model, cache, output, *, years, members, seed, first, ppm, cycle_months=30):
    forecasts={"latent_only":model.initialize(co2=ppm,members=members,seed=seed,encoded_noise_start=first)}
    initial=copy.deepcopy(forecasts["latent_only"].state_dict())
    for method in METHODS[1:]:
        f=model.initialize(co2=ppm,members=members,seed=seed,gaussian_start=first)
        f.load_state_dict(copy.deepcopy(initial))
        forecasts[method]=f
    source_dates=np.asarray(cache["validation_time"]).astype("datetime64[M]")
    indices=[int(np.flatnonzero(source_dates==np.datetime64(d,"M"))[0]) for d in ("1958-12","1958-11")]
    for ci,c in enumerate(COMPONENTS):
        pair=torch.as_tensor(cache["validation"][c][indices],dtype=torch.float32,device=model.device)
        forecasts["observed"].history[ci]=pair[None].expand(members,-1,-1,-1).clone()
    if pd.Timestamp(first).month!=1:
        raise ValueError("This experiment's observed Nov/Dec histories require a January forecast start")
    shards=sorted(output.glob("year-*.pt"))
    if [p.name for p in shards]!=[f"year-{i:03d}.pt" for i in range(1,len(shards)+1)]:
        raise ValueError("Missing annual block")
    if shards:
        saved=torch.load(shards[-1],map_location="cpu",weights_only=False)
        for method in METHODS:
            forecasts[method].load_state_dict(saved["states"][method])
            if forecasts[method].step!=12*len(shards):
                raise ValueError("Restart step/date count mismatch")
    else:
        atomic_save({m:f.state_dict() for m,f in forecasts.items()},output/"initial_states.pt")
    start=time.monotonic()
    for year in range(len(shards)+1,years+1):
        predicted={c:[] for c in COMPONENTS}; fed_back={c:[] for c in COMPONENTS}
        physical,corrections,dates=[],[],[]
        for month in range(12):
            step=(year-1)*12+month+1
            monthly={}
            for method in METHODS:
                f=forecasts[method]
                if method=="continuous" and step<=cycle_months:
                    # These arms are the SAME treatment until switch-off.
                    # Clone the actual result instead of introducing GPU
                    # roundoff differences by recomputing identical encodings.
                    f.load_state_dict(copy.deepcopy(forecasts["cycle30"].state_dict()))
                    monthly[method]=monthly["cycle30"]
                    continue
                previous=[h[:,0].clone() for h in f.history]
                za,zo=next(f)
                for h,before in zip(f.history,previous):
                    torch.testing.assert_close(h[:,1],before,rtol=0,atol=0)
                before=[h[:,0].detach().cpu().clone() for h in f.history]
                if not all(torch.isfinite(z).all() for z in before):
                    raise FloatingPointError("Nonfinite prediction")
                a,o=model.decode(za,zo,co2=ppm)
                idx=physical_indices(a,o)
                if not bool(np.isfinite(idx.to_array()).all()):
                    raise FloatingPointError("Nonfinite physical indices")
                if method=="continuous" or (method=="cycle30" and step<=cycle_months):
                    encoded=[[],[]]
                    for member in range(members):
                        zs=model.encode(a.isel(member=member,drop=True),o.isel(member=member,drop=True),co2=ppm)
                        for ci,z in enumerate(zs):
                            encoded[ci].append(z.values[0])
                    for ci in range(2):
                        z=torch.as_tensor(np.stack(encoded[ci]),dtype=torch.float32,device=model.device)
                        if not torch.isfinite(z).all():
                            raise FloatingPointError("Nonfinite cycle")
                        f.history[ci][:,0]=z
                for h,old in zip(f.history,previous):
                    torch.testing.assert_close(h[:,1],old,rtol=0,atol=0)
                after=[h[:,0].detach().cpu().clone() for h in f.history]
                monthly[method]=dict(predicted=before,fed_back=after,
                    physical=np.column_stack((idx.nino34.values[0],idx.pressure_difference.values[0])),
                    correction=np.stack([((after[i]-before[i])**2).mean((1,2)).sqrt().numpy() for i in range(2)],axis=-1))
                del a,o,idx
            expected=(np.datetime64(first,"M")+np.timedelta64(step-1,"M")).astype("datetime64[ns]")
            for method,f in forecasts.items():
                if f.times[0]!=expected:
                    raise ValueError("Skipped/misordered forecast date")
                for n,base in zip(f.noise,forecasts["latent_only"].noise):
                    torch.testing.assert_close(n,base,rtol=0,atol=0)
            for ci,c in enumerate(COMPONENTS):
                predicted[c].append(np.stack([monthly[m]["predicted"][ci].numpy() for m in METHODS]))
                fed_back[c].append(np.stack([monthly[m]["fed_back"][ci].numpy() for m in METHODS]))
            physical.append(np.stack([monthly[m]["physical"] for m in METHODS]))
            corrections.append(np.stack([monthly[m]["correction"] for m in METHODS]))
            dates.append(expected)
        atomic_save(dict(time=np.asarray(dates),methods=METHODS,
            predicted={c:np.stack(v) for c,v in predicted.items()},
            fed_back={c:np.stack(v) for c,v in fed_back.items()},
            physical=np.stack(physical),correction=np.stack(corrections),
            states={m:f.state_dict() for m,f in forecasts.items()}), output/f"year-{year:03d}.pt")
        elapsed=time.monotonic()-start
        per_year=elapsed/(year-len(shards))
        print(f"Year {year}/{years} | {per_year:.1f}s/year | estimated remaining {(years-year)*per_year/60:.1f}min",flush=True)


def index_statistics(pair,reference):
    result=pair_metrics(pair,reference)
    for j,c in enumerate(("ocean","atmosphere")):
        x=xr.DataArray(pair[:,j],dims="time")
        result[c+"_acf24"]=float(lag_autocorrelation(x,max_lag=24,detrend="constant").sel(lag_months=24))
        result[c+"_mean_reference_sd"]=float(pair[:,j].mean()/reference[:,j].std())
    return result


def analyze(output, cache, probes, reference_path, cycle_months):
    run_metadata=json.loads((output/"configuration.json").read_text())
    reference_metadata=json.loads((reference_path.parent/"configuration.json").read_text())
    for key in ("weight_digest","co2_ppm"):
        if run_metadata[key]!=reference_metadata[key]:
            raise ValueError(f"Decoder reference differs in {key}")
    shards=[torch.load(p,map_location="cpu",weights_only=False) for p in sorted(output.glob("year-*.pt"))]
    dates=np.concatenate([s["time"] for s in shards])
    if len(dates)<360:
        raise ValueError("At least 30 years required for the prespecified late-period comparison")
    train_dates=np.asarray(cache["training_time"])
    times=pd.DatetimeIndex(dates).month.values
    trmonths=pd.DatetimeIndex(train_dates).month.values
    latent,feedback,reference=[],[],[]
    for c in ("ocean","atmosphere"):
        z=np.concatenate([s["predicted"][c] for s in shards]).astype(float)
        post=np.concatenate([s["fed_back"][c] for s in shards]).astype(float)
        training=cache["training"][c].numpy().astype(float)
        climate=np.stack([training[trmonths==i].mean(0) for i in range(1,13)])
        coef=probes[c]
        latent.append((z-climate[times-1,None,None]).reshape(*z.shape[:3],-1)@coef)
        feedback.append((post-climate[times-1,None,None]).reshape(*z.shape[:3],-1)@coef)
        reference.append((training-climate[trmonths-1]).reshape(len(training),-1)@coef)
    latent=np.stack(latent,axis=-1); feedback=np.stack(feedback,axis=-1)
    latent_ref=np.stack(reference,axis=-1)
    phys=np.concatenate([s["physical"] for s in shards]); phys[...,1]*=-1
    r=xr.load_dataset(reference_path)
    np.testing.assert_array_equal(r.time.values,train_dates)
    physical_raw=np.column_stack((r.nino34.values,-r.pressure_difference.values))
    physical_ref=physical_raw.copy()
    for i in range(1,13):
        mean=physical_raw[trmonths==i].mean(0)
        phys[times==i]-=mean
        physical_ref[trmonths==i]-=mean
    members=latent.shape[2]
    ds=xr.Dataset(dict(latent=(("time","method","member","component"),latent),
        fed_back_latent=(("time","method","member","component"),feedback),
        physical=(("time","method","member","component"),phys),
        latent_reference=(("training_time","component"),latent_ref),
        physical_reference=(("training_time","component"),physical_ref)),
        coords=dict(time=dates,training_time=train_dates,method=list(METHODS),member=np.arange(members),
                    component=["ocean","atmosphere"]),
        attrs=dict(anomalies="Fixed training monthly means; no detrend or ensemble mean removal",
                   physical_units="ocean: degC; atmosphere: minus Tahiti-Darwin pressure difference, Pa",
                   latent_units="degrees C-equivalent fixed ridge readouts",cycle_months=cycle_months))
    ds.to_netcdf(output/"index_trajectories.nc",encoding={v:dict(zlib=True,complevel=3) for v in ds})
    rows,refrows,curverows,sensitivity_rows=[],[],[],[]
    windows={"years01_05":(0,60),"years06_15":(60,180),"years11_30":(120,360)}
    windows.update({f"block{i//12+1:02d}_{(i+60)//12:02d}":(i,i+60) for i in range(0,360,60)})
    for kind,values,ref in (("latent",latent,latent_ref),("physical",phys,physical_ref),
                            ("fed_back_latent",feedback,latent_ref)):
        clean_ref=anomalies(ref,train_dates,"window_monthly_detrend")
        for window,(lo,hi) in windows.items():
            for mi,method in enumerate(METHODS):
                for member in range(members):
                    rows.append(dict(kind=kind,window=window,method=method,member=member+1,
                                     **index_statistics(values[lo:hi,mi,member],ref)))
                    if window in ("years06_15","years11_30"):
                        clean=anomalies(values[lo:hi,mi,member],dates[lo:hi],"window_monthly_detrend")
                        sensitivity_rows.append(dict(kind=kind,window=window,method=method,member=member+1,
                            **index_statistics(clean,clean_ref)))
        for start in range(0,len(ref)-239,12):
            refrows.append(dict(kind=kind,start=str(train_dates[start]),**index_statistics(ref[start:start+240],ref)))
        for mi,method in enumerate(METHODS):
            for member in range(members):
                curve=lag_curve(values[120:360,mi,member],maxlag=12)
                curverows.extend(dict(kind=kind,method=method,member=member+1,lag=lag,correlation=v)
                                 for lag,v in zip(range(-12,13),curve))
    table=pd.DataFrame(rows); ref_table=pd.DataFrame(refrows)
    table.to_csv(output/"member_metrics.csv",index=False)
    ref_table.to_csv(output/"reference_20year_windows.csv",index=False)
    pd.DataFrame(curverows).to_csv(output/"lead_lag_correlations.csv",index=False)
    pd.DataFrame(sensitivity_rows).to_csv(output/"window_climatology_detrend_sensitivity.csv",index=False)
    summary=table.groupby(["kind","window","method"]).mean(numeric_only=True).drop(columns="member").reset_index()
    summary.to_csv(output/"summary.csv",index=False)
    # Paired differences are descriptive member-by-member contrasts, not
    # independent-month significance tests.
    paired=[]
    for kind in ("latent","physical"):
        late=table[(table.kind==kind)&(table.window=="years11_30")]
        base=late[late.method=="latent_only"].set_index("member")
        for method in METHODS[1:]:
            group=late[late.method==method].set_index("member")
            for member in group.index:
                paired.append(dict(kind=kind,method=method,member=member,
                    **{metric:float(group.loc[member,metric]-base.loc[member,metric]) for metric in
                       ("correlation","relationship_rmse_ratio","ocean_sd_ratio","ocean_acf24")}))
    pd.DataFrame(paired).to_csv(output/"paired_differences.csv",index=False)
    (output/"analysis_configuration.json").write_text(json.dumps(dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference=str(reference_path.resolve()),primary_window="simulation years 11–30",
        primary_anomalies="Fixed training monthly means; preserve offsets and trends; no ensemble mean subtraction",
        sensitivity="Independently regress each member/window and the training reference on 12 month indicators plus linear time; no ensemble mean subtraction",
        uncertainty="Member ranges and overlapping reference-window percentiles are descriptive, not confidence intervals",
        correlation="Positive physical coupling means Niño3.4 correlated with NEGATIVE Tahiti-minus-Darwin pressure proxy",
        validation="Monthly terminal-noise equality, current/previous order, month continuity, and reference checkpoint/CO2/date matches checked"),indent=2))
    for kind,values,ref in (("latent",latent,latent_ref),("physical",phys,physical_ref)):
        fig,axes=plt.subplots(members,len(METHODS),figsize=(16,2*members+1),sharex=True,sharey=True,squeeze=False)
        for mi,method in enumerate(METHODS):
            for member in range(members):
                ax=axes[member,mi]
                x=values[:,mi,member]/ref.std(0)
                ax.plot(np.arange(len(x))/12,x[:,0],color=COLORS[method],lw=.9)
                ax.plot(np.arange(len(x))/12,x[:,1],color=".2",ls="--",lw=.7,alpha=.75)
                ax.axhspan(-1,1,color=".5",alpha=.10); ax.axhline(0,color=".5",lw=.5)
                ax.axvline(cycle_months/12,color=".4",ls=":",lw=.8)
                if member==0: ax.set_title(LABELS[method],fontsize=10)
                if mi==0: ax.set_ylabel(f"Member {member+1}\nReference SD")
                if member==members-1: ax.set_xlabel("Simulation years")
                ax.grid(alpha=.12)
        fig.suptitle(f"{'Latent ENSO' if kind=='latent' else 'Physical Niño3.4 / −SOI'} · solid ocean, dashed atmosphere · fixed training-month anomalies")
        fig.tight_layout(rect=(0,0,1,.95))
        for ext in ("png","pdf"): fig.savefig(output/f"{kind}_all_trajectories.{ext}",dpi=180)
        plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        for ci,component in enumerate(("ocean","atmosphere")):
            refx=ref[:,ci]/ref[:,ci].std()
            support=np.linspace(min(-4.,values[...,ci].min()/ref[:,ci].std()),
                                max(4.,values[...,ci].max()/ref[:,ci].std()),300)
            axes[ci,0].plot(support,gaussian_kde(refx)(support),color="black",label="Encoded reference")
            ref_da=xr.DataArray(refx,dims="time")
            acf=lag_autocorrelation(ref_da,max_lag=60,detrend="constant")
            axes[ci,1].plot(np.arange(61)/12,acf,color="black")
            psd=temporal_spectrum(ref_da,nperseg=120)
            freq=psd.frequency.values; period=1/freq[1:]/12
            axes[ci,2].loglog(period,psd.values[1:],color="black")
            for mi,method in enumerate(METHODS):
                x=values[120:360,mi,:,ci]/ref[:,ci].std()
                # Pool only for the marginal PDF. Time statistics NEVER join members.
                axes[ci,0].plot(support,gaussian_kde(x.ravel())(support),color=COLORS[method],label=LABELS[method].replace("\n"," "))
                da=xr.DataArray(x.T,dims=("member","time"))
                acf=lag_autocorrelation(da,max_lag=60,detrend="constant")
                axes[ci,1].plot(np.arange(61)/12,acf.mean("member"),color=COLORS[method])
                axes[ci,1].fill_between(np.arange(61)/12,acf.min("member"),acf.max("member"),color=COLORS[method],alpha=.08)
                psd=temporal_spectrum(da,nperseg=120).mean("member")
                axes[ci,2].loglog(period,psd.values[1:],color=COLORS[method])
            axes[ci,0].set(xlabel="Fixed-reference SD (offsets retained)",ylabel=f"{component.title()} density")
            axes[ci,1].set(xlabel="Lag (years)",ylabel="ACF",ylim=(-.6,1)); axes[ci,1].axhline(0,color=".5",lw=.6)
            axes[ci,2].set(xlabel="Period (years)",ylabel="PSD (reference SD² × month)",xlim=(.2,10))
            for ax in axes[ci]: ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=7,frameon=False)
        fig.suptitle(f"{kind.title()} ENSO dynamics · years 11–30 · five-member mean ACF/PSD; ACF shading is member range")
        for ext in ("png","pdf"): fig.savefig(output/f"{kind}_dynamics.{ext}",dpi=190)
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for i,kind in enumerate(("latent","physical")):
        r=latent_ref if kind=="latent" else physical_ref
        baseline=index_statistics(r,r)
        for j,metric in enumerate(("correlation","relationship_rmse_ratio","ocean_sd_ratio")):
            ax=axes[i,j]
            rr=ref_table[ref_table.kind==kind][metric]
            ax.axhspan(rr.quantile(.05),rr.quantile(.95),color=".5",alpha=.15)
            ax.axhline(baseline[metric],color="black",lw=.9)
            for mi,method in enumerate(METHODS):
                v=table[(table.kind==kind)&(table.window=="years11_30")&(table.method==method)][metric]
                ax.scatter(mi+np.linspace(-.12,.12,len(v)),v,color=COLORS[method],s=20)
                ax.plot([mi-.18,mi+.18],[v.mean(),v.mean()],color=COLORS[method],lw=3)
            ax.set_xticks(range(4),["No\ncycling","Cycle first\n30 months","Continuous\ncycling","Observed\ninitialization"],fontsize=8)
            ax.set_ylabel(f"{kind.title()} · {metric.replace('_',' ')}"); ax.grid(axis="y",alpha=.15)
    fig.suptitle("Years 11–30 · dots = members; bars = member means\nBlack = full encoded reference; gray = matched 20-year reference-window range (5–95%)")
    for ext in ("png","pdf"): fig.savefig(output/f"coupling_summary.{ext}",dpi=190)
    plt.close(fig)
    print(summary[(summary.window=="years11_30")&(summary.kind!="fed_back_latent")][
        ["kind","method","correlation","relationship_rmse_ratio","ocean_sd_ratio","ocean_acf24","ocean_mean_reference_sd"]].to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"analysis/cycled_initialization_ensemble_30years")
    p.add_argument("--pilot",type=Path,default=ROOT/"analysis/cycled_noise_initialization_era5_30months")
    p.add_argument("--readouts",type=Path,default=ROOT/"analysis/encoded_vs_direct_noise_enso_rollouts")
    p.add_argument("--years",type=int,default=30)
    p.add_argument("--members",type=int,default=5)
    p.add_argument("--seed",type=int,default=20260716)
    p.add_argument("--device",default="mps")
    p.add_argument("--no-analysis",action="store_true")
    p.add_argument("--analysis-only",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(4)
    source=json.loads((args.pilot/"configuration.json").read_text())
    probe_config=json.loads((args.readouts/"configuration.json").read_text())
    if source["weight_digest"]!=probe_config["checkpoint_sha256"]:
        raise ValueError("Probe checkpoint mismatch")
    run=Path(source["run"])
    cache=torch.load(run/"latents.pt",map_location="cpu",weights_only=False)
    record=dict(checkpoint=str(run/"model.pt"),weight_digest=source["weight_digest"],ae_digest=cache["ae_digest"],
        years=args.years,members=args.members,seed=args.seed,device=args.device,first="1800-01",cycle_months=30,
        co2_ppm=source["co2_ppm"],methods=list(METHODS),history_order="current, previous",
        observation_source="Cached actual-CO2 encoded November/December 1958, same checkpoint; months aligned, simulation years relabeled",
        pairing="Identical monthly terminal innovations and AR1 memory across treatments; cycle30/continuous exactly identical until month30",
        storage="All monthly predicted and fed-back latents, physical scalar indices, annual restart states; NO full decoded fields",
        scope="Temporary cycling changes only initialization; continuous cycling changes effective model dynamics",
        anomalies="Fixed training monthly baselines; primary diagnostics retain level shifts; no linear detrending or ensemble-mean subtraction")
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"configuration.json"
    if path.exists() and json.loads(path.read_text())!=record:
        raise ValueError("Configuration changed; choose a new output directory")
    path.write_text(json.dumps(record,indent=2))
    if not args.analysis_only:
        if args.device=="mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no CPU fallback")
        model=MonthlyDiffusionV1.load(run/"model.pt",device=args.device)
        if model.weight_digest()!=source["weight_digest"] or model.weight_digest(autoencoder_only=True)!=cache["ae_digest"]:
            raise ValueError("Checkpoint changed")
        run_experiment(model,cache,args.output,years=args.years,members=args.members,seed=args.seed,
                       first="1800-01",ppm=source["co2_ppm"])
        del model
    if not args.no_analysis:
        probes={c:np.load(args.readouts/f"{c}_readout.npz")["coef"] for c in COMPONENTS}
        analyze(args.output,cache,probes,args.pilot/"coupling/encoded_reference_indices.nc",30)


if __name__=="__main__":
    main()
