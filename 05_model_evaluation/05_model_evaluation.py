"""Stage 05: scores, uncertainty, coverage, interpretation and paper figures.

Default evaluation does not retrain models. Generated tables/figures are local
outputs; this source file contains no archived results. Figure numbering and
the full-year example follow the submitted Final Revised Manuscript.
"""
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, re, sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_selection import mutual_info_regression

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

def load_stage(number):
    """Load a numbered sibling program without relying on the current directory."""
    paths = {
        2: "02_data_preprocessing/02_data_preprocessing.py",
        3: "03_build_prediction_dataset/03_build_prediction_dataset.py",
        4: "04_train_cnn_bilstm/04_train_cnn_bilstm.py",
        5: "05_model_evaluation/05_model_evaluation.py",
    }
    name = f"_forecast_gic_stage_{number:02d}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / paths[number])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]

dataset = load_stage(3)
ROOT = REPOSITORY_ROOT
DATA = ROOT / "data"
OUT = ROOT / "evaluation"
FIGURES = ROOT / "figures"
THRESHOLDS = (3, 5, 10, 20)
FEATURES = dataset.FEATURES
future_peak = dataset.future_peak
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def configure_workspace(workspace=None):
    global ROOT, DATA, OUT, FIGURES
    ROOT = Path(workspace).resolve() if workspace is not None else REPOSITORY_ROOT
    DATA = ROOT / "data"
    OUT, FIGURES = ROOT / "evaluation", ROOT / "figures"
    dataset.configure_workspace(ROOT)

def ordered_probabilities(prob):
    """Exact Euclidean projection onto p3 >= p5 >= p10 >= p20.

    Enumerate the eight contiguous partitions of four ordered outputs.
    This is equivalent to four-point PAV, vectorized over all rows.
    """
    p = np.asarray(prob,dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all():
        raise ValueError("Expected finite N-by-4 probabilities")
    best = np.empty_like(p)
    cost = np.full(len(p),np.inf)
    for mask in range(8):
        cuts = [0] + [j+1 for j in range(3) if mask & (1<<j)] + [4]
        q = np.empty_like(p)
        for left,right in zip(cuts[:-1],cuts[1:]):
            q[:,left:right] = p[:,left:right].mean(axis=1,keepdims=True)
        feasible = (np.diff(q,axis=1)<=1e-12).all(axis=1)
        c = ((p-q)**2).sum(axis=1)
        use = feasible & (c<cost)
        best[use] = q[use]
        cost[use] = c[use]
    assert np.isfinite(cost).all()
    return np.clip(best,0,1)


def fit_calibration(logits,y):
    params=[]
    for j in range(4):
        z=np.asarray(logits[:,j],float)
        target=np.asarray(y[:,j],float)
        if np.unique(target).size < 2:
            # Intercept-only smoothed prevalence, explicitly recorded.
            prevalence=(target.sum()+.5)/(len(target)+1)
            params.append(dict(a=0.,b=float(np.log(prevalence/(1-prevalence))),single_class=True))
            continue
        def objective(ab):
            a,b=np.exp(ab[0]),ab[1]
            q=a*z+b
            loss=np.mean(np.logaddexp(0,q)-target*q)+1e-5*np.dot(ab,ab)
            error=expit(q)-target
            grad=np.array([np.mean(error*z)*a,np.mean(error)])+2e-5*ab
            return loss,grad
        result=minimize(objective,np.zeros(2),jac=True,method="L-BFGS-B",bounds=[(-4,4),(-12,12)])
        if not result.success: raise RuntimeError(f"Calibration failed: {result.message}")
        params.append(dict(a=float(np.exp(result.x[0])),b=float(result.x[1]),single_class=False))
    return params


def calibrated_probabilities(logits,params):
    result=np.column_stack([expit(params[j]["a"]*logits[:,j]+params[j]["b"]) for j in range(4)])
    return ordered_probabilities(result)


def binary_metrics(y,p,cutoff=.5):
    y=np.asarray(y,bool); pred=np.asarray(p)>=cutoff
    tp=int((y&pred).sum()); fp=int((~y&pred).sum())
    fn=int((y&~pred).sum()); tn=int((~y&~pred).sum())
    ratio=lambda a,b: float(a/b) if b else np.nan
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,POD=ratio(tp,tp+fn),POFD=ratio(fp,fp+tn),
                FAR=ratio(fp,tp+fp),CSI=ratio(tp,tp+fp+fn),F1=ratio(2*tp,2*tp+fp+fn),
                Bias=ratio(tp+fp,tp+fn),cutoff=float(cutoff))


def score(y,p,cutoffs=None,climatology=None):
    rows=[]
    for j,t in enumerate(THRESHOLDS):
        cutoff=.5 if cutoffs is None else cutoffs[j]
        r=binary_metrics(y[:,j],p[:,j],cutoff)
        r.update(threshold_A=t,N=len(y),prevalence=float(y[:,j].mean()),
                 Brier=float(np.mean((p[:,j]-y[:,j])**2)),
                 ROC_AUC=float(roc_auc_score(y[:,j],p[:,j])) if np.unique(y[:,j]).size>1 else np.nan,
                 PR_AUC=float(average_precision_score(y[:,j],p[:,j])) if y[:,j].sum()>0 else np.nan)
        if climatology is not None:
            ref=np.mean((climatology[j]-y[:,j])**2)
            r["Brier_skill"]=1-r["Brier"]/ref if ref else np.nan
            r["reference_prevalence"]=float(climatology[j])
        bins=reliability(y[:,j],p[:,j])
        r["ECE"]=sum(b["count"]*abs(b["forecast_mean"]-b["observed_frequency"]) for b in bins if b["count"])/len(y)
        rows.append(r)
    return rows


def reliability(y,p):
    idx=np.minimum((np.asarray(p)*10).astype(int),9)
    rows=[]
    for b in range(10):
        use=idx==b; n=int(use.sum())
        rows.append(dict(bin=b,lower=b/10,upper=(b+1)/10,count=n,
                         forecast_mean=float(p[use].mean()) if n else np.nan,
                         observed_frequency=float(y[use].mean()) if n else np.nan))
    return rows


def select_cutoffs(y,p):
    cutoffs=[]
    for j in range(4):
        candidates=np.unique(np.r_[.5,np.quantile(p[:,j],np.linspace(0,1,201))])
        scored=[(binary_metrics(y[:,j],p[:,j],c)["CSI"],float(c)) for c in candidates]
        scored=[pair for pair in scored if np.isfinite(pair[0])]
        cutoffs.append(max(scored)[1] if scored else .5)
    return cutoffs


def causal_confirm(mask, times, groups, required=1):
    """Optional prefix-invariant confirmation; never backfill prior alarms."""
    if required < 1: raise ValueError("required must be positive")
    out=np.zeros(len(mask),bool); run=0
    times=np.asarray(times,dtype="datetime64[ns]").astype(np.int64)
    for i,value in enumerate(mask):
        if i and (times[i]-times[i-1]!=60_000_000_000 or groups[i]!=groups[i-1]): run=0
        run=run+1 if value else 0
        out[i]=run>=required
    return out


def segments(mask,times,groups):
    times=np.asarray(times,dtype="datetime64[ns]").astype(np.int64)
    mask=np.asarray(mask,bool)
    contiguous=np.r_[False,(np.diff(times)==60_000_000_000)&(np.asarray(groups)[1:]==np.asarray(groups)[:-1])]
    starts=np.flatnonzero(mask & ~(np.r_[False,mask[:-1]] & contiguous))
    ends=np.flatnonzero(mask & ~(np.r_[mask[1:],False] & np.r_[contiguous[1:],False]))
    return list(zip(starts,ends))


def episode_metrics(truth,prediction,times,groups):
    """Chronological one-to-one overlap matching, identical in all splits."""
    true=segments(truth,times,groups); predicted=segments(prediction,times,groups)
    used=set(); matches=[]
    for pi,(a,b) in enumerate(predicted):
        candidates=[j for j,(c,d) in enumerate(true) if j not in used and a<=d and b>=c]
        if candidates:
            j=candidates[0]; used.add(j); matches.append((pi,j))
    tp=len(matches);fp=len(predicted)-tp;fn=len(true)-tp
    return dict(TP=tp,FP=fp,FN=fn,CSI=tp/(tp+fp+fn) if tp+fp+fn else np.nan,
                FAR=fp/(tp+fp) if tp+fp else np.nan,matches=matches)


def load(run,split):
    z=np.load(run/f"{split}.npz")
    return {k:z[k] for k in z.files}


def paired_bootstrap(a,b,replicates=2000,seed=617):
    common,ia,ib=np.intersect1d(a["time"],b["time"],return_indices=True)
    if not len(common):raise ValueError("No common forecast issue times")
    ya,yb=a["y"][ia],b["y"][ib]
    np.testing.assert_array_equal(ya,yb)
    ga,gb=a["group"][ia],b["group"][ib]
    np.testing.assert_array_equal(ga,gb)
    units=np.unique(ga);rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(units),(replicates,len(units)))
    rows=[]
    for j,t in enumerate(THRESHOLDS):
        counts=[]
        for p in [a["p"][ia,j],b["p"][ib,j]]:
            one=[]
            for group in units:
                use=ga==group;m=binary_metrics(ya[use,j],p[use])
                one.append([m["TP"],m["FP"],m["FN"]])
            counts.append(np.asarray(one))
        ca,cb=counts
        for metric in ["CSI","F1"]:
            def ratio(c):
                tp,fp,fn=c[...,0],c[...,1],c[...,2]
                numerator=tp*(2 if metric=="F1" else 1)
                denominator=numerator+fp+fn
                return np.divide(numerator,denominator,out=np.full(numerator.shape,np.nan,dtype=float),where=denominator>0)
            diffs=ratio(ca[draws].sum(axis=1))-ratio(cb[draws].sum(axis=1))
            effect=float(ratio(ca.sum(axis=0))-ratio(cb.sum(axis=0)))
            finite=diffs[np.isfinite(diffs)]
            ci=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
            rows.append(dict(threshold_A=t,metric=metric,effect=effect,ci_lower=float(ci[0]),ci_upper=float(ci[1]),
                             units=len(units),common_minutes=len(common),replicates=replicates,valid_replicates=len(finite)))
    return rows


def warnings(primary,metadata):
    events=pd.read_csv(DATA/"supplied_events.csv")
    blocks=pd.read_csv(DATA/"supplied_blocks.csv")
    # Catalogue membership, not magnitude-based post-hoc event selection.
    test_ids=set()
    for text in blocks.loc[blocks.split.eq("test"),"event_ids"]:
        test_ids.update(int(v.strip()) for v in str(text).split(","))
    events=events[events.event_id.isin(test_ids)]
    rows=[]
    h=int(metadata["h"])
    issue=primary["time"].astype("datetime64[ns]")
    for event in events.itertuples():
        block=next(b for b in blocks.itertuples() if event.event_id in [int(v.strip()) for v in str(b.event_ids).split(",")])
        z=np.load(DATA/"segments"/f"block_{block.block:02d}.npz")
        times=z["time"].astype("datetime64[ns]");g=np.abs(z["gic"])
        peak=pd.Timestamp(event.peak_time).floor("min").to_datetime64().astype("datetime64[ns]")
        where=np.flatnonzero(times==peak)
        for j,t in enumerate(THRESHOLDS):
            row=dict(event_id=event.event_id,block=block.block,threshold_A=t,catalogue_peak_time=str(peak),status="")
            if not len(where) or not np.isfinite(g[where[0]]):
                row["status"]="catalogue_peak_unavailable";rows.append(row);continue
            k=int(where[0])
            if g[k]<t:
                row.update(status="catalogue_peak_below_threshold_in_recovered_data",recovered_peak_bin_abs_A=float(g[k]))
                rows.append(row);continue
            begin=k
            while begin>0 and np.isfinite(g[begin-1]) and g[begin-1]>=t:begin-=1
            if begin==0 or not np.isfinite(g[begin-1]):
                row["status"]="onset_left_censored_by_gap";rows.append(row);continue
            onset=times[begin]
            eligible=(issue+np.timedelta64(h,"m")<=onset)&(onset<issue+np.timedelta64(h+30,"m"))
            at=np.flatnonzero(eligible)
            row.update(onset_time=str(onset),available_issuances=len(at))
            if not len(at):row["status"]="no_valid_forecast"
            else:
                hit=at[primary["p"][at,j]>=.5]
                if len(hit):
                    first=issue[hit[0]]
                    row.update(status="warned",first_warning_time=str(first),
                               warning_minutes=float((onset-first)/np.timedelta64(1,"m")))
                else:row["status"]="missed"
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_main():
    OUT.mkdir(parents=True, exist_ok=True)
    validate_completed_experiment()
    selected=json.loads((ROOT/"selection.json").read_text())["delta"]
    primary_name=f"fb_h0_d{selected}_s42"
    runs=[p for p in sorted((ROOT/"runs").iterdir()) if (p/"development_complete.json").exists()]
    all_rows=[];reliability_rows=[]
    for run in runs:
        m=json.loads((run/"development_complete.json").read_text())
        for split in ["test","continuous"]:
            data=load(run,split)
            for version,p in [("raw",expit(data["logits"])),("calibrated",data["p"])]:
                for policy,cutoffs in [("fixed_0.5",None),("calibration_CSI",m["secondary_cutoffs"])]:
                    if version=="raw" and policy!="fixed_0.5":continue
                    rows=score(data["y"],p,cutoffs,m["train_prevalence"])
                    for row in rows:row.update(run=run.name,split=split,probabilities=version,policy=policy)
                    all_rows.extend(rows)
                if run.name==primary_name:
                    for j,t in enumerate(THRESHOLDS):
                        for row in reliability(data["y"][:,j],p[:,j]):
                            row.update(split=split,probabilities=version,threshold_A=t)
                            reliability_rows.append(row)
    metrics=pd.DataFrame(all_rows)
    metrics.to_csv(OUT/"all_metrics.csv",index=False)
    pd.DataFrame(reliability_rows).to_csv(OUT/"primary_reliability_bins.csv",index=False)
    primary_run=ROOT/"runs"/primary_name
    primary=load(primary_run,"test")
    main_meta=json.loads((primary_run/"development_complete.json").read_text())
    warnings(primary,main_meta).to_csv(OUT/"catalogue_warning_times.csv",index=False)
    comparisons=[]
    for other in [f"logistic_h0_d{selected}",f"lstm_h0_d{selected}_s42",f"fa_h0_d{selected}_s42",f"no_integral_h0_d{selected}_s42"]:
        b=load(ROOT/"runs"/other,"test")
        for row in paired_bootstrap(primary,b):
            row.update(reference=primary_name,comparison=other,split="test")
            comparisons.append(row)
    # Whole-day units for the continuous-year comparison, preserving minute dependence.
    continuous=load(primary_run,"continuous")
    b=load(ROOT/"runs"/f"logistic_h0_d{selected}","continuous")
    for array in [continuous,b]:
        array["group"]=(array["time"].astype("datetime64[ns]").astype("datetime64[D]").astype(int))
    for row in paired_bootstrap(continuous,b):
        row.update(reference=primary_name,comparison=f"logistic_h0_d{selected}",split="continuous")
        comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(OUT/"paired_bootstrap.csv",index=False)
    seed_names=[f"fb_h0_d{selected}_s{s}" for s in [42,7,2026]]
    chosen=metrics[metrics.run.isin(seed_names)&metrics.probabilities.eq("calibrated")&metrics.policy.eq("fixed_0.5")]
    chosen.groupby(["split","threshold_A"])[["CSI","F1","Brier","POD","POFD","FAR","ROC_AUC","PR_AUC"]].agg(["mean","std"]).to_csv(OUT/"seed_variability.csv")
    # A canonical issue-time table suitable for external independent scoring.
    for split in ["test","continuous"]:
        a=load(primary_run,split)
        frame=pd.DataFrame(dict(issue_time=pd.to_datetime(a["time"]),event_group=a["group"]))
        frame["target_start"]=frame.issue_time+pd.Timedelta(minutes=main_meta["h"])
        frame["target_end_exclusive"]=frame.target_start+pd.Timedelta(minutes=30)
        for j,t in enumerate(THRESHOLDS):
            frame[f"target_exceeds_{t}A_30min"]=a["y"][:,j].astype(int)
            frame[f"probability_exceeds_{t}A_30min"]=a["p"][:,j]
            frame[f"raw_probability_exceeds_{t}A_30min"]=expit(a["logits"][:,j])
            frame[f"alarm_{t}A_fixed_0_5"]=(a["p"][:,j]>=.5).astype(int)
        frame.to_csv(OUT/f"{split}_probabilities.csv.gz",index=False,compression="gzip")
    # Compare all horizon/shift configurations on common issue times within each panel.
    sensitivity=[]
    panels={"horizon":[f"fb_h{h}_d0_s42" for h in [0,30,45,60,90]],
            "propagation_context":[f"fb_h0_d{d}_s42" for d in [0,30,45,60,90]]}
    for panel,names in panels.items():
        arrays={n:load(ROOT/"runs"/n,"test") for n in names}
        common=arrays[names[0]]["time"]
        for n in names[1:]:common=np.intersect1d(common,arrays[n]["time"])
        for n,a in arrays.items():
            use=np.isin(a["time"],common)
            for r in score(a["y"][use],a["p"][use]):
                r.update(panel=panel,run=n,common_minutes=len(common))
                sensitivity.append(r)
    pd.DataFrame(sensitivity).to_csv(OUT/"matched_sensitivity.csv",index=False)
    coverage_tables(primary_run)
    print(f"Evaluated {len(runs)} models; primary={primary_name}",flush=True)


class ProbabilityModel(nn.Module):
    def __init__(self,model,params):
        super().__init__();self.model=model
        self.register_buffer("a",torch.tensor([p["a"] for p in params],device=DEVICE))
        self.register_buffer("b",torch.tensor([p["b"] for p in params],device=DEVICE))
    def forward(self,x):
        p=torch.sigmoid(self.a*self.model(x)+self.b)
        candidates=[];costs=[]
        for mask in range(8):
            cuts=[0]+[j+1 for j in range(3) if mask&(1<<j)]+[4]
            q=torch.cat([p[:,l:r].mean(dim=1,keepdim=True).expand(-1,r-l) for l,r in zip(cuts[:-1],cuts[1:])],dim=1)
            valid=(q[:,1:]<=q[:,:-1]+1e-12).all(dim=1)
            cost=((p-q)**2).sum(dim=1).masked_fill(~valid,float("inf"))
            candidates.append(q);costs.append(cost)
        which=torch.stack(costs,dim=1).argmin(dim=1)
        return torch.stack(candidates,dim=1)[torch.arange(len(p),device=p.device),which]


def explain_main():
    import shap
    training = load_stage(4)
    training.configure_workspace(ROOT)
    training.DEVICE = DEVICE
    Data, model_for, batch_x = training.Data, training.model_for, training.batch_x
    delta=json.loads((ROOT/"selection.json").read_text())["delta"]
    run=ROOT/"runs"/f"fb_h0_d{delta}_s42"
    saved=torch.load(run/"best.pt",map_location=DEVICE,weights_only=False)
    meta=json.loads((run/"development_complete.json").read_text())
    data=Data(development=False);data.normalize(saved["mean"],saved["std"])
    model=model_for("fb", len(saved["cols"])*2);model.load_state_dict(saved["state_dict"])
    wrapper=ProbabilityModel(model,meta["calibration"]).eval()
    tr,ty=data.indices(0,delta,"train",5)
    test,y=data.indices(0,delta,"test")
    rng=np.random.default_rng(628)
    background_ix=rng.choice(tr,64,replace=False)
    # Equal stratum sample sizes improve rare-event coverage. Weights recover
    # the actual eligible-test distribution, rather than a 50:50 population.
    strata=y.sum(axis=1).astype(int)
    chosen=[];weights=[]
    for group in range(5):
        pool=np.flatnonzero(strata==group)
        if not len(pool):continue
        selected=rng.choice(pool,min(64,len(pool)),replace=False)
        chosen.extend(selected);weights.extend([len(pool)/len(y)/len(selected)]*len(selected))
    chosen=np.asarray(chosen);weights=np.asarray(weights)
    bg=batch_x(data,background_ix,delta,saved["cols"])
    x=batch_x(data,test[chosen],delta,saved["cols"])
    # Gradient computation through recurrent evaluation requires the native
    # RNN path; this changes the backend, not the trained weights or outputs.
    out=ROOT/"evaluation";out.mkdir(exist_ok=True)
    cache_path=out/"shap_computation_checkpoint.npz"
    model_hash=hashlib.sha256((run/"best.pt").read_bytes()).hexdigest()
    with torch.backends.cudnn.flags(enabled=False):
        with torch.no_grad():
            prediction=wrapper(x).cpu().numpy()
            baseline=wrapper(bg).mean(dim=0).cpu().numpy()
            native_logits=model(x).cpu().numpy()
        values=None
        if cache_path.exists():
            cached=np.load(cache_path)
            # Only reuse a completed gradient calculation when its exact
            # deterministic sample, weights and native outputs still agree.
            if ("model_sha256" in cached and str(cached["model_sha256"].item())==model_hash
                and np.array_equal(cached["chosen"],chosen) and np.array_equal(cached["weights"],weights)
                and np.allclose(cached["native_logits"],native_logits,rtol=0,atol=1e-6)
                and np.allclose(cached["baseline"],baseline,rtol=0,atol=1e-6)):
                values=cached["values"]
                print("Reusing verified gradient computation; repeating output checks.",flush=True)
        if values is None:
            explainer=shap.GradientExplainer(wrapper,bg,batch_size=64)
            values=np.asarray(explainer.shap_values(x,nsamples=200,rseed=628))
    np.savez_compressed(cache_path,values=values.astype(np.float32),
                        prediction=prediction,baseline=baseline,chosen=chosen,weights=weights,
                        native_logits=native_logits,model_sha256=model_hash)
    assert values.shape==(len(chosen),30,98,4),values.shape
    # Validate the interpretation target against the saved scored probabilities.
    # Verify the wrapper against the independent calibration implementation on
    # the same backend, rather than conflating TF32/cuDNN numerical differences
    # with an error in the probability definition.
    np.testing.assert_allclose(prediction,calibrated_probabilities(native_logits,meta["calibration"]),atol=2e-6)
    scored=np.load(run/"test.npz")
    np.testing.assert_array_equal(scored["time"][chosen],data.time[test[chosen]])
    scored_difference=float(np.abs(prediction-scored["p"][chosen]).max())
    # Independently adjudicate reduced-precision differences against CPU double.
    reference=copy.deepcopy(model).cpu().double().eval()
    with torch.no_grad():reference_logits=reference(x.cpu().double()).numpy()
    reference_prob=calibrated_probabilities(reference_logits,meta["calibration"])
    reference_difference=float(np.abs(prediction-reference_prob).max())
    assert reference_difference < 2e-5,reference_difference
    assert scored_difference < 2e-3,scored_difference
    alarm_differences=int(np.sum((prediction>=.5)!=(scored["p"][chosen]>=.5)))
    del reference
    residual=prediction-baseline-values.sum(axis=(1,2))
    per_input=np.abs(values).sum(axis=1)
    grouped=per_input[:,:49,:]+per_input[:,49:,:]
    importance=np.einsum("n,nfj->fj",weights,grouped)
    rows=[dict(feature=feature,threshold_A=t,mean_abs_SHAP=float(importance[i,j]))
          for i,feature in enumerate(FEATURES) for j,t in enumerate([3,5,10,20])]
    pd.DataFrame(rows).to_csv(out/"shap_importance.csv",index=False)
    np.savez_compressed(out/"shap_values.npz",values=values.astype(np.float32),weights=weights,
                        issue_time=data.time[test[chosen]],background_time=data.time[background_ix],
                        y=y[chosen],prediction=prediction,baseline=baseline,residual=residual)
    (out/"shap_metadata.json").write_text(json.dumps(dict(
        library_version=shap.__version__,evaluation_sequences=len(chosen),background_sequences=64,
        draws_per_sequence=200,output="Calibrated, ordered probabilities for each threshold separately",
        grouping="Sum absolute contributions across 30 times and two context streams, then population-weighted mean",
        weighting="Five mutually exclusive maximum-threshold strata; actual eligible-test stratum prevalences",
        max_absolute_completeness_residual=float(np.abs(residual).max()),
        mean_absolute_completeness_residual=np.abs(residual).mean(axis=0).tolist(),
        max_absolute_difference_from_scored_mixed_precision_probabilities=scored_difference,
        max_absolute_difference_from_independent_CPU_float64=reference_difference,
        primary_alarm_differences_on_explanation_sample=alarm_differences,
        precision_note="Native float32 gradient outputs verified against CPU float64 within2e-5; same-backend wrapper/calibration within2e-6. Mixed-precision scored probability differences recorded and bounded below2e-3.",
        limitations="Marginal background can create unrealistic combinations of dependent features; attribution is not causal."
    ),indent=2))
    # Associations are estimated from training data only, never used to select
    # a model using the test labels.
    sample=rng.choice(tr,min(50000,len(tr)),replace=False)
    targets=np.full(len(data.gic),np.nan,dtype=np.float32)
    for part in data.parts:
        if part["split"]=="train":
            s,n=part["offset"],part["n"]
            targets[s:s+n]=future_peak(np.abs(data.gic[s:s+n]),0)
    feature_ids=[FEATURES.index(n) for n in ["Btot","Np","V","Epsilon","Pdyn_proton","Bz_gsm","Ey_south","Newell","VBt_clock_proxy"]]
    raw=data.x[sample][:,feature_ids].copy()
    raw[:,5]=np.maximum(-raw[:,5],0)
    names=["Btot","Np","V","Epsilon","Pdyn_proton","Bz_south","Ey_south","Newell","VBt_clock_proxy"]
    mi=mutual_info_regression(raw,targets[sample],n_neighbors=3,random_state=628)
    association=[dict(feature=n,MI=float(mi[j]),Spearman=float(spearmanr(raw[:,j],targets[sample]).statistic),N=len(sample))
                 for j,n in enumerate(names)]
    pd.DataFrame(association).to_csv(out/"training_associations.csv",index=False)
    np.savez_compressed(out/"association_plot_data.npz",x=raw,y=targets[sample],names=np.array(names))
    print("Per-threshold SHAP and training associations complete.",flush=True)


def validate_completed_experiment():
    """Require the comparisons used by the submitted manuscript."""
    selection = ROOT / "selection.json"
    if not selection.exists():
        raise FileNotFoundError("Run Stage 04 --phase search before evaluation; selection.json is a generated result.")
    delta = int(json.loads(selection.read_text(encoding="utf-8"))["delta"])
    expected = {f"fb_h{h}_d0_s42" for h in [0, 30, 45, 60, 90]}
    expected.update(f"fb_h0_d{d}_s42" for d in [30, 45, 60, 90])
    expected.update(f"{kind}_h0_d{delta}_s42" for kind in ["fa", "lstm", "no_integral"])
    expected.update(f"fb_h0_d{delta}_s{seed}" for seed in [7, 2026])
    expected.add(f"logistic_h0_d{delta}")
    missing = []
    for name in sorted(expected):
        for filename in ["development_complete.json", "test.npz", "continuous.npz"]:
            if not (ROOT / "runs" / name / filename).exists():
                missing.append(f"runs/{name}/{filename}")
    if missing:
        raise FileNotFoundError("Complete Stage 04 search/selected/infer first. Missing:\n" + "\n".join(missing))


def coverage_tables(primary_run):
    """Recompute Tables 3/S4/S5 inputs from actual segments and eligibility."""
    training = load_stage(4)
    training.configure_workspace(ROOT)
    delta = int(json.loads((primary_run / "development_complete.json").read_text(encoding="utf-8"))["delta"])
    data = training.Data(development=False)
    block_rows = []
    for part in data.parts:
        if part["block"] > 49:
            continue
        start, n = part["offset"], part["n"]
        local = data.in_block[start:start+n]
        complete = np.isfinite(data.x[start:start+n]).all(axis=1)
        total = int(local.sum())
        block_rows.append(dict(
            block=part["block"], split=part["meta"]["split"], total_minutes=total,
            valid_gic_minutes=int((local & np.isfinite(data.gic[start:start+n])).sum()),
            complete_feature_minutes=int((local & complete).sum()),
            feature_completeness_percent=float((local & complete).sum()/total*100),
        ))
    pd.DataFrame(block_rows).to_csv(OUT / "block_coverage.csv", index=False)
    partitions = []
    for split in ["train", "selection", "calibration", "test", "continuous"]:
        indices, labels = data.common_selection(0, delta) if split == "selection" else data.indices(0, delta, split, 5 if split == "train" else 1)
        row = dict(partition=split, N=len(indices), stride_minutes=5 if split == "train" else 1)
        for j, threshold in enumerate(THRESHOLDS):
            row[f"positive_percent_{threshold}A"] = float(labels[:, j].mean()*100) if len(labels) else np.nan
        partitions.append(row)
    pd.DataFrame(partitions).to_csv(OUT / "eligible_samples.csv", index=False)
    selected = load(primary_run, "continuous")
    with np.load(DATA / "segments/continuous_2022.npz") as year:
        times, mask, gic = pd.to_datetime(year["time"]), year["in_block"], year["gic"]
        issued_months = pd.to_datetime(selected["time"]).month
        rows = []
        for month in range(1, 13):
            use = mask & (times.month == month)
            total, issued = int(use.sum()), int((issued_months == month).sum())
            rows.append(dict(month=month, total_minutes=total,
                             valid_GIC_minutes=int(np.isfinite(gic[use]).sum()),
                             forecast_minutes=issued, coverage_fraction=issued/total if total else np.nan))
    pd.DataFrame(rows).to_csv(OUT / "continuous_coverage.csv", index=False)
    # Record the progressive current/delayed/target availability used in 4.2.
    part = next(p for p in data.parts if p["split"] == "continuous")
    start, n = part["offset"], part["n"]
    current = data.valid_window[start:start+n].copy()
    delayed = np.zeros(n, bool)
    if delta:
        delayed[delta:] = data.valid_window[start:start+n-delta]
    else:
        delayed = current.copy()
    in_year = data.in_block[start:start+n]
    pd.DataFrame([dict(total_minutes=int(in_year.sum()),
                       complete_current_minutes=int((in_year & current).sum()),
                       complete_both_streams_minutes=int((in_year & current & delayed).sum()),
                       complete_inputs_and_targets_minutes=len(selected["time"]))]).to_csv(OUT / "continuous_eligibility.csv", index=False)
    rows = []
    for context in [0, 30, 45, 60, 90]:
        path = ROOT / "runs" / f"fb_h0_d{context}_s42/development_complete.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        rows.append(dict(delta=context, selection_BCE=metadata["best_selection_bce"], N=metadata["selection_N"]))
    pd.DataFrame(rows).to_csv(OUT / "context_selection.csv", index=False)


def self_check(with_catalogue=False):
    """Independent boundary/cross-library checks without fitting a model."""
    from sklearn.isotonic import isotonic_regression
    from sklearn.metrics import confusion_matrix, brier_score_loss
    checks = {}
    if with_catalogue:
        events, blocks = dataset.preprocessing.supplied_tables()
        checks["catalogue"] = [len(events), sum(bool(e["retained"]) for e in events), len(blocks)]
    source = pd.DataFrame({"v": [10., 20., np.nan, 40.]}, index=pd.DatetimeIndex([
        "2020-01-01 00:00:30", "2020-01-01 00:01:45", "2020-01-01 00:02:30", "2020-01-01 00:10:30"
    ], name="source_time"))
    grid = pd.date_range("2020-01-01 00:01", periods=6, freq="min")
    aligned, age = dataset.preprocessing.align_prior(source, grid, 120)
    np.testing.assert_allclose(aligned.v.to_numpy(), [10, 20, np.nan, np.nan, np.nan, np.nan], equal_nan=True)
    checks["causal_alignment_and_gap_preservation"] = True
    gic = np.arange(250, dtype=float)
    gic[110] = np.nan
    for horizon in [0, 30, 45, 60, 90]:
        expected = np.array([np.max(gic[i+horizon:i+horizon+30]) if i+horizon+30 <= len(gic) else np.nan for i in range(len(gic))])
        np.testing.assert_allclose(future_peak(gic, horizon), expected, equal_nan=True)
    checks["future_labels_direct_enumeration"] = True
    rng = np.random.default_rng(2419)
    probabilities = ordered_probabilities(rng.uniform(size=(1000, 4)))
    original = rng.uniform(size=(1000, 4))
    np.testing.assert_allclose(ordered_probabilities(original), np.array([isotonic_regression(row, increasing=False) for row in original]), atol=1e-12)
    checks["ordered_projection_against_sklearn"] = True
    mask = rng.integers(0, 2, 100).astype(bool)
    times = pd.date_range("2020-01-01", periods=100, freq="min").to_numpy()
    groups = np.repeat([1, 2], 50)
    whole = causal_confirm(mask, times, groups, 3)
    for n in range(1, 101):
        np.testing.assert_array_equal(whole[:n], causal_confirm(mask[:n], times[:n], groups[:n], 3))
    checks["causal_alarm_prefix_invariance"] = True
    result = episode_metrics([1, 1, 1, 1, 1], [1, 0, 0, 0, 1], times[:5], [1]*5)
    assert (result["TP"], result["FP"], result["FN"]) == (1, 1, 0)
    checks["one_to_one_episode_matching"] = True
    labels = np.sort(rng.integers(0, 2, (1000, 4)), axis=1)[:, ::-1]
    for j, row in enumerate(score(labels, probabilities)):
        tn, fp, fn, tp = confusion_matrix(labels[:, j], probabilities[:, j] >= .5, labels=[0, 1]).ravel()
        assert [row[key] for key in ["TN", "FP", "FN", "TP"]] == [tn, fp, fn, tp]
        assert abs(row["Brier"] - brier_score_loss(labels[:, j], probabilities[:, j])) < 1e-12
    checks["confusion_and_brier_against_sklearn"] = True
    print(json.dumps(checks, indent=2))
    return checks
COLORS = ["#0018C0", "#00A0FF", "#FB5959", "#C00000"]
MODEL_COLORS = {"CNN-BiLSTM": "#0018C0", "LSTM": "#FB5959", "Logistic": "#2B2B2B"}


def nice(feature):
    return (feature.replace("Bz_south_integral_30min", "Southward IMF integral")
            .replace("VBt_clock_proxy", "V Bt clock proxy").replace("Pdyn_proton", "Proton pressure")
            .replace("_30min", " (30 min)").replace("_gsm", " GSM").replace("_", " "))


def panel(axis, letter, title):
    axis.set_title(f"({letter}) {title}", fontsize=9.5, fontweight="bold", pad=5)
    axis.minorticks_on()


def save_figure(figure, name):
    figure.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def figures_main():
    """Figures 1-7 and S1-S2 follow the final submission's panel contents."""
    global plt, mdates
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from sklearn.metrics import roc_curve, precision_recall_curve
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": .9,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "legend.fontsize": 7.5, "legend.frameon": True, "legend.fancybox": False,
        "savefig.dpi": 400, "pdf.fonttype": 42,
    })
    required = ["all_metrics.csv", "training_associations.csv", "matched_sensitivity.csv",
                "shap_importance.csv", "primary_reliability_bins.csv"]
    missing = [str(OUT / name) for name in required if not (OUT / name).exists()]
    if missing:
        raise FileNotFoundError("Run --mode evaluate and --mode explain before figures. Missing:\n" + "\n".join(missing))
    FIGURES.mkdir(parents=True, exist_ok=True)
    delay = int(json.loads((ROOT / "selection.json").read_text(encoding="utf-8"))["delta"])
    primary = f"fb_h0_d{delay}_s42"
    names = [primary, f"lstm_h0_d{delay}_s42", f"logistic_h0_d{delay}"]
    legends = ["CNN-BiLSTM", "LSTM", "Logistic"]
    arrays = {name: load(ROOT / "runs" / name, "test") for name in names}
    metrics = pd.read_csv(OUT / "all_metrics.csv")
    fixed = metrics[metrics.probabilities.eq("calibrated") & metrics.policy.eq("fixed_0.5")]
    blocks = pd.read_csv(DATA / "supplied_blocks.csv")
    metadata = json.loads((ROOT / "runs" / primary / "development_complete.json").read_text(encoding="utf-8"))

    # Figure 1: actual event-block durations, histogram inset, GIC exceedance.
    figure, axes = plt.subplots(2, 1, figsize=(7, 5.0), layout="constrained")
    for split, color, level in [("train", COLORS[0], 0), ("val", COLORS[1], 1), ("test", COLORS[2], 2)]:
        for row in blocks[blocks.split.eq(split)].itertuples():
            left = mdates.date2num(pd.Timestamp(row.start))
            days = (pd.Timestamp(row.end) - pd.Timestamp(row.start)) / pd.Timedelta(days=1)
            axes[0].broken_barh([(left, days)], (level-.3, .6), facecolors=color)
    axes[0].broken_barh([(mdates.date2num(pd.Timestamp("2022-01-01")), 365)], (2.7, .6), facecolors=COLORS[3])
    axes[0].set(yticks=[0, 1, 2, 3], yticklabels=["train", "val", "test", "2022 continuous"], ylim=(-.55, 3.6), xlabel="UTC year")
    axes[0].xaxis_date()
    axes[0].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    panel(axes[0], "a", "Event blocks and continuous-year interval")
    durations = (pd.to_datetime(blocks.end) - pd.to_datetime(blocks.start)).dt.total_seconds()/86400
    inset = axes[0].inset_axes([.12, .18, .25, .35])
    inset.hist(durations, bins=np.linspace(9.5, max(21, durations.max()+.5), 8), color=COLORS[0], edgecolor="black", linewidth=.3)
    inset.set(xlabel="Block days", ylabel="Count")
    inset.tick_params(labelsize=6)
    for split, label, color in [("test", "Event test", COLORS[0]), ("continuous", "Continuous 2022", COLORS[2])]:
        paths = [DATA / "segments" / f"block_{i:02d}.npz" for i in range(41, 50)] if split == "test" else [DATA / "segments/continuous_2022.npz"]
        values = []
        for path in paths:
            with np.load(path) as segment:
                values.append(np.abs(segment["gic"][segment["in_block"]]))
        values = np.concatenate(values)
        values = np.sort(values[np.isfinite(values)])
        axes[1].step(values, 1-np.arange(len(values))/float(len(values)), where="post", label=label, color=color, lw=1.1)
    axes[1].set(xscale="log", yscale="log", xlim=(.05, 150), ylim=(1e-6, 1),
                xlabel=r"Minute-peak $|GIC|$ (A)", ylabel=r"Fraction of valid minutes $\geq x$")
    axes[1].legend(loc="lower left")
    panel(axes[1], "b", "Exceedance of minute-peak GIC")
    save_figure(figure, "Figure_1")

    # Figure 2 / S1: all four thresholds, ranking rather than calibration.
    for thresholds, name in [([3, 5], "Figure_2"), ([10, 20], "Figure_S1")]:
        figure, axes = plt.subplots(2, 2, figsize=(7, 6.2), layout="constrained")
        for row, threshold in enumerate(thresholds):
            j = THRESHOLDS.index(threshold)
            for run_name, legend in zip(names, legends):
                y, p = arrays[run_name]["y"][:, j], arrays[run_name]["p"][:, j]
                fpr, tpr, _ = roc_curve(y, p)
                precision, recall, _ = precision_recall_curve(y, p)
                axes[row, 0].plot(fpr, tpr, color=MODEL_COLORS[legend], label=f"{legend}: {roc_auc_score(y,p):.3f}")
                axes[row, 1].plot(recall, precision, color=MODEL_COLORS[legend], label=f"{legend}: {average_precision_score(y,p):.3f}")
            axes[row, 0].plot([0, 1], [0, 1], ls=":", color="#555555")
            axes[row, 1].axhline(arrays[primary]["y"][:, j].mean(), ls=":", color="#555555", label="Observed prevalence")
            for column, axis in enumerate(axes[row]):
                axis.set(xlim=(0, 1), ylim=(0, 1))
                axis.legend(loc="best", fontsize=7)
                panel(axis, chr(97+row*2+column), f"{'ROC' if column==0 else 'PR'}, {threshold} A")
            axes[row, 0].set(xlabel="False positive rate", ylabel="True positive rate")
            axes[row, 1].set(xlabel="Recall", ylabel="Precision")
        save_figure(figure, name)

    # Figure 3: the complete calendar year 2022, with both alarm policies.
    continuous = load(ROOT / "runs" / primary, "continuous")
    grid = pd.date_range("2022-01-01", "2023-01-01", freq="min", inclusive="left")
    probabilities = pd.DataFrame(continuous["p"], index=pd.to_datetime(continuous["time"])).reindex(grid)
    targets = pd.DataFrame(continuous["y"], index=pd.to_datetime(continuous["time"])).reindex(grid)
    with np.load(DATA / "segments/continuous_2022.npz") as segment:
        raw = pd.Series(np.abs(segment["gic"]), index=pd.to_datetime(segment["time"])).reindex(grid)
    figure, axes = plt.subplots(5, 1, figsize=(7.2, 7.6), sharex=True, layout="constrained")
    axes[0].plot(grid, raw, color="black", lw=.5)
    axes[0].set_ylabel(r"$|GIC|$ (A)")
    panel(axes[0], "a", "Minute-peak GIC, calendar year 2022")
    for j, threshold in enumerate(THRESHOLDS):
        axis = axes[j+1]
        axis.plot(grid, probabilities[j], color="black", lw=.5)
        axis.axhline(.5, ls="--", color=COLORS[3], lw=.9, label="Fixed cutoff = 0.5")
        axis.axhline(metadata["secondary_cutoffs"][j], ls=":", color="#666666", lw=.9,
                     label=f"Calibration cutoff = {metadata['secondary_cutoffs'][j]:.3f}")
        alarms = probabilities[j].ge(.5).to_numpy()
        axis.plot(grid[alarms], np.full(int(alarms.sum()), .025), "|", ms=2.5, color=COLORS[0], label="Fixed-0.5 alarms")
        axis.fill_between(grid, 0, targets[j]*.07, step="post", color=COLORS[3], alpha=.65, label="Future-window target")
        axis.set(ylim=(0, 1), ylabel="Probability")
        axis.legend(loc="upper right", fontsize=6, ncol=2)
        panel(axis, chr(98+j), f"{threshold} A at issue time")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].set_xlim(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01"))
    axes[-1].set_xlabel("Forecast issue time (UTC, 2022); gaps denote unavailable forecasts")
    save_figure(figure, "Figure_3")

    # Figure 4: associations (a,b) and controlled feature comparisons (c,d).
    association = pd.read_csv(OUT / "training_associations.csv")
    figure, axes = plt.subplots(2, 2, figsize=(7.6, 7.2), layout="constrained")
    for axis, metric, color, letter in zip(axes[0], ["Spearman", "MI"], [COLORS[0], COLORS[2]], ["a", "b"]):
        axis.barh([nice(name) for name in association.feature], association[metric], color=color, edgecolor="black", linewidth=.3)
        axis.invert_yaxis()
        axis.tick_params(axis="y", labelsize=7)
        axis.set_xlabel("Spearman rank correlation" if metric == "Spearman" else "Mutual information (nats)")
        panel(axis, letter, "Training-only associations")
    comparison_names = [primary, f"fa_h0_d{delay}_s42", f"no_integral_h0_d{delay}_s42", f"logistic_h0_d{delay}"]
    for run_name, label, color in zip(comparison_names, ["FB", "FA", "FB without integral", "Logistic"], COLORS):
        rows = fixed[fixed.run.eq(run_name) & fixed.split.eq("test")].sort_values("threshold_A")
        for axis, metric in zip(axes[1], ["CSI", "Brier"]):
            axis.plot(range(4), rows[metric], marker="o", ms=3.2, label=label, color=color)
            axis.set(xticks=range(4), xticklabels=["3", "5", "10", "20"], xlabel="GIC threshold (A)", ylabel=metric)
    axes[1, 0].legend(loc="best", fontsize=7)
    panel(axes[1, 0], "c", "Event-test CSI")
    panel(axes[1, 1], "d", "Event-test Brier score")
    save_figure(figure, "Figure_4")

    # Figure 5: separate horizon and delayed-context sensitivity.
    sensitivity = pd.read_csv(OUT / "matched_sensitivity.csv")
    figure, axes = plt.subplots(2, 2, figsize=(7, 5.6), layout="constrained")
    for row, kind in enumerate(["horizon", "propagation_context"]):
        for threshold, color in zip(THRESHOLDS, COLORS):
            rows = sensitivity[sensitivity.panel.eq(kind) & sensitivity.threshold_A.eq(threshold)].copy()
            token = r"_h(\d+)_" if kind == "horizon" else r"_d(\d+)_"
            rows["x"] = rows.run.str.extract(token)[0].astype(int)
            rows = rows.sort_values("x")
            for column, (axis, metric) in enumerate(zip(axes[row], ["CSI", "Brier"])):
                axis.plot(rows.x, rows[metric], marker="o", ms=3.2, color=color, label=f"{threshold} A")
                axis.set(xlabel="Forecast horizon h (min)" if row == 0 else "Delayed context delta (min)", ylabel=metric, xticks=[0,30,45,60,90])
                panel(axis, chr(97+row*2+column), f"{metric}, {'horizon' if row==0 else 'delayed context'}")
    axes[0, 0].legend(loc="best", ncol=2, fontsize=7)
    save_figure(figure, "Figure_5")

    # Figure 6: separate calibrated-output attribution rankings.
    importance = pd.read_csv(OUT / "shap_importance.csv")
    figure, axes = plt.subplots(2, 2, figsize=(8, 6.4), layout="constrained")
    for k, (axis, threshold) in enumerate(zip(axes.flat, THRESHOLDS)):
        rows = importance[importance.threshold_A.eq(threshold)].nlargest(8, "mean_abs_SHAP").iloc[::-1]
        axis.barh([nice(name) for name in rows.feature], rows.mean_abs_SHAP, color=COLORS[k], edgecolor="black", linewidth=.3)
        axis.set_xlabel("Population-weighted mean absolute attribution")
        axis.tick_params(axis="y", labelsize=7)
        panel(axis, chr(97+k), f"{threshold} A")
    save_figure(figure, "Figure_6")

    # Figure 7 / S2: event-test and full-year reliability with identical bins.
    reliability_rows = pd.read_csv(OUT / "primary_reliability_bins.csv")
    for split, name in [("test", "Figure_7"), ("continuous", "Figure_S2")]:
        figure, axes = plt.subplots(2, 2, figsize=(7, 6.2), layout="constrained")
        for k, (axis, threshold) in enumerate(zip(axes.flat, THRESHOLDS)):
            for version, color in [("raw", COLORS[2]), ("calibrated", COLORS[0])]:
                rows = reliability_rows[reliability_rows.split.eq(split) & reliability_rows.probabilities.eq(version)
                                        & reliability_rows.threshold_A.eq(threshold) & reliability_rows["count"].gt(0)]
                axis.plot(rows.forecast_mean, rows.observed_frequency, marker="o", ms=3.2, color=color, label=version.capitalize())
            axis.plot([0,1], [0,1], ls=":", color="#555555")
            axis.set(xlim=(0,1), ylim=(0,1), xlabel="Mean forecast probability", ylabel="Observed frequency")
            panel(axis, chr(97+k), f"{threshold} A")
        axes[0,0].legend(loc="best", fontsize=7.5)
        save_figure(figure, name)
    print("Generated Figure_1 through Figure_7, Figure_S1 and Figure_S2.", flush=True)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["evaluate", "explain", "figures", "all", "check"], default="evaluate")
    parser.add_argument("--workspace", type=Path, help="Local data/run root; default is the repository root.")
    parser.add_argument("--with-catalogue", action="store_true", help="Also check local input catalogue counts in --mode check.")
    args = parser.parse_args()
    configure_workspace(args.workspace)
    if args.mode == "check":
        self_check(args.with_catalogue)
        return
    if args.mode in {"evaluate", "all"}:
        evaluate_main()
    if args.mode in {"explain", "all"}:
        explain_main()
    if args.mode in {"figures", "all"}:
        figures_main()


if __name__ == "__main__":
    main()
