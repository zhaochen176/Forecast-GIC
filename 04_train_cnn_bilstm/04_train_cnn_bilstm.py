"""Stage 04: network, training, validation selection, calibration and inference.

Manuscript Sections 2.3, 3.4 and 4.1: four outputs, h and delta varied
separately, a logistic baseline, retrained ablation and seeds 42/7/2026.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, random, sys, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression

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
verification = load_stage(5)
ROOT = REPOSITORY_ROOT
DATA = ROOT / "data"
FEATURES, FA, THRESHOLDS = dataset.FEATURES, dataset.FA, dataset.THRESHOLDS
future_peak = dataset.future_peak
fit_calibration = verification.fit_calibration
calibrated_probabilities = verification.calibrated_probabilities
select_cutoffs = verification.select_cutoffs
TOTAL_OUTPUT_COUNT = 4
torch.set_num_threads(4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def configure_workspace(workspace=None):
    global ROOT, DATA
    ROOT = Path(workspace).resolve() if workspace is not None else REPOSITORY_ROOT
    DATA = ROOT / "data"
    dataset.configure_workspace(ROOT)
    verification.configure_workspace(ROOT)

class CNNBiLSTMClassifier(nn.Module):
    """Fuse direct CNN peak evidence with attentive BiLSTM sequence context.

    The CNN branch pools its local and dilated responses directly with mean
    and max pooling.  In parallel, the BiLSTM branch summarizes the same CNN
    sequence with terminal states and learned temporal attention.  A learned
    element-wise gate fuses the two representations before classification,
    allowing the model to retain short-duration CNN responses rather than
    forcing every decision through the recurrent encoder.
    """
    encoder_type = 'multi_scale_residual_cnn_direct_pool_gated_attentive_bilstm'
    pooling_type = 'cnn_temporal_mean_max_plus_bilstm_terminal_attention_gated_fusion'

    def __init__(self, feature_count: int, channels: int=24, lstm_hidden_size: int=64, lstm_layers: int=2, dropout: float=0.1, attention_hidden_size: int=32, output_count: int=TOTAL_OUTPUT_COUNT):
        super().__init__()
        self.input_projection = nn.Linear(feature_count, channels) if feature_count != channels else nn.Identity()
        self.input_norm = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.bn2 = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.conv_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=channels, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        cnn_representation_size = channels * 2
        lstm_representation_size = lstm_hidden_size * 4
        self.sequence_norm = nn.LayerNorm(lstm_hidden_size * 2)
        self.attention = nn.Sequential(nn.Linear(lstm_hidden_size * 2, attention_hidden_size), nn.Tanh(), nn.Linear(attention_hidden_size, 1))
        self.cnn_pool_norm = nn.LayerNorm(cnn_representation_size)
        self.lstm_pool_norm = nn.LayerNorm(lstm_representation_size)
        self.lstm_projection = nn.Sequential(nn.Linear(lstm_representation_size, cnn_representation_size), nn.GELU(), nn.Dropout(dropout))
        self.fusion_gate = nn.Sequential(nn.Linear(cnn_representation_size * 2, cnn_representation_size), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(cnn_representation_size)
        self.classification_head = nn.Sequential(nn.Linear(cnn_representation_size, cnn_representation_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(cnn_representation_size, output_count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.input_norm(self.input_projection(x)).transpose(1, 2)
        residual = encoded
        local = self.activation(self.bn1(self.conv1(encoded)))
        wide = self.activation(self.bn2(self.conv2(local)))
        encoded = self.conv_dropout(local + wide) + residual
        cnn_representation = self.cnn_pool_norm(torch.cat([encoded.mean(dim=2), encoded.amax(dim=2)], dim=1))
        sequence, (hidden, _) = self.lstm(encoded.transpose(1, 2))
        sequence = self.sequence_norm(sequence)
        attention_weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        attended = torch.sum(sequence * attention_weights.unsqueeze(-1), dim=1)
        terminal = torch.cat([hidden[-2], hidden[-1]], dim=1)
        lstm_representation = self.lstm_projection(self.lstm_pool_norm(torch.cat([terminal, attended], dim=1)))
        cnn_gate = self.fusion_gate(torch.cat([cnn_representation, lstm_representation], dim=1))
        representation = cnn_gate * cnn_representation + (1.0 - cnn_gate) * lstm_representation
        return self.classification_head(self.fusion_norm(representation))


class LSTM(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.lstm=nn.LSTM(n,64,2,batch_first=True,dropout=.35)
        self.head=nn.Sequential(nn.Dropout(.35),nn.Linear(64,4))
    def forward(self,x):
        _,(h,_)=self.lstm(x)
        return self.head(h[-1])


class Data:
    def __init__(self, development=True, available=False):
        paths=[DATA/"segments"/f"block_{i:02d}.npz" for i in range(1,41 if development else 50)]
        if not development: paths.append(DATA/"segments/continuous_2022.npz")
        if available: paths=[p for p in paths if p.exists()]
        self.parts=[];xs=[];gs=[];ts=[];groups=[];masks=[];offset=0
        for path in paths:
            a=np.load(path)
            meta=json.loads(path.with_suffix(".json").read_text())
            n=len(a["x"]);block=meta["block"]
            split=("train" if block<=30 else "selection" if block<=35 else "calibration" if block<=40
                   else "test" if block<=49 else "continuous")
            self.parts.append(dict(offset=offset,n=n,block=block,split=split,meta=meta))
            xs.append(a["x"]);gs.append(a["gic"]);ts.append(a["time"])
            groups.append(np.full(n,block,dtype=np.int16));masks.append(a["in_block"])
            offset+=n
        self.x=np.concatenate(xs); self.gic=np.concatenate(gs);self.time=np.concatenate(ts)
        self.group=np.concatenate(groups);self.in_block=np.concatenate(masks)
        self.valid_window=np.zeros(offset,bool)
        self.last_label_time=np.zeros(offset,np.int64)
        for part in self.parts:
            sl=slice(part["offset"],part["offset"]+part["n"])
            valid=np.isfinite(self.x[sl]).all(axis=1)
            self.valid_window[sl]=pd.Series(valid.astype(int)).rolling(30,min_periods=30).sum().eq(30)
            self.last_label_time[sl]=pd.Timestamp(part["meta"]["end"]).value

    def indices(self,h,delta,split,stride=1):
        idx=[]; ys=[]
        for part in self.parts:
            if part["split"]!=split:continue
            start,n=part["offset"],part["n"];sl=slice(start,start+n)
            peak=future_peak(np.abs(self.gic[sl]),h)
            local=np.arange(n)
            valid=self.valid_window[sl].copy()
            if delta:
                delayed=np.zeros(n,bool);delayed[delta:]=self.valid_window[start:start+n-delta]
                valid &= delayed
            valid &= self.in_block[sl] & np.isfinite(peak)
            valid &= self.time[sl]+(h+30)*60_000_000_000 <= self.last_label_time[sl]
            # The fixed-stride origin is the block's first issue minute.
            first=np.flatnonzero(self.in_block[sl])[0]
            valid &= (local-first)%stride==0
            at=np.flatnonzero(valid)
            idx.append(at+start);ys.append((peak[at,None]>=THRESHOLDS).astype(np.float32))
        return (np.concatenate(idx),np.concatenate(ys)) if idx else (np.array([],int),np.empty((0,4),np.float32))

    def common_selection(self,h,delta):
        # Delta selection is compared on exactly the same validation issues.
        ix,y=self.indices(h,delta,"selection")
        if h==0:
            for d in [0,30,45,60,90]:
                other,_=self.indices(0,d,"selection")
                use=np.isin(ix,other);ix=ix[use];y=y[use]
        return ix,y

    def normalize(self,mean=None,std=None):
        transformed=np.sign(self.x)*np.log1p(np.abs(self.x))
        if mean is None:
            train=(self.group<=30)&self.in_block
            mean=np.nanmean(transformed[train],axis=0,dtype=np.float64).astype(np.float32)
            std=np.nanstd(transformed[train],axis=0,dtype=np.float64).astype(np.float32)
            std=np.maximum(std,1e-6)
        self.mean,self.std=mean,std
        values=np.nan_to_num((transformed-mean)/std,nan=0,posinf=0,neginf=0).astype(np.float32)
        self.tensor=torch.as_tensor(values,device=DEVICE)
        del transformed,values


def columns(kind):
    if kind=="fa":return [FEATURES.index(n) for n in FA]
    if kind=="no_integral":return [i for i,n in enumerate(FEATURES) if "Bz_south_integral" not in n]
    return list(range(len(FEATURES)))


def batch_x(data,idx,delta,cols):
    at=torch.as_tensor(idx,device=DEVICE,dtype=torch.long)
    offsets=torch.arange(-29,1,device=DEVICE)
    latest=data.tensor[(at[:,None]+offsets[None,:])]
    earlier=data.tensor[(at[:,None]-delta+offsets[None,:])]
    return torch.cat([latest[:,:,cols],earlier[:,:,cols]],dim=2)


def model_for(kind,n):
    if kind=="lstm":return LSTM(n).to(DEVICE)
    return CNNBiLSTMClassifier(n,channels=32,lstm_hidden_size=64,lstm_layers=2,
            dropout=.35,attention_hidden_size=32,output_count=4).to(DEVICE)


def predict(model,data,idx,delta,cols,batch=2048):
    model.eval();pred=[]
    with torch.no_grad():
        for start in range(0,len(idx),batch):
            with torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=DEVICE.type=="cuda"):
                out=model(batch_x(data,idx[start:start+batch],delta,cols))
            pred.append(out.float().cpu().numpy())
    return np.concatenate(pred) if pred else np.empty((0,4),np.float32)


def name(kind,h,delta,seed):
    return f"{kind}_h{h}_d{delta}_s{seed}"


def train(data,kind,h,delta,seed,epochs=30,smoke=False):
    run=ROOT/("smoke-runs" if smoke else "runs")/name(kind,h,delta,seed)
    run.mkdir(parents=True,exist_ok=True)
    if (run/"development_complete.json").exists():return
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    cols=columns(kind);model=model_for(kind,len(cols)*2)
    tr,y=data.indices(h,delta,"train",5)
    va,vy=data.common_selection(h,delta)
    ca,cy=data.indices(h,delta,"calibration")
    if smoke:
        take=np.unique(np.r_[np.linspace(0,len(tr)-1,min(1024,len(tr)),dtype=int),
                             np.flatnonzero(y[:,3])[:32]])
        tr,y=tr[take],y[take];va,vy=tr[:256],y[:256];ca,cy=tr[:256],y[:256]
    if not len(tr) or not len(va) or not len(ca):raise ValueError("An empty development partition")
    pos=y.sum(axis=0)
    if np.any(pos==0):raise ValueError(f"No positive training class: {pos}")
    weights=(len(y)-pos)/pos
    pos_weight=torch.as_tensor(weights,device=DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=.0005,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=2,min_lr=.00001)
    scaler=torch.amp.GradScaler("cuda",enabled=DEVICE.type=="cuda")
    best=float("inf");bad=0;history=[];started=time.time()
    for epoch in range(1,epochs+1):
        model.train();perm=np.random.permutation(len(tr));total=0
        for start in range(0,len(tr),1024):
            ids=perm[start:start+1024]
            target=torch.as_tensor(y[ids],device=DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type,dtype=torch.float16,enabled=DEVICE.type=="cuda"):
                z=model(batch_x(data,tr[ids],delta,cols))
                bce=nn.functional.binary_cross_entropy_with_logits(z,target,pos_weight=pos_weight)
                probability=z.sigmoid()
                loss=bce+.05*torch.relu(probability[:,1:]-probability[:,:-1]).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt);scaler.update()
            total+=float(loss.detach())*len(ids)
        valz=predict(model,data,va,delta,cols)
        val_loss=float(np.mean(np.logaddexp(0,valz.astype(float))-vy*valz))
        if not np.isfinite(val_loss):raise ValueError("Nonfinite validation loss")
        scheduler.step(val_loss)
        row=dict(epoch=epoch,train_loss=total/len(tr),selection_bce=val_loss,
                 learning_rate=opt.param_groups[0]["lr"],elapsed_seconds=time.time()-started)
        history.append(row)
        (run/"history.json").write_text(json.dumps(history,indent=2))
        if val_loss<best:
            best=val_loss;bad=0
            torch.save(dict(state_dict=model.state_dict(),kind=kind,h=h,delta=delta,seed=seed,
                            cols=cols,mean=data.mean,std=data.std,best_epoch=epoch,
                            best_selection_bce=best,feature_names=FEATURES),run/"best.pt")
        else:bad+=1
        print(f"{run.name} epoch={epoch} selection_BCE={val_loss:.6f} seconds={time.time()-started:.1f}",flush=True)
        if bad>=8:break
    saved=torch.load(run/"best.pt",map_location=DEVICE,weights_only=False)
    model.load_state_dict(saved["state_dict"])
    valz=predict(model,data,va,delta,cols)
    calz=predict(model,data,ca,delta,cols)
    params=fit_calibration(calz,cy)
    calp=calibrated_probabilities(calz,params)
    cutoffs=select_cutoffs(cy,calp)
    np.savez_compressed(run/"selection.npz",time=data.time[va],group=data.group[va],y=vy,logits=valz)
    np.savez_compressed(run/"calibration.npz",time=data.time[ca],group=data.group[ca],y=cy,logits=calz,p=calp)
    metadata=dict(kind=kind,h=h,delta=delta,seed=seed,train_N=len(tr),selection_N=len(va),calibration_N=len(ca),
                  class_weights=weights.tolist(),train_prevalence=y.mean(axis=0).tolist(),
                  calibration=params,secondary_cutoffs=cutoffs,best_epoch=saved["best_epoch"],
                  best_selection_bce=best,actual_epochs=len(history),elapsed_seconds=time.time()-started,
                  protocol=experiment_protocol(),protocol_sha256=protocol_hash(),
                  device=str(DEVICE),torch_version=torch.__version__,smoke_only=smoke)
    (run/"development_complete.json").write_text(json.dumps(metadata,indent=2))
    del model
    torch.cuda.empty_cache()


def infer(data,run):
    metadata=json.loads((run/"development_complete.json").read_text())
    saved=torch.load(run/"best.pt",map_location=DEVICE,weights_only=False)
    # Reapply training-only preprocessing exactly, even if new test data exist.
    data.normalize(saved["mean"],saved["std"])
    model=model_for(saved["kind"],len(saved["cols"])*2)
    model.load_state_dict(saved["state_dict"])
    for split in ["test","continuous"]:
        target=run/f"{split}.npz"
        if target.exists():continue
        ix,y=data.indices(saved["h"],saved["delta"],split)
        if not len(ix):raise ValueError(f"Missing {split} samples")
        logits=predict(model,data,ix,saved["delta"],saved["cols"])
        p=calibrated_probabilities(logits,metadata["calibration"])
        np.savez_compressed(target,time=data.time[ix],group=data.group[ix],y=y,logits=logits,p=p)
        print(f"{run.name} {split}: {len(ix)} predictions",flush=True)
    del model
    torch.cuda.empty_cache()


def logistic(data,delta):
    run=ROOT/"runs"/f"logistic_h0_d{delta}"
    if (run/"development_complete.json").exists() and (run/"continuous.npz").exists():return
    run.mkdir(parents=True,exist_ok=True)
    integral_index=FEATURES.index("Bz_south_integral_30min")
    def features(ix):
        latest=data.x[ix,integral_index]
        mat=np.column_stack([latest,data.x[ix-delta,integral_index]]) if delta else latest[:,None]
        return np.log1p(mat)
    tr,y=data.indices(0,delta,"train",5)
    ca,cy=data.indices(0,delta,"calibration")
    tx=features(tr);mean=tx.mean(axis=0);std=np.maximum(tx.std(axis=0),1e-6)
    models=[];cal=[]
    for j in range(4):
        m=LogisticRegression(C=1.,solver="lbfgs",max_iter=2000)
        m.fit((tx-mean)/std,y[:,j])
        if m.n_iter_[0]>=2000:raise RuntimeError("Logistic did not converge")
        models.append(m)
        cal.append(m.decision_function((features(ca)-mean)/std))
    calz=np.column_stack(cal);params=fit_calibration(calz,cy)
    calp=calibrated_probabilities(calz,params)
    np.savez_compressed(run/"calibration.npz",time=data.time[ca],group=data.group[ca],y=cy,logits=calz,p=calp)
    for split in ["test","continuous"]:
        ix,sy=data.indices(0,delta,split)
        z=np.column_stack([m.decision_function((features(ix)-mean)/std) for m in models])
        p=calibrated_probabilities(z,params)
        np.savez_compressed(run/f"{split}.npz",time=data.time[ix],group=data.group[ix],y=sy,logits=z,p=p)
    metadata=dict(kind="logistic",h=0,delta=delta,train_N=len(tr),calibration_N=len(ca),
                  train_prevalence=y.mean(axis=0).tolist(),calibration=params,
                  secondary_cutoffs=select_cutoffs(cy,calp),
                  mean=mean.tolist(),std=std.tolist(),
                  coefficients=[m.coef_.tolist() for m in models],intercepts=[m.intercept_.tolist() for m in models])
    (run/"development_complete.json").write_text(json.dumps(metadata,indent=2))


def experiment_protocol():
    """Record executable experiment settings and identify newly generated runs."""
    return {
        "provenance": "Forecast-GIC executable experiment configuration",
        "thresholds_A": [3, 5, 10, 20],
        "target": "maximum absolute native-minute GIC in [t+h,t+h+30min), peak >= threshold",
        "features": FEATURES,
        "input": {"minutes": 30, "streams": "latest and delta-delayed; duplicate latest at delta=0"},
        "alignment": {"mfi_age_seconds": 60, "swe_guard_seconds": 120, "swe_guarded_age_seconds": 120,
                      "strictly_prior": True, "interpolation": False},
        "blocks": {"train": [1, 30], "selection": [31, 35], "calibration": [36, 40], "test": [41, 49]},
        "continuous_year": 2022,
        "horizons_minutes": [0, 30, 45, 60, 90],
        "context_delays_minutes": [0, 30, 45, 60, 90],
        "search_rule": "minimum BCE on common selection issues; smaller delay breaks ties",
        "normalization": "signed log1p then in-block training mean/std; std floor=1e-6",
        "architecture": {"cnn_channels": 32, "kernel": 3, "dilations": [1, 4], "lstm_hidden": 64,
                         "lstm_layers": 2, "attention_hidden": 32, "dropout": 0.35, "outputs": 4},
        "training": {"seed": 42, "repeat_seeds": [7, 2026], "optimizer": "Adam", "learning_rate": 0.0005,
                     "weight_decay": 0.0005, "batch_size": 1024, "max_epochs": 30, "patience": 8,
                     "gradient_norm_clip": 1, "training_stride_minutes": 5, "evaluation_stride_minutes": 1,
                     "scheduler": {"factor": 0.5, "patience": 2, "min_lr": 0.00001},
                     "loss": "positive-weighted BCE +0.05 mean ordering violation; no onset head"},
        "calibration": {"rule": "positive-slope affine sigmoid; exact decreasing isotonic projection",
                        "regularizer": 0.00001, "alpha_bounds": [-4, 4], "intercept_bounds": [-12, 12]},
        "primary_cutoff": 0.5,
        "secondary_cutoff": "per-threshold calibration CSI maximum",
        "logistic_baseline": {"feature": "latest and delayed log1p southward integral", "C": 1,
                              "solver": "lbfgs", "max_iter": 2000},
        "bootstrap": {"replicates": 2000, "seed": 617, "units": "event blocks or UTC days"},
        "interpretation": {"background": 64, "max_sequences_per_stratum": 64, "strata": 5,
                           "draws": 200, "seed": 628, "mi_neighbours": 3},
    }


def protocol_hash():
    encoded = json.dumps(experiment_protocol(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def predict_from_features(checkpoint, input_path, output_path):
    """Issue calibrated CNN forecasts using only upstream x/time arrays."""
    checkpoint, input_path, output_path = map(Path, (checkpoint, input_path, output_path))
    saved = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    if saved["kind"] not in {"fb", "fa", "no_integral"}:
        raise ValueError("Feature-only prediction accepts CNN-BiLSTM checkpoints.")
    metadata = json.loads(checkpoint.with_name("development_complete.json").read_text(encoding="utf-8"))
    with np.load(input_path) as arrays:
        x, times = arrays["x"].astype(np.float32), arrays["time"].astype(np.int64)
    if x.shape != (len(times), len(FEATURES)):
        raise ValueError(f"Expected N x {len(FEATURES)} upstream features in the documented order.")
    if len(times) < 30 or not np.all(np.diff(times) > 0):
        raise ValueError("Expected at least 30 strictly increasing UTC nanosecond timestamps.")
    if saved.get("feature_names", FEATURES) != FEATURES:
        raise ValueError("Checkpoint feature order differs from this source version.")
    delta, horizon = int(saved["delta"]), int(saved["h"])
    good = np.isfinite(x).all(axis=1)
    complete = pd.Series(good.astype(int)).rolling(30, min_periods=30).sum().eq(30).to_numpy()
    # Thirty observations contain 29 minute intervals. This check handles
    # irregular feature-only inputs without bridging gaps or losing an extra row.
    contiguous = np.zeros(len(times), bool)
    contiguous[29:] = times[29:] - times[:-29] == 29 * 60_000_000_000
    complete &= contiguous
    eligible = complete.copy()
    if delta:
        earlier = np.zeros(len(x), bool)
        exact_delay = np.zeros(len(x), bool)
        if delta < len(x):
            earlier[delta:] = complete[:-delta]
            exact_delay[delta:] = times[delta:] - times[:-delta] == delta * 60_000_000_000
        eligible &= earlier & exact_delay
    indices = np.flatnonzero(eligible)
    cols = saved["cols"]
    model = model_for(saved["kind"], len(cols) * 2).eval()
    model.load_state_dict(saved["state_dict"])
    values = (np.sign(x) * np.log1p(abs(x)) - saved["mean"]) / saved["std"]
    tensor = torch.as_tensor(np.nan_to_num(values, nan=0, posinf=0, neginf=0), device=DEVICE)
    logits = []
    with torch.no_grad():
        for start in range(0, len(indices), 2048):
            at = torch.as_tensor(indices[start:start + 2048], device=DEVICE)
            window = at[:, None] + torch.arange(-29, 1, device=DEVICE)
            batch = torch.cat([tensor[window][:, :, cols], tensor[window - delta][:, :, cols]], dim=2)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                logits.append(model(batch).float().cpu().numpy())
    probabilities = calibrated_probabilities(np.concatenate(logits), metadata["calibration"]) if logits else np.empty((0, 4))
    frame = pd.DataFrame({"issue_time": pd.to_datetime(times), "available": eligible})
    frame["target_start"] = frame.issue_time + pd.Timedelta(minutes=horizon)
    frame["target_end_exclusive"] = frame.target_start + pd.Timedelta(minutes=30)
    for j, threshold in enumerate(THRESHOLDS):
        column = f"probability_exceeds_{threshold}A_30min"
        frame[column] = np.nan
        frame.loc[indices, column] = probabilities[:, j]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def main():
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "search", "selected", "infer", "predict"], required=True)
    parser.add_argument("--workspace", type=Path, help="Local data/run root; default is the repository root.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint", type=Path, help="For --phase predict: CNN checkpoint.")
    parser.add_argument("--features", type=Path, help="For --phase predict: NPZ with x and time only.")
    parser.add_argument("--output", type=Path, help="For --phase predict: local prediction CSV.")
    args = parser.parse_args()
    configure_workspace(args.workspace)
    DEVICE = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    verification.DEVICE = DEVICE
    if args.phase == "predict":
        if any(value is None for value in (args.checkpoint, args.features, args.output)):
            parser.error("--phase predict requires --checkpoint, --features and --output.")
        paths = [p if p.is_absolute() else ROOT / p for p in (args.checkpoint, args.features, args.output)]
        frame = predict_from_features(*paths)
        print(f"{int(frame.available.sum())} forecasts; {int((~frame.available).sum())} unavailable input histories.")
        return
    if args.phase == "smoke":
        data = Data(available=True)
        data.normalize()
        train(data, "fb", 0, 0, 42, epochs=2, smoke=True)
        return
    if args.phase in {"search", "selected"}:
        data = Data()
        data.normalize()
        if args.phase == "search":
            for horizon in [0, 30, 45, 60, 90]:
                train(data, "fb", horizon, 0, 42)
            for delta in [30, 45, 60, 90]:
                train(data, "fb", 0, delta, 42)
            scores = []
            for delta in [0, 30, 45, 60, 90]:
                path = ROOT / "runs" / name("fb", 0, delta, 42) / "development_complete.json"
                metadata = json.loads(path.read_text(encoding="utf-8"))
                scores.append((metadata["best_selection_bce"], delta))
            chosen = min(scores)[1]
            (ROOT / "selection.json").write_text(json.dumps(dict(
                delta=chosen, scores=scores,
                rule="Minimum BCE on common selection-block issue times; no test scores used; smaller delay breaks ties"
            ), indent=2), encoding="utf-8")
        else:
            delta = json.loads((ROOT / "selection.json").read_text(encoding="utf-8"))["delta"]
            for kind in ["fa", "lstm", "no_integral"]:
                train(data, kind, 0, delta, 42)
            for seed in [7, 2026]:
                train(data, "fb", 0, delta, seed)
    if args.phase == "infer":
        data = Data(development=False)
        for run in sorted((ROOT / "runs").iterdir()):
            if (run / "best.pt").exists():
                infer(data, run)
        delta = json.loads((ROOT / "selection.json").read_text(encoding="utf-8"))["delta"]
        logistic(data, delta)


if __name__ == "__main__":
    main()
