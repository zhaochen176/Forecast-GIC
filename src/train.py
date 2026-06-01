"""
GIC 预测项目 - 训练模块 (v6 峰值专家架构)

v6 核心:
  ★ Peak Expert Head: 独立 MLP + Softplus, 直接预测峰值幅度
  ★ Gate 机制: output = median + sigmoid(peak_logit) × relu(expert - median)
  ★ 损失 = Pinball(全样本) + BCE(pos_weight=20) + Expert_MSE(相对误差, 幅度加权)
  ★ 所有指标聚焦 ≥10A 峰值
"""
import os
import time
import warnings
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from typing import List, Dict, Optional
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable
import torch.nn.functional as F

from src.config import (
    DEVICE, QUANTILES, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, GRAD_CLIP, MODEL_DIR, SEED, WARMUP_EPOCHS,
    PEAK_CLS_WEIGHT, USE_AMP, AMP_DTYPE, LOG_TRANSFORM_TARGET,
    PEAK_MSE_WEIGHT, PEAK_MSE_THRESHOLD, PEAK_MSE_UNDER_RATIO,
    PEAK_VAL_THRESHOLD, TARGET_ROBUST_SCALE, PEAK_CLS_POS_WEIGHT,
    FUSED_POINT_LOSS_WEIGHT, FUSED_POINT_PEAK_BOOST,
    PEAK_TIER_WEIGHTS,
    POINT_PREDICTION_MODE,
    POINT_BLEND_LOW_Q,
    POINT_BLEND_HIGH_Q,
    POINT_BLEND_MAX_ALPHA,
    POINT_BLEND_LOGIT_SCALE,
    POINT_BLEND_LOGIT_BIAS,
    POINT_REG_LOSS_TYPE,
    POINT_HUBER_DELTA,
    POINT_HUBER_WEIGHT,
    POINT_HUBER_UNDER_RATIO,
    POINT_HUBER_USE_TIER_WEIGHT,
)


class PeakFocusedLoss(nn.Module):
    """
    v6 峰值专家损失。

    = Pinball 分位数损失 (所有样本)
    + BCE 峰值分类 (加强, pos_weight=20, 补0.05%的峰值比例)
    + ★ Peak Expert MSE (独立头, >5A, 在原始空间直接回归)
    """

    def __init__(self, quantiles=QUANTILES,
                 peak_cls_weight=PEAK_CLS_WEIGHT,
                 peak_cls_pos_weight=PEAK_CLS_POS_WEIGHT,
                 fused_point_loss_weight=FUSED_POINT_LOSS_WEIGHT,
                 fused_point_peak_boost=FUSED_POINT_PEAK_BOOST,
                 peak_mse_weight=PEAK_MSE_WEIGHT,
                 peak_mse_threshold=PEAK_MSE_THRESHOLD,
                 peak_mse_under_ratio=PEAK_MSE_UNDER_RATIO,
                 target_scaler=None):
        super().__init__()
        self.quantiles = quantiles
        self.peak_cls_weight = peak_cls_weight
        self.fused_point_loss_weight = fused_point_loss_weight
        self.fused_point_peak_boost = fused_point_peak_boost
        self.peak_mse_weight = peak_mse_weight
        self.peak_mse_threshold = peak_mse_threshold
        self.peak_mse_under_ratio = peak_mse_under_ratio
        self.target_scaler = target_scaler
        self.median_idx = len(quantiles) // 2
        self.register_buffer(
            'q_tensor',
            torch.tensor(quantiles, dtype=torch.float32).unsqueeze(0)
        )
        self.register_buffer(
            'bce_pos_weight',
            torch.tensor([peak_cls_pos_weight], dtype=torch.float32)
        )

    def _to_original(self, x):
        """如果用了 RobustScaler, 反变换。保留梯度流。"""
        if self.target_scaler is not None:
            center = torch.tensor(self.target_scaler.center_[0],
                                  device=x.device, dtype=x.dtype)
            scale = torch.tensor(self.target_scaler.scale_[0],
                                 device=x.device, dtype=x.dtype)
            return x * scale + center
        return x

    @staticmethod
    def _build_tier_weight(values: torch.Tensor) -> torch.Tensor:
        tw = torch.ones_like(values)
        w5 = float(PEAK_TIER_WEIGHTS.get(5, 1.0))
        w10 = float(PEAK_TIER_WEIGHTS.get(10, w5))
        w15 = float(PEAK_TIER_WEIGHTS.get(15, w10))
        w20 = float(PEAK_TIER_WEIGHTS.get(20, w15))
        tw = torch.where(values >= 5, tw * w5, tw)
        tw = torch.where(values >= 10, tw * (w10 / max(w5, 1e-6)), tw)
        tw = torch.where(values >= 15, tw * (w15 / max(w10, 1e-6)), tw)
        tw = torch.where(values >= 20, tw * (w20 / max(w15, 1e-6)), tw)
        return tw.detach()

    def forward(self, predictions, targets, weights=None,
                peak_logits=None, peak_expert_out=None,
                peak_threshold=None, point_pred=None):
        """
        predictions     : (B, Q) 分位数预测
        targets         : (B,)
        weights         : (B,)
        peak_logits     : (B, 1) 分类 logit
        peak_expert_out : (B, 1) ★ 峰值专家预测 (原始空间 A)
        peak_threshold  : float
        """
        # ── 1. Pinball 分位数损失 ──
        # ★ v7: 不加峰值权重! 让分位数头学习自然分布, 避免baseline被拉高
        targets_exp = targets.unsqueeze(1).expand_as(predictions)
        errors = targets_exp - predictions
        q = self.q_tensor.to(predictions.device).expand_as(errors)
        pinball = torch.where(errors >= 0, q * errors, (q - 1.0) * errors)
        sample_loss = pinball.mean(dim=1)
        # weights 仅用于采样, 不乘到 Pinball 上
        total_loss = sample_loss.mean()

        # ── 2. 峰值分类 BCE (pos_weight=20, 补偿类别不平衡) ──
        if peak_logits is not None and peak_threshold is not None:
            is_peak = (targets >= peak_threshold).float()
            cls_loss = F.binary_cross_entropy_with_logits(
                peak_logits.squeeze(-1), is_peak,
                pos_weight=self.bce_pos_weight.to(peak_logits.device))
            total_loss = total_loss + self.peak_cls_weight * cls_loss

        # ── 3. 直接监督最终门控点预测(与推理一致) ──
        point_loss_weight = (
            self.fused_point_loss_weight
            if POINT_REG_LOSS_TYPE == "smooth_l1"
            else float(POINT_HUBER_WEIGHT)
        )
        if point_pred is not None and point_loss_weight > 0:
            true_point = self._to_original(targets)
            pred_point = point_pred
            if self.target_scaler is not None:
                pred_point = self._to_original(pred_point)

            if POINT_REG_LOSS_TYPE == "tier_huber":
                err = pred_point - true_point
                abs_err = torch.abs(err)
                delta = float(max(POINT_HUBER_DELTA, 1e-6))
                point_base = torch.where(
                    abs_err <= delta,
                    0.5 * (err ** 2) / delta,
                    abs_err - 0.5 * delta,
                )
                point_w = torch.ones_like(point_base)
                if peak_threshold is not None:
                    is_peak = (targets >= peak_threshold).float()
                    point_w = point_w * (
                        1.0 + (self.fused_point_peak_boost - 1.0) * is_peak
                    )
                if POINT_HUBER_USE_TIER_WEIGHT:
                    point_w = point_w * self._build_tier_weight(true_point)
                if POINT_HUBER_UNDER_RATIO > 1.0:
                    under = (pred_point < true_point).float()
                    point_w = point_w * (
                        1.0 + (float(POINT_HUBER_UNDER_RATIO) - 1.0) * under
                    )
                point_loss = (point_base * point_w).mean()
            else:
                point_l1 = F.smooth_l1_loss(pred_point, true_point, reduction="none")
                if peak_threshold is not None:
                    is_peak = (targets >= peak_threshold).float()
                    point_w = 1.0 + (self.fused_point_peak_boost - 1.0) * is_peak
                    point_l1 = point_l1 * point_w
                point_loss = point_l1.mean()
            total_loss = total_loss + point_loss_weight * point_loss

        # ── 4. ★ v11 Peak Expert MSE (分段权重 + 强低估惩罚) ──
        if peak_expert_out is not None:
            y_true_orig = self._to_original(targets)
            expert_pred = peak_expert_out.squeeze(-1)  # 已经是原始空间
            peak_mask = y_true_orig > self.peak_mse_threshold
            if peak_mask.any():
                true_p = y_true_orig[peak_mask]
                pred_p = expert_pred[peak_mask]
                # ★ 相对误差 MSE: (pred/true - 1)² → 30A→15A 和 10A→5A 梯度相同
                rel_err = (pred_p - true_p) / (true_p + 1e-6)
                rel_mse = rel_err ** 2

                # ★ v11 核心改进: 分段权重 — 越高的峰值获得越大的损失权重
                # 使用config中的PEAK_TIER_WEIGHTS配置
                tier_weight = torch.ones_like(true_p)
                tier_weight = torch.where(
                    true_p >= 5, tier_weight * PEAK_TIER_WEIGHTS.get(5, 2.0), tier_weight)
                tier_weight = torch.where(
                    true_p >= 10, tier_weight * (PEAK_TIER_WEIGHTS.get(10, 8.0) / PEAK_TIER_WEIGHTS.get(5, 2.0)), tier_weight)
                tier_weight = torch.where(
                    true_p >= 15, tier_weight * (PEAK_TIER_WEIGHTS.get(15, 20.0) / PEAK_TIER_WEIGHTS.get(10, 8.0)), tier_weight)
                tier_weight = torch.where(
                    true_p >= 20, tier_weight * (PEAK_TIER_WEIGHTS.get(20, 50.0) / PEAK_TIER_WEIGHTS.get(15, 20.0)), tier_weight)
                tier_weight = tier_weight.detach()

                # ★ v11: 更强的低估惩罚 (低估越多惩罚指数增长)
                under = (pred_p < true_p).float()
                under_degree = torch.clamp(
                    (true_p - pred_p) / (true_p + 1e-6), 0, 1)
                # 低估惩罚: 1 + (ratio-1) * under * (1 + degree)^2 — 平方增长
                asym = 1.0 + (self.peak_mse_under_ratio - 1.0) * \
                    under * (1 + under_degree) ** 2

                expert_loss = (rel_mse * tier_weight * asym).mean()
                total_loss = total_loss + self.peak_mse_weight * expert_loss

        return total_loss


def set_seed(seed: int = SEED):
    """设置随机种子。GPU 服务器上使用 benchmark 模式加速固定尺寸输入。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('high')


def _atomic_torch_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path), suffix=".tmp", dir=os.path.dirname(path))
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _unpack_model_outputs(model_out):
    """
    Backward-compatible unpack:
    - legacy: (pred_q, attn, peak_logit, peak_expert_out)
    - new:    (pred_q, attn, peak_logit, peak_expert_out, point_reg_out)
    """
    if not isinstance(model_out, (tuple, list)):
        raise TypeError(f"Unexpected model output type: {type(model_out)}")
    if len(model_out) == 4:
        pred, attn, peak_logit, peak_expert_out = model_out
        point_reg_out = None
        return pred, attn, peak_logit, peak_expert_out, point_reg_out
    if len(model_out) >= 5:
        pred, attn, peak_logit, peak_expert_out, point_reg_out = model_out[:5]
        return pred, attn, peak_logit, peak_expert_out, point_reg_out
    raise ValueError(f"Unexpected model output tuple length: {len(model_out)}")


def _build_point_prediction(
    pred,
    peak_logit,
    peak_expert_out,
    point_reg_out=None,
    target_scaler=None,
):
    """Build the same point prediction used in inference."""
    def _q_idx(qv: float) -> int:
        if qv in QUANTILES:
            return QUANTILES.index(qv)
        arr = np.asarray(QUANTILES, dtype=np.float32)
        return int(np.argmin(np.abs(arr - float(qv))))

    q50 = pred[:, _q_idx(0.5)]

    # When target scaler is enabled, quantiles are in scaled space while expert head
    # stays in original space; avoid mixing different spaces.
    if target_scaler is not None:
        return q50

    if POINT_PREDICTION_MODE == "median":
        return q50

    if POINT_PREDICTION_MODE == "point_reg_head":
        if point_reg_out is None:
            return q50
        return point_reg_out.squeeze(-1)

    if POINT_PREDICTION_MODE == "quantile_blend":
        q_low = pred[:, _q_idx(POINT_BLEND_LOW_Q)]
        q_high = pred[:, _q_idx(POINT_BLEND_HIGH_Q)]
        blend_logit = peak_logit.squeeze(-1) * POINT_BLEND_LOGIT_SCALE + POINT_BLEND_LOGIT_BIAS
        alpha = torch.sigmoid(blend_logit) * POINT_BLEND_MAX_ALPHA
        return q_low + alpha * (q_high - q_low)

    gate = torch.sigmoid(peak_logit.squeeze(-1))
    expert = peak_expert_out.squeeze(-1)
    if POINT_PREDICTION_MODE == "gated_residual":
        return q50 + gate * (expert - q50)
    if POINT_PREDICTION_MODE == "gated_positive":
        return q50 + gate * torch.relu(expert - q50)
    raise ValueError(f"Unsupported POINT_PREDICTION_MODE={POINT_PREDICTION_MODE}")


def train_one_epoch(model, loader, criterion, optimizer, device,
                    grad_clip=GRAD_CLIP, peak_threshold=None,
                    scaler=None):
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    amp_ctx = torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE)

    for X, y, w in tqdm(loader, desc="  训练", leave=False):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            pred, _, peak_logit, peak_expert_out, point_reg_out = _unpack_model_outputs(model(X))
            point_pred = _build_point_prediction(
                pred,
                peak_logit,
                peak_expert_out,
                point_reg_out=point_reg_out,
                target_scaler=criterion.target_scaler,
            )
            loss = criterion(pred, y, w, peak_logit, peak_expert_out,
                             peak_threshold, point_pred=point_pred)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, peak_threshold=None):
    """验证。"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    amp_ctx = torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE)

    for X, y, w in tqdm(loader, desc="  验证", leave=False):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        with amp_ctx:
            pred, _, peak_logit, peak_expert_out, point_reg_out = _unpack_model_outputs(model(X))
            point_pred = _build_point_prediction(
                pred,
                peak_logit,
                peak_expert_out,
                point_reg_out=point_reg_out,
                target_scaler=criterion.target_scaler,
            )
            loss = criterion(pred, y, w, peak_logit, peak_expert_out,
                             peak_threshold, point_pred=point_pred)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_peak_score(model, loader, device,
                        peak_threshold_abs=3.0,
                        target_scaler=None):
    """
    ★ v11 验证集峰值指标 (≥3A), 用于模型选择和早停。
    返回: peak_capture, peak_mae, peak_mape, peak_score
    """
    model.eval()
    all_true, all_pred = [], []
    amp_ctx = torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE)

    for X, y, _ in loader:
        X = X.to(device, non_blocking=True)
        with amp_ctx:
            pred, _, peak_logit, peak_expert_out, point_reg_out = _unpack_model_outputs(model(X))
        # v6 门控输出: median + gate * relu(expert - median)
        point_pred = _build_point_prediction(
            pred,
            peak_logit,
            peak_expert_out,
            point_reg_out=point_reg_out,
            target_scaler=target_scaler,
        ).cpu().float()
        all_true.append(y.numpy())
        all_pred.append(point_pred.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    # 逆变换到原始空间
    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    # ≥5A 峰值指标
    pm = y_true >= peak_threshold_abs
    n_peak = pm.sum()
    if n_peak < 5:
        return {"peak_r2": -999.0, "peak_capture": 0.0, "peak_mae": 999.0,
                "peak_mape": 999.0, "peak_score": -999.0, "n_peak": int(n_peak)}

    errs = y_true[pm] - y_pred[pm]
    peak_mae = np.mean(np.abs(errs))
    peak_mape = np.mean(np.abs(errs) / (y_true[pm] + 1e-8)) * 100

    # 捕获率: 真实≥5A 中, 预测也≥阈值×0.8 的比例
    capture_thr = peak_threshold_abs * 0.8
    peak_capture = (y_pred[pm] >= capture_thr).sum() / n_peak

    # 综合分: 直接基于预测精度
    # 0.4*(1 - clamp(mae/mean_true)) + 0.3*capture + 0.3*(1 - clamp(mape/100))
    mean_true = np.mean(y_true[pm])
    mae_score = max(0, 1.0 - peak_mae / (mean_true + 1e-8))
    mape_score = max(0, 1.0 - peak_mape / 100)
    peak_score = 0.4 * mae_score + 0.3 * peak_capture + 0.3 * mape_score

    return {"peak_capture": float(peak_capture),
            "peak_mae": float(peak_mae), "peak_mape": float(peak_mape),
            "peak_score": float(peak_score), "n_peak": int(n_peak)}


class EarlyStopping:
    """早停 (maximize peak_score)。"""

    def __init__(self, patience=PATIENCE, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("-inf")
        self.should_stop = False

    def __call__(self, value):
        if value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    device: torch.device = DEVICE,
    model_save_path: Optional[str] = None,
    target_scaler=None,
    resume: bool = False,
    checkpoint_path: Optional[str] = None,
) -> Dict:
    """
    v5 单阶段训练。

    损失 = Pinball(全样本) + Peak_Classification(辅助) + ★ Peak_MSE(>5A, 原始空间)

    Parameters
    ----------
    resume : bool
        是否断点续训。如果为True，将自动查找 checkpoint_last.pt 并加载
        模型权重、优化器状态、调度器状态和训练历史，从上次中断的epoch继续。

    Returns
    -------
    history : dict  训练历史
    """
    set_seed()
    model = model.to(device)
    use_validation = (
        val_loader is not None
        and len(getattr(val_loader, "dataset", [])) > 0
    )

    if model_save_path is None:
        model_save_path = os.path.join(MODEL_DIR, "best_model.pt")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    peak_threshold = getattr(train_loader.dataset, 'peak_threshold', None)
    criterion = PeakFocusedLoss(QUANTILES, target_scaler=target_scaler)

    optimizer = AdamW(model.parameters(), lr=learning_rate,
                      weight_decay=weight_decay)

    # ★ GradScaler: float16 需要, bfloat16 不需要
    use_scaler = USE_AMP and (AMP_DTYPE == torch.float16)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    warmup_ep = min(WARMUP_EPOCHS, num_epochs // 3)
    if warmup_ep > 0 and num_epochs > warmup_ep:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_ep)
        cosine = CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_ep, eta_min=1e-6)
        scheduler = SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_ep])
    else:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6)

    early_stopping = EarlyStopping(patience=patience)
    peak_eval_thr = float(PEAK_VAL_THRESHOLD)

    history = {"train_losses": [], "val_losses": [], "peak_scores": [],
               "best_epoch": 0, "best_peak_score": float("-inf")}

    start_epoch = 1
    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODEL_DIR, "checkpoint_last.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # ★ 断点续训: 加载上次的 checkpoint
    if resume and os.path.exists(checkpoint_path):
        print(f"\n[断点续训] 加载 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if "history" in ckpt:
            history = ckpt["history"]
        if "early_stopping_counter" in ckpt:
            early_stopping.counter = ckpt["early_stopping_counter"]
            early_stopping.best = history["best_peak_score"]
        print(f"[断点续训] 从 epoch {start_epoch}/{num_epochs} 继续, "
              f"已有最优 peak_score: {history['best_peak_score']:.4f}, "
              f"早停计数: {early_stopping.counter}/{patience}")
    elif resume:
        print(f"[断点续训] 未找到 checkpoint_last.pt, 从头开始训练")

    # 构建分段权重描述字符串
    tier_desc = ", ".join([f"{k}-{k+5 if k < 20 else 30}A×{int(v)}"
                           for k, v in sorted(PEAK_TIER_WEIGHTS.items())])

    print(f"\n{'='*60}")
    print(f"[训练] v12 双专家架构 | 设备: {device}")
    print(f"  AMP: {USE_AMP} ({AMP_DTYPE}), Batch: {train_loader.batch_size}")
    print(f"  Epochs: {start_epoch}-{num_epochs}, LR: {learning_rate}")
    if not use_validation:
        print("  Validation: disabled; checkpoint selection uses minimum training loss.")
    print(f"  Fused point loss: weight={FUSED_POINT_LOSS_WEIGHT}, "
          f"peak_boost={FUSED_POINT_PEAK_BOOST}")
    if POINT_PREDICTION_MODE == "quantile_blend":
        print(
            "  Point head: quantile_blend "
            f"(q_low={POINT_BLEND_LOW_Q}, q_high={POINT_BLEND_HIGH_Q}, "
            f"alpha_max={POINT_BLEND_MAX_ALPHA}, "
            f"logit_scale={POINT_BLEND_LOGIT_SCALE}, "
            f"logit_bias={POINT_BLEND_LOGIT_BIAS})"
        )
    else:
        print(f"  Point head: {POINT_PREDICTION_MODE}")
    if POINT_REG_LOSS_TYPE == "tier_huber":
        print(
            f"  Point loss: tier_huber weight={POINT_HUBER_WEIGHT}, "
            f"delta={POINT_HUBER_DELTA}, "
            f"under_ratio={POINT_HUBER_UNDER_RATIO}, "
            f"use_tier_weight={POINT_HUBER_USE_TIER_WEIGHT}"
        )
    else:
        print(f"  Point loss: smooth_l1 weight={FUSED_POINT_LOSS_WEIGHT}")
    print(f"  Peak MSE: weight={PEAK_MSE_WEIGHT}, threshold={PEAK_MSE_THRESHOLD}A, "
          f"under_ratio={PEAK_MSE_UNDER_RATIO}")
    print(f"  分段权重: {tier_desc}")
    print(f"  评估阈值: >={peak_eval_thr}A")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            peak_threshold=peak_threshold, scaler=grad_scaler)
        if use_validation:
            val_loss = validate(
                model, val_loader, criterion, device,
                peak_threshold=peak_threshold)

            pm = validate_peak_score(
                model, val_loader, device, peak_eval_thr,
                target_scaler=target_scaler)
            peak_score = pm["peak_score"]
            selection_score = peak_score
        else:
            val_loss = float("nan")
            pm = {
                "peak_mae": float("nan"),
                "peak_mape": float("nan"),
                "peak_capture": float("nan"),
                "peak_score": -float(train_loss),
            }
            peak_score = pm["peak_score"]
            selection_score = -float(train_loss)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*epoch parameter.*deprecated.*")
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["peak_scores"].append(peak_score)

        mark = ""
        if selection_score > history["best_peak_score"]:
            history["best_peak_score"] = selection_score
            history["best_epoch"] = epoch
            _atomic_torch_save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "peak_score": peak_score,
                "peak_metrics": pm,
                "history": history,
                "early_stopping_counter": 0,
            }, model_save_path)
            mark = " ★ BEST"

        # ★ 每个 epoch 保存 checkpoint (用于断点续训)
        _atomic_torch_save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "peak_score": peak_score,
            "history": history,
            "early_stopping_counter": early_stopping.counter,
        }, checkpoint_path)

        print(f"[Train] Ep {epoch:3d}/{num_epochs} | "
              f"Loss: {train_loss:.5f}/{val_loss:.5f} | "
              f"PkMAE: {pm['peak_mae']:.2f}A "
              f"MAPE: {pm['peak_mape']:.1f}% "
              f"Cap: {pm['peak_capture']:.3f} "
              f"Score: {peak_score:.4f} | "
              f"LR: {lr:.2e} | {dt:.0f}s{mark}")

        early_stopping(peak_score)
        if early_stopping.should_stop:
            print(f"\n[早停] peak_score 已 {patience} 个 epoch 未改善")
            break

    # 加载最优模型
    checkpoint = torch.load(
        model_save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\n[训练完成] 最优 epoch: {history['best_epoch']}, "
          f"peak_score: {history['best_peak_score']:.4f}")
    print(f"[模型保存] {model_save_path}")

    return history
