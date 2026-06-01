"""
GIC 棰勬祴椤圭洰 - 浜旀ā鍨嬪姣旀ā鍧?

妯″瀷涓€: 鍩轰簬 LSTM 鐨勯娴?(涓嶅姞鐗瑰緛宸ョ▼鍜屾敞鎰忓姏鏈哄埗, 鐐归娴?
妯″瀷浜? 鍩轰簬鐗瑰緛宸ョ▼涓嬬殑 CNN-LSTM 棰勬祴 (鐐归娴?
妯″瀷涓? 鍩轰簬鐗瑰緛宸ョ▼鐨?CNN-BiLSTM-Attention 棰勬祴妯″瀷 (鍒嗙被浜嬩欢棰勬祴)
妯″瀷鍥? 鍦ㄤ笁鐨勫熀纭€鍔犱笂绠€鍗曞姞娉曡瀺鍚?(鍒嗙被浜嬩欢棰勬祴)
妯″瀷浜? 鍦ㄥ洓鐨勫熀纭€鍔犱笂闂ㄦ帶铻嶅悎鏈哄埗铻嶅悎 (鍒嗙被浜嬩欢棰勬祴) 鈥?褰撳墠鏈€浼?

鎵€鏈夋ā鍨嬬粺涓€鎺ュ彛:
    forward(x) -> (quantile_preds, attn_weights, peak_logit, peak_expert_out)
    - 妯″瀷涓€浜屾棤鍒嗙被/涓撳澶? peak_logit=zeros, peak_expert_out=zeros
    - 妯″瀷涓夋湁鍒嗙被浣嗘棤涓撳铻嶅悎
    - 妯″瀷鍥涙湁鍒嗙被+鍔犳硶铻嶅悎
    - 妯″瀷浜旀湁鍒嗙被+闂ㄦ帶铻嶅悎 (瀹屾暣鐗?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

from src.config import QUANTILES, MODEL_PARAMS, SEQ_LEN


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
#  鍏辩敤缁勪欢
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

class MultiHeadAttention(nn.Module):
    """澶氬ご鑷敞鎰忓姏灞傘€?"""

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        output = self.W_o(context)
        return output, attn_weights


class Conv1DFeatureExtractor(nn.Module):
    """1D 鍗风Н鐗瑰緛鎻愬彇鍣ㄣ€?"""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        residual = x_t
        out = self.dropout(self.act(self.bn1(self.conv1(x_t))))
        out = self.dropout(self.act(self.bn2(self.conv2(out))))
        out = out + residual
        return out.transpose(1, 2)


def _enforce_monotonicity(q_pred, nonnegative: bool = True):
    """纭繚鍒嗕綅鏁伴娴嬪€煎崟璋冮€掑銆?"""
    result = torch.zeros_like(q_pred)
    if nonnegative:
        result[:, 0] = F.softplus(q_pred[:, 0])
    else:
        # Signed targets can be negative; keep q1 unconstrained.
        result[:, 0] = q_pred[:, 0]
    for i in range(1, q_pred.shape[1]):
        result[:, i] = result[:, i - 1] + F.softplus(q_pred[:, i] - q_pred[:, i - 1])
    return result


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
#  妯″瀷涓€: 鍩虹 LSTM (鏃犵壒寰佸伐绋? 鏃犳敞鎰忓姏, 绾偣棰勬祴)
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

class GICModel1_LSTM(nn.Module):
    """
    妯″瀷涓€: 鍩虹 LSTM 棰勬祴妯″瀷銆?
    - 鍗曞悜 LSTM
    - 鏃?CNN, 鏃?Attention
    - 鍒嗕綅鏁拌緭鍑?(鍏煎鎺ュ彛), 鏃犲垎绫?涓撳澶?
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = MODEL_PARAMS["hidden_size"],
        num_layers: int = MODEL_PARAMS["num_layers"],
        dropout: float = MODEL_PARAMS["dropout"],
        quantiles: List[float] = QUANTILES,
        output_nonnegative: bool = True,
        **kwargs,  # 鍚告敹澶氫綑鍙傛暟
    ):
        super().__init__()
        self.quantiles = quantiles
        self.hidden_size = hidden_size
        self.has_classification = False
        self.model_name = "妯″瀷涓€:LSTM"
        self.output_nonnegative = bool(output_nonnegative)

        # LSTM (鍗曞悜)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        # FC
        fc_hidden = MODEL_PARAMS["fc_hidden"]
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 鍒嗕綅鏁拌緭鍑哄ご
        self.quantile_heads = nn.ModuleList([
            nn.Linear(fc_hidden // 2, 1) for _ in quantiles
        ])

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        # 鍙栨渶鍚庢椂闂存
        last_step = lstm_out[:, -1, :]
        fc_out = self.fc(last_step)

        # 鍒嗕綅鏁拌緭鍑?
        quantile_outputs = [head(fc_out) for head in self.quantile_heads]
        output = torch.cat(quantile_outputs, dim=1)
        output = _enforce_monotonicity(output, nonnegative=self.output_nonnegative)

        B = x.shape[0]
        dummy_attn = torch.zeros(B, 1, 1, 1, device=x.device)
        dummy_logit = torch.zeros(B, 1, device=x.device)
        dummy_expert = torch.zeros(B, 1, device=x.device)

        return output, dummy_attn, dummy_logit, dummy_expert

    def predict_median(self, x):
        output, _, _, _ = self.forward(x)
        median_idx = self.quantiles.index(0.5)
        return output[:, median_idx]


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
#  妯″瀷浜? CNN-LSTM (鍔犵壒寰佸伐绋? 鐐归娴?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

class GICModel2_CNN_LSTM(nn.Module):
    """
    妯″瀷浜? CNN + LSTM 棰勬祴妯″瀷銆?
    - Conv1D 鎻愬彇灞€閮ㄦā寮?
    - 鍗曞悜 LSTM
    - 鍒嗕綅鏁拌緭鍑? 鏃犲垎绫?涓撳澶?
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = MODEL_PARAMS["hidden_size"],
        num_layers: int = MODEL_PARAMS["num_layers"],
        dropout: float = MODEL_PARAMS["dropout"],
        quantiles: List[float] = QUANTILES,
        output_nonnegative: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.hidden_size = hidden_size
        self.has_classification = False
        self.model_name = "妯″瀷浜?CNN-LSTM"
        self.output_nonnegative = bool(output_nonnegative)

        # CNN 鐗瑰緛鎻愬彇
        self.conv_extractor = Conv1DFeatureExtractor(input_size, dropout)

        # LSTM (鍗曞悜)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        # FC
        fc_hidden = MODEL_PARAMS["fc_hidden"]
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 鍒嗕綅鏁拌緭鍑哄ご
        self.quantile_heads = nn.ModuleList([
            nn.Linear(fc_hidden // 2, 1) for _ in quantiles
        ])

    def forward(self, x):
        # CNN
        x = self.conv_extractor(x)

        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        last_step = lstm_out[:, -1, :]
        fc_out = self.fc(last_step)

        # 鍒嗕綅鏁拌緭鍑?
        quantile_outputs = [head(fc_out) for head in self.quantile_heads]
        output = torch.cat(quantile_outputs, dim=1)
        output = _enforce_monotonicity(output, nonnegative=self.output_nonnegative)

        B = x.shape[0]
        dummy_attn = torch.zeros(B, 1, 1, 1, device=x.device)
        dummy_logit = torch.zeros(B, 1, device=x.device)
        dummy_expert = torch.zeros(B, 1, device=x.device)

        return output, dummy_attn, dummy_logit, dummy_expert

    def predict_median(self, x):
        output, _, _, _ = self.forward(x)
        median_idx = self.quantiles.index(0.5)
        return output[:, median_idx]


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
#  妯″瀷涓? CNN-BiLSTM-Attention + 鍒嗙被 (鏃犺瀺鍚?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

class GICModel6_BiLSTM(nn.Module):
    """BiLSTM quantile model without CNN or attention."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = MODEL_PARAMS["hidden_size"],
        num_layers: int = MODEL_PARAMS["num_layers"],
        dropout: float = MODEL_PARAMS["dropout"],
        quantiles: List[float] = QUANTILES,
        output_nonnegative: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.hidden_size = hidden_size
        self.has_classification = False
        self.model_name = "BiLSTM"
        self.output_nonnegative = bool(output_nonnegative)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        bilstm_size = hidden_size * 2
        self.layer_norm = nn.LayerNorm(bilstm_size)

        fc_hidden = MODEL_PARAMS["fc_hidden"]
        self.fc = nn.Sequential(
            nn.Linear(bilstm_size, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.quantile_heads = nn.ModuleList([
            nn.Linear(fc_hidden // 2, 1) for _ in quantiles
        ])

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        fc_out = self.fc(lstm_out[:, -1, :])
        quantile_outputs = [head(fc_out) for head in self.quantile_heads]
        output = torch.cat(quantile_outputs, dim=1)
        output = _enforce_monotonicity(output, nonnegative=self.output_nonnegative)

        batch_size = x.shape[0]
        dummy_attn = torch.zeros(batch_size, 1, 1, 1, device=x.device)
        dummy_logit = torch.zeros(batch_size, 1, device=x.device)
        dummy_expert = torch.zeros(batch_size, 1, device=x.device)
        return output, dummy_attn, dummy_logit, dummy_expert

    def predict_median(self, x):
        output, _, _, _ = self.forward(x)
        median_idx = self.quantiles.index(0.5)
        return output[:, median_idx]


class GICModel7_CNN_BiLSTM(nn.Module):
    """CNN-BiLSTM quantile model without attention."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = MODEL_PARAMS["hidden_size"],
        num_layers: int = MODEL_PARAMS["num_layers"],
        dropout: float = MODEL_PARAMS["dropout"],
        quantiles: List[float] = QUANTILES,
        output_nonnegative: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.hidden_size = hidden_size
        self.has_classification = False
        self.model_name = "CNN-BiLSTM"
        self.output_nonnegative = bool(output_nonnegative)

        self.conv_extractor = Conv1DFeatureExtractor(input_size, dropout)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        bilstm_size = hidden_size * 2
        self.layer_norm = nn.LayerNorm(bilstm_size)

        fc_hidden = MODEL_PARAMS["fc_hidden"]
        self.fc = nn.Sequential(
            nn.Linear(bilstm_size, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.quantile_heads = nn.ModuleList([
            nn.Linear(fc_hidden // 2, 1) for _ in quantiles
        ])

    def forward(self, x):
        x = self.conv_extractor(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        fc_out = self.fc(lstm_out[:, -1, :])
        quantile_outputs = [head(fc_out) for head in self.quantile_heads]
        output = torch.cat(quantile_outputs, dim=1)
        output = _enforce_monotonicity(output, nonnegative=self.output_nonnegative)

        batch_size = x.shape[0]
        dummy_attn = torch.zeros(batch_size, 1, 1, 1, device=x.device)
        dummy_logit = torch.zeros(batch_size, 1, device=x.device)
        dummy_expert = torch.zeros(batch_size, 1, device=x.device)
        return output, dummy_attn, dummy_logit, dummy_expert

    def predict_median(self, x):
        output, _, _, _ = self.forward(x)
        median_idx = self.quantiles.index(0.5)
        return output[:, median_idx]


class GICModel3_BiLSTM_Attn(nn.Module):
    """
    妯″瀷涓? CNN + BiLSTM + Multi-Head Attention + 鍒嗙被棰勬祴銆?
    - Conv1D + BiLSTM + Attention
    - 宄板€煎垎绫昏緟鍔╁ご (浜嬩欢棰勬祴)
    - 涓撳澶磋緭鍑轰絾涓嶅仛铻嶅悎 (浠呯嫭绔嬮娴?
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = MODEL_PARAMS["hidden_size"],
        num_layers: int = MODEL_PARAMS["num_layers"],
        dropout: float = MODEL_PARAMS["dropout"],
        attention_heads: int = MODEL_PARAMS["attention_heads"],
        fc_hidden: int = MODEL_PARAMS["fc_hidden"],
        quantiles: List[float] = QUANTILES,
        output_nonnegative: bool = True,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.hidden_size = hidden_size
        self.has_classification = True
        self.model_name = "妯″瀷涓?CNN-BiLSTM-Attention"
        self.output_nonnegative = bool(output_nonnegative)

        # CNN
        self.conv_extractor = Conv1DFeatureExtractor(input_size, dropout)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        lstm_out_size = hidden_size * 2
        self.layer_norm = nn.LayerNorm(lstm_out_size)

        # Attention
        self.attention = MultiHeadAttention(lstm_out_size, attention_heads, dropout)
        self.attn_layer_norm = nn.LayerNorm(lstm_out_size)

        # 鏃堕棿姹犲寲
        self.temporal_weight = nn.Linear(lstm_out_size, 1)

        # FC
        fc_input_size = lstm_out_size * 2
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 鍒嗕綅鏁拌緭鍑哄ご
        self.quantile_heads = nn.ModuleList([
            nn.Linear(fc_hidden // 2, 1) for _ in quantiles
        ])

        # 宄板€煎垎绫昏緟鍔╁ご
        self.peak_classifier = nn.Sequential(
            nn.Linear(fc_input_size, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, 1),
        )

        # 宄板€间笓瀹跺ご (鐙珛, 涓嶅弬涓庤瀺鍚?
        self.peak_expert = nn.Sequential(
            nn.Linear(fc_input_size, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(fc_hidden, 1),
            nn.Softplus(),
        )
        with torch.no_grad():
            self.peak_expert[-2].bias.fill_(10.0)

    def forward(self, x):
        x = self.conv_extractor(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        attn_out, attn_weights = self.attention(lstm_out)
        attn_out = self.attn_layer_norm(attn_out + lstm_out)

        tw = torch.softmax(self.temporal_weight(attn_out).squeeze(-1), dim=1)
        weighted_avg = torch.bmm(tw.unsqueeze(1), attn_out).squeeze(1)
        last_step = attn_out[:, -1, :]
        pooled = torch.cat([weighted_avg, last_step], dim=1)

        fc_out = self.fc(pooled)

        # 鍒嗕綅鏁拌緭鍑?
        quantile_outputs = [head(fc_out) for head in self.quantile_heads]
        output = torch.cat(quantile_outputs, dim=1)
        output = _enforce_monotonicity(output, nonnegative=self.output_nonnegative)

        # 鍒嗙被
        peak_logit = self.peak_classifier(pooled)
        peak_expert_out = self.peak_expert(pooled)

        return output, attn_weights, peak_logit, peak_expert_out

    def predict_median(self, x):
        """妯″瀷涓? 涓嶅仛铻嶅悎, 鐩存帴杈撳嚭涓綅鏁般€?"""
        output, _, _, _ = self.forward(x)
        median_idx = self.quantiles.index(0.5)
        return output[:, median_idx]


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
#  妯″瀷鍥? CNN-BiLSTM-Attention + 鍒嗙被 + 绠€鍗曞姞娉曡瀺鍚?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

class GICModel4_AddFusion(nn.Module):
    """
    妯″瀷鍥? 鍦ㄦā鍨嬩笁鐨勫熀纭€涓婂姞涓婄畝鍗曞姞娉曡瀺鍚堛€?
    - 涓撳杈撳嚭鐩存帴涓庝腑浣嶆暟鐩稿姞 (褰撳垎绫讳负宄板€兼椂)
    - output = median + sigmoid(peak_logit) * relu(expert - median)
    - 浣嗕笉浣跨敤闂ㄦ帶, sigmoid 鐩存帴浣滀负寮€鍏?
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = MODEL_PARAMS["hidden_size"],
        num_layers: int = MODEL_PARAMS["num_layers"],
        dropout: float = MODEL_PARAMS["dropout"],
        attention_heads: int = MODEL_PARAMS["attention_heads"],
        fc_hidden: int = MODEL_PARAMS["fc_hidden"],
        quantiles: List[float] = QUANTILES,
        output_nonnegative: bool = True,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.hidden_size = hidden_size
        self.has_classification = True
        self.model_name = "妯″瀷鍥?鍔犳硶铻嶅悎"
        self.output_nonnegative = bool(output_nonnegative)

        # CNN
        self.conv_extractor = Conv1DFeatureExtractor(input_size, dropout)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        lstm_out_size = hidden_size * 2
        self.layer_norm = nn.LayerNorm(lstm_out_size)

        # Attention
        self.attention = MultiHeadAttention(lstm_out_size, attention_heads, dropout)
        self.attn_layer_norm = nn.LayerNorm(lstm_out_size)

        # 鏃堕棿姹犲寲
        self.temporal_weight = nn.Linear(lstm_out_size, 1)

        # FC
        fc_input_size = lstm_out_size * 2
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 鍒嗕綅鏁拌緭鍑哄ご
        self.quantile_heads = nn.ModuleList([
            nn.Linear(fc_hidden // 2, 1) for _ in quantiles
        ])

        # 宄板€煎垎绫?
        self.peak_classifier = nn.Sequential(
            nn.Linear(fc_input_size, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, 1),
        )

        # 鍗曚笓瀹跺ご (绠€鍗曞姞娉曡瀺鍚?
        self.peak_expert = nn.Sequential(
            nn.Linear(fc_input_size, fc_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(fc_hidden * 2, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(fc_hidden, 1),
            nn.Softplus(),
        )
        with torch.no_grad():
            self.peak_expert[-2].bias.fill_(12.0)

    def forward(self, x):
        x = self.conv_extractor(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        attn_out, attn_weights = self.attention(lstm_out)
        attn_out = self.attn_layer_norm(attn_out + lstm_out)

        tw = torch.softmax(self.temporal_weight(attn_out).squeeze(-1), dim=1)
        weighted_avg = torch.bmm(tw.unsqueeze(1), attn_out).squeeze(1)
        last_step = attn_out[:, -1, :]
        pooled = torch.cat([weighted_avg, last_step], dim=1)

        fc_out = self.fc(pooled)

        # 鍒嗕綅鏁拌緭鍑?
        quantile_outputs = [head(fc_out) for head in self.quantile_heads]
        output = torch.cat(quantile_outputs, dim=1)
        output = _enforce_monotonicity(output, nonnegative=self.output_nonnegative)

        # 鍒嗙被 + 涓撳
        peak_logit = self.peak_classifier(pooled)
        peak_expert_out = self.peak_expert(pooled)

        return output, attn_weights, peak_logit, peak_expert_out

    def predict_median(self, x):
        """妯″瀷鍥? 绠€鍗曞姞娉曡瀺鍚?鈥?median + sigmoid(logit) * relu(expert - median)"""
        output, _, peak_logit, peak_expert_out, _ = self.forward(x)
        median_idx = self.quantiles.index(0.5)
        median = output[:, median_idx]
        gate = torch.sigmoid(peak_logit.squeeze(-1))
        expert = peak_expert_out.squeeze(-1)
        return median + gate * F.relu(expert - median)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
#  妯″瀷浜? CNN-BiLSTM-Attention + 鍒嗙被 + 闂ㄦ帶铻嶅悎 (瀹屾暣鐗?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

class GICModel5_GateFusion(nn.Module):
    """
    妯″瀷浜? 瀹屾暣闂ㄦ帶铻嶅悎妯″瀷 (褰撳墠鏈€浼?銆?
    - CNN + BiLSTM + Multi-Head Attention
    - 宄板€煎垎绫昏緟鍔╁ご
    - 鍙屼笓瀹跺ご (鍩虹涓撳 + 楂樺嘲鍊间笓瀹?
    - 涓撳闂ㄦ帶缃戠粶 (瀛︿範閫夋嫨鍝釜涓撳)
    - 闂ㄦ帶铻嶅悎: output = median + sigmoid(peak_logit) * relu(mixed_expert - median)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = MODEL_PARAMS["hidden_size"],
        num_layers: int = MODEL_PARAMS["num_layers"],
        dropout: float = MODEL_PARAMS["dropout"],
        attention_heads: int = MODEL_PARAMS["attention_heads"],
        fc_hidden: int = MODEL_PARAMS["fc_hidden"],
        quantiles: List[float] = QUANTILES,
        output_nonnegative: bool = True,
    ):
        super().__init__()
        self.quantiles = quantiles
        self.hidden_size = hidden_size
        self.has_classification = True
        self.model_name = "妯″瀷浜?闂ㄦ帶铻嶅悎"
        self.output_nonnegative = bool(output_nonnegative)

        # CNN
        self.conv_extractor = Conv1DFeatureExtractor(input_size, dropout)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        lstm_out_size = hidden_size * 2
        self.layer_norm = nn.LayerNorm(lstm_out_size)

        # Attention
        self.attention = MultiHeadAttention(lstm_out_size, attention_heads, dropout)
        self.attn_layer_norm = nn.LayerNorm(lstm_out_size)

        # 鏃堕棿姹犲寲
        self.temporal_weight = nn.Linear(lstm_out_size, 1)

        # FC
        fc_input_size = lstm_out_size * 2
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 鍒嗕綅鏁拌緭鍑哄ご
        self.quantile_heads = nn.ModuleList([
            nn.Linear(fc_hidden // 2, 1) for _ in quantiles
        ])

        # 宄板€煎垎绫?
        self.peak_classifier = nn.Sequential(
            nn.Linear(fc_input_size, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, 1),
        )

        # 鍙屼笓瀹舵灦鏋?
        self.peak_expert_base = nn.Sequential(
            nn.Linear(fc_input_size, fc_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(fc_hidden * 2, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(fc_hidden, 1),
            nn.Softplus(),
        )
        self.peak_expert_high = nn.Sequential(
            nn.Linear(fc_input_size, fc_hidden * 3),
            nn.GELU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(fc_hidden * 3, fc_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(fc_hidden * 2, 1),
            nn.Softplus(),
        )
        # 涓撳闂ㄦ帶
        self.expert_gate = nn.Sequential(
            nn.Linear(fc_input_size, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid(),
        )

        with torch.no_grad():
            self.peak_expert_base[-2].bias.fill_(8.0)
            self.peak_expert_high[-2].bias.fill_(20.0)

        # 鍏煎鎬?
        self.peak_expert = self.peak_expert_base

    def forward(self, x):
        x = self.conv_extractor(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        attn_out, attn_weights = self.attention(lstm_out)
        attn_out = self.attn_layer_norm(attn_out + lstm_out)

        tw = torch.softmax(self.temporal_weight(attn_out).squeeze(-1), dim=1)
        weighted_avg = torch.bmm(tw.unsqueeze(1), attn_out).squeeze(1)
        last_step = attn_out[:, -1, :]
        pooled = torch.cat([weighted_avg, last_step], dim=1)

        fc_out = self.fc(pooled)

        # 鍒嗕綅鏁拌緭鍑?
        quantile_outputs = [head(fc_out) for head in self.quantile_heads]
        output = torch.cat(quantile_outputs, dim=1)
        output = _enforce_monotonicity(output, nonnegative=self.output_nonnegative)

        # 鍒嗙被
        peak_logit = self.peak_classifier(pooled)

        # 鍙屼笓瀹?+ 闂ㄦ帶娣峰悎
        expert_base = self.peak_expert_base(pooled)
        expert_high = self.peak_expert_high(pooled)
        gate = self.expert_gate(pooled)
        peak_expert_out = expert_base * (1 - gate) + expert_high * gate

        return output, attn_weights, peak_logit, peak_expert_out

    def predict_median(self, x):
        """妯″瀷浜? 闂ㄦ帶铻嶅悎 鈥?median + sigmoid(logit) * relu(expert - median)"""
        output, _, peak_logit, peak_expert_out = self.forward(x)
        median_idx = self.quantiles.index(0.5)
        median = output[:, median_idx]
        gate = torch.sigmoid(peak_logit.squeeze(-1))
        expert = peak_expert_out.squeeze(-1)
        return median + gate * F.relu(expert - median)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
#  妯″瀷娉ㄥ唽涓庢瀯寤?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

MODEL_REGISTRY = {
    1: GICModel1_LSTM,
    2: GICModel2_CNN_LSTM,
    3: GICModel3_BiLSTM_Attn,
    4: GICModel4_AddFusion,
    5: GICModel5_GateFusion,
    6: GICModel6_BiLSTM,
    7: GICModel7_CNN_BiLSTM,
}

MODEL_NAMES = {
    1: "LSTM",
    2: "CNN-LSTM",
    3: "CNN-BiLSTM-Attention",
    4: "CNN-BiLSTM-Attention-AddFusion",
    5: "CNN-BiLSTM-Attention-GateFusion",
    6: "BiLSTM",
    7: "CNN-BiLSTM",
}


def build_model_by_id(
    model_id: int,
    input_size: int,
    quantiles: List[float] = QUANTILES,
    output_nonnegative: bool = True,
):
    """鏍规嵁妯″瀷 ID 鏋勫缓妯″瀷骞舵墦鍗板弬鏁伴噺銆?"""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"鏈煡妯″瀷 ID: {model_id}, 鍙€? {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[model_id]
    model = model_cls(
        input_size=input_size,
        quantiles=quantiles,
        output_nonnegative=bool(output_nonnegative),
        **MODEL_PARAMS,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] {MODEL_NAMES[model_id]} built")
    print(f"  input_size: {input_size}")
    print(f"  quantiles: {quantiles}")
    print(f"  trainable_params: {n_params:,}")
    return model

