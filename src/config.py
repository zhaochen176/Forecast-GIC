"""
GIC 棰勬祴椤圭洰 - 鍏ㄥ眬閰嶇疆
"""
import os
import torch

# ==================== 璺緞閰嶇疆 ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "experiments")

DATA_FILE = os.path.join(DATA_DIR, "merged_2012_2022_processed.parquet")
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "featured_data.parquet")
LOUKHI_CACHE_FILE = os.path.join(DATA_DIR, "loukhi_gic_2012_2022_1min.parquet")
LOUKHI_RAW_DIR = os.path.join(DATA_DIR, "loukhi_raw_files")
LOUKHI_BASE_URL = "http://gic.en51.ru/data/lkh_gic/"
LOUKHI_REQUEST_TIMEOUT = 10
LOUKHI_RETRY_TIMES = 3
LOUKHI_RETRY_SLEEP = 2.0
DATA_START = "2012-01-01 00:00:00"
# 蹇€熷疄楠屽厛鐢?2012-2014
DATA_END = "2022-12-31 23:59:00"

# 鍒涘缓杈撳嚭鐩綍
for d in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR, EXPERIMENT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 璁惧閰嶇疆 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 鐗瑰緛鍒楅厤缃?====================
# 澶槼椋庡弬鏁帮紙鐢ㄤ簬鏋勯€犳粸鍚庣壒寰佺殑鏍稿績鍒楋級
SOLAR_WIND_COLS = [
    "Btot", "Bx_gse", "By_gse", "Bz_gse",
    "Vp", "Np_filled", "P_dyn_nPa", "Ey_mV/m",
    "Ma", "epsilon_norm", "Newell", "Borovsky",
]
# 澶槼椋庢爣璁板垪锛堜笉鍋氭粸鍚庯級
SOLAR_WIND_FLAG_COLS = ["density_missing", "filled_by_model"]

# 鍦扮瑙傛祴鍒?
GEOMAG_COLS = [
    "X", "Y", "Z",
    "X_pert", "Y_pert", "Z_pert", "H_pert",
    "dX_pert_dt", "dY_pert_dt", "dZ_pert_dt", "dH_pert_dt",
]
GEOMAG_RAW_PERT_COLS = [
    "X", "Y", "Z",
    "X_pert", "Y_pert", "Z_pert", "H_pert",
]

# 目标列
TARGET_COL = "gic"
TARGET_VYK_RAW_COL = "gic_vyk_raw"
TARGET_LOU_RAW_COL = "gic_lou_raw"
TARGET_DBHDT_RAW_COL = "dbhdt_raw"
TARGET_VYK_COL = "gic_vyk_abs"
TARGET_LOU_COL = "gic_lou_abs"
TARGET_DBHDT_COL = "dbhdt_abs"
# Input feature for D group: absolute dBH/dt (kept separate from target columns).
TARGET_DBHDT_FEATURE_COL = "dbhdt_abs_feature"
TARGET_COLUMNS = [TARGET_VYK_COL, TARGET_LOU_COL, TARGET_DBHDT_COL]
RAW_TARGET_COLUMNS = [TARGET_VYK_RAW_COL, TARGET_LOU_RAW_COL, TARGET_DBHDT_RAW_COL]
DBHDT_SOURCE_CANDIDATES = ["dH_pert_dt", "dBH_dt", "dB_dt", "dH_dt"]
DEFAULT_STORM_TARGET_COL = TARGET_VYK_COL

# 鈽?v10: GIC 鎴柇涓婇檺 (瓒呰繃姝ゅ€兼埅鏂?鍑忓皯鏋佺鍊煎奖鍝?
TARGET_MAX_CLIP = 20.0
GIC_MAX_CLIP = TARGET_MAX_CLIP
USE_TARGET_CLIP = False

# 鎵€鏈夊師濮嬬壒寰佸垪
ALL_FEATURE_COLS = SOLAR_WIND_COLS + SOLAR_WIND_FLAG_COLS + GEOMAG_COLS

# 瀹為獙鐗瑰緛缁勫畾涔夛紙A/B/C锛?
SOLAR_WIND_RAW_COLS = [
    "Btot", "Bx_gse", "By_gse", "Bz_gse", "Vp", "Np_filled", "Ma",
]
SOLAR_WIND_COUPLING_COLS = [
    "P_dyn_nPa", "Ey_mV/m", "epsilon_norm", "Newell", "Borovsky",
]
FEATURE_SET_DEFINITIONS = {
    "A": SOLAR_WIND_RAW_COLS,
    "B": SOLAR_WIND_RAW_COLS + SOLAR_WIND_COUPLING_COLS,
    # C缁勪弗鏍奸檺鍒? 浠呭湴纾佸師濮嬪弬鏁?+ 鎵板姩, 涓嶅惈瀵兼暟
    "C": SOLAR_WIND_RAW_COLS + SOLAR_WIND_COUPLING_COLS + GEOMAG_RAW_PERT_COLS,
    # D group: C + |dBH/dt| feature
    "D": SOLAR_WIND_RAW_COLS + SOLAR_WIND_COUPLING_COLS + GEOMAG_RAW_PERT_COLS + [TARGET_DBHDT_FEATURE_COL],
}
EXPERIMENT_FEATURE_SETS = ["A", "B", "C", "D"]
EXPERIMENT_TARGETS = [TARGET_VYK_COL]
EXPERIMENT_HORIZONS = [30, 60, 90]

# Event-type conditional features. These are only populated by the grouped
# event-type experiment path; the default experiment path leaves them absent.
EVENT_TYPE_COL = "event_type"
EVENT_TYPE_RAW_COL = "event_type_raw"
EVENT_TYPE_FEATURE_COLS = [
    "event_type_CME",
    "event_type_CIR",
    "event_type_NO_WEAK",
    "event_type_SC_SI",
]
EVENT_TYPE_TRAIN_TYPES = ["CME", "CIR"]
EVENT_TYPE_SPLIT_RATIOS = (0.8, 0.2)

# ==================== 鐗瑰緛宸ョ▼閰嶇疆 ====================
# 澶槼椋庢粸鍚庡垎閽熸暟锛堝熀浜?0-60鍒嗛挓浼犳挱寤惰繜锛?
LAG_MINUTES = [30, 45, 60]

# 鐢ㄤ簬鏋勯€犳粸鍚庣壒寰佺殑澶槼椋庡叧閿垪
LAG_FEATURE_COLS = [
    "Btot", "Bz_gse", "Vp", "Np_filled",
    "P_dyn_nPa", "Ey_mV/m", "epsilon_norm", "Newell",
]

# 婊氬姩绐楀彛澶у皬锛堝垎閽燂級
ROLLING_WINDOWS = [15, 30, 60]

# 鐢ㄤ簬璁＄畻婊氬姩缁熻鐨勫垪
ROLLING_FEATURE_COLS = [
    "Bz_gse", "Vp", "P_dyn_nPa", "Ey_mV/m",
    "epsilon_norm", "Newell",
    "dX_pert_dt", "dH_pert_dt",
]

# 婊氬姩缁熻绫诲瀷
ROLLING_STATS = ["mean", "std", "max"]

# ==================== 鏁版嵁鍒掑垎閰嶇疆 ====================
# 鎸夋椂闂村垝鍒嗭紙鏃堕棿搴忓垪涓嶈兘闅忔満鍒掑垎锛?TRAIN_END = "2013-09-30 23:59:00"
VAL_END = "2014-03-31 23:59:00"
USE_RATIO_SPLIT = True
SPLIT_RATIOS = (0.8, 0.1, 0.1)

# ==================== 妯″瀷閰嶇疆 ====================
# 杈撳叆绐楀彛闀垮害锛堝垎閽燂級
SEQ_LEN = 120
# 棰勬祴姝ラ暱锛堥娴嬫湭鏉ョ鍑犲垎閽燂級
PRED_HORIZON = 1
# 璁粌鏃剁殑閲囨牱姝ラ暱锛堥鏆存椂娈垫暟鎹噺灏忥紝鐢ㄥ皬姝ラ暱锛?
TRAIN_STRIDE = 2
# 楠岃瘉/娴嬭瘯鏃剁殑閲囨牱姝ラ暱
EVAL_STRIDE = 1

# 鍒嗕綅鏁板垪琛紙鐢ㄤ簬姒傜巼鍖洪棿棰勬祴锛?
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

# 妯″瀷瓒呭弬鏁?
MODEL_PARAMS = {
    "hidden_size": 256,
    "num_layers": 3,
    "dropout": 0.3,
    "attention_heads": 8,
    "fc_hidden": 256,
}

# ==================== 璁粌閰嶇疆 ====================
BATCH_SIZE = 256                  # 鈽?RTX 4060 (8GB VRAM) 閫傞厤
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 30
PATIENCE = 10         # 鏃╁仠鑰愬績
GRAD_CLIP = 1.0       # 姊害瑁佸壀
WARMUP_EPOCHS = 3     # 瀛︿範鐜囬鐑疆鏁?

# 鐩爣鍙樻崲 鈥?v4: 褰诲簳鍘绘帀 log1p, 鍦ㄥ師濮嬬┖闂磋缁?
LOG_TRANSFORM_TARGET = False    # 鈽?涓嶅仛 log1p 鍙樻崲锛屼繚鐣欏嘲鍊肩殑鐪熷疄姊害淇″彿
TARGET_ROBUST_SCALE = False     # 鈽?v6: 鍘绘帀 RobustScaler, 鐩存帴鍦ㄥ師濮嬬┖闂磋缁?

# 宄板€兼潈閲嶉厤缃?
PEAK_THRESHOLD_QUANTILE = 0.85  # |GIC| 85鍒嗕綅浠ヤ笂瑙嗕负宄板€?(鍒嗙被鐢?
PEAK_WEIGHT = 8.0               # 鈽?v7: 宄板€兼潈閲嶇缉鏀剧郴鏁?(鈫?30鈫?, 鍑忓皯閲囨牱鍋忓樊)
NORMAL_WEIGHT = 1.0             # 鏅€氭牱鏈潈閲?
PEAK_STRIDE = 1                 # 宄板€兼牱鏈殑閲囨牱姝ラ暱
PEAK_LABEL_STRATEGY = "fixed"   # fixed | quantile
PEAK_LABEL_THRESHOLD_A = 3.0
PEAK_CLS_WEIGHT = 2.0           # 鈽?v6: 宄板€煎垎绫绘崯澶辨潈閲?(鈫戔啈 浠?.2鈫?.0)
PEAK_CLS_POS_WEIGHT = 20.0      # 鈽?v6: BCE 姝ｇ被鏉冮噸, 琛ュ伩鏋佺绫诲埆涓嶅钩琛?FUSED_POINT_LOSS_WEIGHT = 1.0   # 鏂板: 鐩存帴鐩戠潱鏈€缁堥棬鎺х偣棰勬祴
FUSED_POINT_PEAK_BOOST = 2.0    # 鏂板: 宄板€兼牱鏈偣棰勬祴鎹熷け鍔犳潈
PEAK_WEIGHT_THRESHOLD = 3.0     # 鈽?v11: 鏍锋湰鏉冮噸璁＄畻闃堝€?(A): >3A 寮€濮嬪澶ф潈閲?USE_WEIGHTED_SAMPLER = True     # 浣跨敤 WeightedRandomSampler
PEAK_WEIGHT = 12.0              # 鈽?v11: 宄板€奸噰鏍锋潈閲?(8鈫?2)
PEAK_SAMPLER_POWER = 1.2

# 娓╁拰鐗? 鍒嗘宄板€兼崯澶?(闄嶄綆瀵规瀬绔牱鏈殑婵€杩涙斁澶э紝鍏堢ǔ浣忔暣浣撴嫙鍚?
PEAK_MSE_WEIGHT = 20.0
PEAK_MSE_THRESHOLD = 4.0
PEAK_MSE_UNDER_RATIO = 4.0

PEAK_TIER_WEIGHTS = {
    3: 1.0,    # 3-5A
    5: 1.2,    # 5-10A
    10: 1.8,   # 10-15A
    15: 2.5,   # 15-20A
    20: 3.5,   # >=20A
}

# ==================== 椋庢毚鏃舵绛涢€?====================
STORM_FILTER = True             # 鏄惁鍚敤椋庢毚鏃舵绛涢€?STORM_GIC_THRESHOLD = 3.0       # 鈽?v11: 闄嶄綆鍒?A鎹曡幏鏇村椋庢毚鏃舵
STORM_CONTEXT_HOURS = 48        # 鍓嶅悗鎵╁睍灏忔椂鏁?STORM_FILTER_METHOD = "driver_chain"  # driver_chain | target_threshold

# 鏃?SYM-H 鐨勯┍鍔ㄩ摼浜嬩欢绛涢€夊弬鏁?DRIVER_SW_ROLL_MIN = 30         # 澶槼椋庨┍鍔ㄦ粴鍔ㄥ潎鍊肩獥鍙?DRIVER_GM_ROLL_MIN = 10         # 鍦扮鍝嶅簲婊氬姩缁濆鍊兼渶澶х獥鍙?DRIVER_PROPAGATION_MIN = 40     # 澶槼椋庡埌鍦扮鍝嶅簲浼犳挱鏃跺欢(鍒嗛挓)
DRIVER_BZ_THRESHOLD = -4.0      # Bz 30min 鍧囧€奸槇鍊?(nT)
DRIVER_EY_THRESHOLD = 1.5       # Ey 30min 鍧囧€奸槇鍊?(mV/m)
DRIVER_HIGH_QUANTILE = 0.90     # Newell/epsilon/浠ｇ悊鑰﹀悎楂樺垎浣嶉槇鍊?DRIVER_DBHDT_THRESHOLD = 5.0    # |dH/dt| 鍝嶅簲闃堝€?(nT/min)
DRIVER_MERGE_GAP_MIN = 60       # 浜嬩欢鐗囨鍚堝苟鍏佽闂存柇(鍒嗛挓)
DRIVER_PRE_CONTEXT_MIN = 120    # 浜嬩欢鍓嶆墿灞?鍒嗛挓)
DRIVER_POST_CONTEXT_MIN = 360   # 浜嬩欢鍚庢墿灞?鍒嗛挓)
DRIVER_MIN_EVENT_MIN = 120      # 鏈€鐭簨浠舵椂闀?鍒嗛挓)

# ==================== 璇勪及閰嶇疆 ====================
# 鈽?v12: 瀹屽杽.md 瑕佹眰 2/3/4/5A 鍥涢槇鍊艰瘎浼?
PEAK_EVAL_THRESHOLDS = [2.0, 3.0, 4.0, 5.0]  # 鍥涢槇鍊艰瘎浼?
PEAK_VAL_THRESHOLD = 3.0          # 楠岃瘉闆嗗嘲鍊奸槇鍊?(鐢ㄤ簬鏃╁仠)

# 鍖洪棿棰勬祴缃俊搴﹀垪琛?(瀹屽杽.md 瑕佹眰 80%/90%/95%)
PI_ALPHA_LIST = [0.2, 0.1, 0.05]  # alpha = 1 - confidence  鈫?80%, 90%, 95%
# 鍒嗕綅鏁颁簨浠堕槇鍊?
# 鍏堝鍊欓€夊垎浣嶇偣鍋氱粺璁★紝鍐嶈嚜鍔ㄩ€夋嫨浜嬩欢鍗犳瘮鏇村厖瓒崇殑涓€缁?QUANTILE_EVENT_CANDIDATES = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97]
QUANTILE_EVENT_LEVELS = [0.80, 0.85, 0.90, 0.93]
QUANTILE_EVENT_MIN_RATIO = 0.01   # 鑷冲皯 1% 鏍锋湰浣滀负浜嬩欢
QUANTILE_EVENT_MAX_LEVELS = 4

# ==================== 闅忔満绉嶅瓙 ====================
SEED = 42

# ==================== GPU 鍔犻€熼厤缃?(RTX 4060) ====================
USE_AMP = True                   # 鑷姩娣风簿搴?(float16, RTX 4060 涓嶆敮鎸?bfloat16)
AMP_DTYPE = torch.float16        # 鈽?RTX 4060 鐢?float16 (闈?bfloat16)
USE_COMPILE = False              # torch.compile (LSTM 鍏煎鎬ф湁闄? 鎸夐渶寮€鍚?
NUM_WORKERS = 2                  # DataLoader 骞惰宸ヤ綔杩涚▼ (Windows 鍏煎)
PREFETCH_FACTOR = 2              # DataLoader 棰勫彇鍥犲瓙

# ===== hotfix: recover constants from merged comment lines =====
TARGET_COL = "gic"

TRAIN_END = "2013-09-30 23:59:00"
VAL_END = "2014-03-31 23:59:00"
USE_RATIO_SPLIT = True
SPLIT_RATIOS = (0.8, 0.1, 0.1)

FUSED_POINT_LOSS_WEIGHT = 1.0
USE_WEIGHTED_SAMPLER = True

STORM_FILTER = True
STORM_GIC_THRESHOLD = 3.0
STORM_CONTEXT_HOURS = 48
STORM_FILTER_METHOD = "fixed_vkh_cme"   # fixed_vkh_cme | driver_chain | target_threshold

DRIVER_SW_ROLL_MIN = 30
DRIVER_GM_ROLL_MIN = 10
DRIVER_PROPAGATION_MIN = 40
DRIVER_BZ_THRESHOLD = -4.0
DRIVER_EY_THRESHOLD = 1.5
DRIVER_HIGH_QUANTILE = 0.90
DRIVER_DBHDT_THRESHOLD = 5.0
DRIVER_MERGE_GAP_MIN = 60
DRIVER_PRE_CONTEXT_MIN = 120
DRIVER_POST_CONTEXT_MIN = 360
DRIVER_MIN_EVENT_MIN = 120

QUANTILE_EVENT_CANDIDATES = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97]

# ===== bias-control overrides (effective defaults) =====
# Point prediction fusion:
# - "median": use quantile median directly (most conservative; avoids systematic uplift)
# - "gated_residual": median + gate * (expert - median)
# - "gated_positive": median + gate * relu(expert - median)  (legacy)
# - "quantile_blend": q_low + alpha * (q_high - q_low), alpha from peak-logit
# NOTE:
# In the current mild profile, peak-logit related losses are disabled.
# Keep point prediction on median by default to avoid using unsupervised peak logits.
POINT_PREDICTION_MODE = "quantile_blend"
POINT_BLEND_LOW_Q = 0.50
POINT_BLEND_HIGH_Q = 0.90
POINT_BLEND_MAX_ALPHA = 0.45
POINT_BLEND_LOGIT_SCALE = 1.25
POINT_BLEND_LOGIT_BIAS = -0.10

# Stronger training profile for event-only peak learning.
USE_WEIGHTED_SAMPLER = True
PEAK_SAMPLER_POWER = 1.15
PEAK_WEIGHT = 3.0

PEAK_CLS_WEIGHT = 0.4
PEAK_CLS_POS_WEIGHT = 6.0

FUSED_POINT_LOSS_WEIGHT = 0.2
FUSED_POINT_PEAK_BOOST = 1.6

PEAK_MSE_WEIGHT = 0.0
PEAK_MSE_THRESHOLD = 4.0
PEAK_MSE_UNDER_RATIO = 1.0

PEAK_TIER_WEIGHTS = {
    3: 1.0,
    5: 1.0,
    10: 1.0,
    15: 1.0,
    20: 1.0,
}

# Independent point head (kept disabled in scheme1 baseline).
POINT_REG_HEAD_ENABLED = False
POINT_REG_HEAD_HIDDEN = 256

# Point-head loss options:
# - "smooth_l1": legacy fused point loss
# - "tier_huber": piecewise weighted Huber on point prediction
POINT_REG_LOSS_TYPE = "smooth_l1"
POINT_HUBER_DELTA = 1.0
POINT_HUBER_WEIGHT = 0.0
POINT_HUBER_UNDER_RATIO = 1.0
POINT_HUBER_USE_TIER_WEIGHT = True

# Fixed VKH event windows (paper-event mode):
# Use shorter windows to avoid pulling too many non-event samples.
FIXED_VKH_PRE_DAYS = 0.5
FIXED_VKH_POST_DAYS = 1.0

# Label construction mode:
# - "future_window_max": y(t) = max[target(t+1) ... target(t+H)]
# - "fixed_lead_point":  y(t) = target(t+H)
LABEL_MODE = "future_window_max"

# Optional event-quality filter for fixed paper-event mode.
FIXED_VKH_QUALITY_FILTER = True
FIXED_VKH_QUALITY_CORE_HOURS = 6
FIXED_VKH_MIN_CORE_PEAK_A = 3.0
FIXED_VKH_MAX_CORE_DENSITY_MISSING = 1.0
FIXED_VKH_MAX_CORE_FILLED_RATIO = 1.0
