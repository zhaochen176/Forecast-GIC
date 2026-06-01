"""
GIC 预测项目 - 相关性分析与特征重要性模块
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.feature_selection import mutual_info_regression
from typing import List, Optional, Dict
import warnings
import os

from src.config import FIGURE_DIR, REPORT_DIR, SEED

# 全局中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def compute_pearson_correlation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "gic_abs",
) -> pd.Series:
    """计算 Pearson 相关系数。"""
    corr = df[feature_cols].corrwith(df[target_col], method="pearson")
    return corr.sort_values(ascending=False, key=abs)


def compute_spearman_correlation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "gic_abs",
) -> pd.Series:
    """计算 Spearman 秩相关系数。"""
    corr = df[feature_cols].corrwith(df[target_col], method="spearman")
    return corr.sort_values(ascending=False, key=abs)


def compute_mutual_information(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "gic_abs",
    n_samples: int = 50000,
) -> pd.Series:
    """
    计算互信息（采样以加速）。
    """
    print(f"[互信息] 采样 {n_samples} 条数据计算...")
    sample = df[feature_cols + [target_col]].dropna().sample(
        n=min(n_samples, len(df)), random_state=SEED
    )
    X = sample[feature_cols].values
    y = sample[target_col].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = mutual_info_regression(X, y, random_state=SEED, n_neighbors=5)

    mi_series = pd.Series(mi, index=feature_cols)
    return mi_series.sort_values(ascending=False)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: List[str],
    title: str = "特征相关性热力图",
    save_name: str = "correlation_heatmap.png",
    figsize: tuple = (16, 14),
) -> None:
    """绘制特征间相关性热力图。"""
    corr_matrix = df[cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, annot=False, fmt=".2f",
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[可视化] 已保存: {save_path}")


def plot_top_correlations(
    pearson: pd.Series,
    spearman: pd.Series,
    mi: pd.Series,
    top_n: int = 20,
    save_name: str = "feature_importance_comparison.png",
) -> None:
    """绘制 Top-N 特征重要性对比图（Pearson / Spearman / MI）。"""
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # Pearson
    top_pearson = pearson.abs().head(top_n)
    axes[0].barh(range(len(top_pearson)),
                 top_pearson.values, color="steelblue")
    axes[0].set_yticks(range(len(top_pearson)))
    axes[0].set_yticklabels(top_pearson.index, fontsize=8)
    axes[0].set_title(f"Pearson |r| Top-{top_n}", fontsize=13)
    axes[0].invert_yaxis()

    # Spearman
    top_spearman = spearman.abs().head(top_n)
    axes[1].barh(range(len(top_spearman)), top_spearman.values, color="coral")
    axes[1].set_yticks(range(len(top_spearman)))
    axes[1].set_yticklabels(top_spearman.index, fontsize=8)
    axes[1].set_title(f"Spearman |ρ| Top-{top_n}", fontsize=13)
    axes[1].invert_yaxis()

    # MI
    top_mi = mi.head(top_n)
    axes[2].barh(range(len(top_mi)), top_mi.values, color="seagreen")
    axes[2].set_yticks(range(len(top_mi)))
    axes[2].set_yticklabels(top_mi.index, fontsize=8)
    axes[2].set_title(f"互信息 Top-{top_n}", fontsize=13)
    axes[2].invert_yaxis()

    plt.suptitle("特征重要性对比 (与 |GIC| 的关联)", fontsize=15, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[可视化] 已保存: {save_path}")


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "gic_abs",
    save_name: str = "gic_distribution.png",
) -> None:
    """绘制 GIC 分布图。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 直方图
    axes[0].hist(df[target_col], bins=200, color="steelblue", edgecolor="none")
    axes[0].set_title("|GIC| 分布（全量）", fontsize=13)
    axes[0].set_xlabel("|GIC| (A)")
    axes[0].set_ylabel("频次")

    # 对数直方图
    vals = df[target_col][df[target_col] > 0]
    axes[1].hist(np.log1p(vals), bins=200, color="coral", edgecolor="none")
    axes[1].set_title("log(1+|GIC|) 分布", fontsize=13)
    axes[1].set_xlabel("log(1+|GIC|)")

    # 超过阈值的 GIC 时间分布
    threshold = df[target_col].quantile(0.95)
    peaks = df[df[target_col] > threshold]
    axes[2].scatter(peaks.index, peaks[target_col], s=1, alpha=0.3, c="red")
    axes[2].set_title(f"|GIC| > {threshold:.2f} 的时间分布 (Top 5%)", fontsize=13)
    axes[2].set_xlabel("时间")
    axes[2].set_ylabel("|GIC| (A)")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[可视化] 已保存: {save_path}")


def run_correlation_analysis(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "gic_abs",
    top_n: int = 20,
) -> Dict[str, pd.Series]:
    """
    运行完整的相关性分析流水线。

    Returns
    -------
    dict  包含 pearson, spearman, mi 三种相关性结果
    """
    print("\n" + "=" * 60)
    print("[相关性分析] 开始分析...")
    print("=" * 60)

    # 1. 目标分布
    plot_target_distribution(df, target_col)

    # 2. Pearson
    print("\n[Pearson 相关系数]")
    pearson = compute_pearson_correlation(df, feature_cols, target_col)
    print(pearson.head(top_n).to_string())

    # 3. Spearman
    print("\n[Spearman 秩相关系数]")
    spearman = compute_spearman_correlation(df, feature_cols, target_col)
    print(spearman.head(top_n).to_string())

    # 4. 互信息
    print("\n[互信息]")
    mi = compute_mutual_information(df, feature_cols, target_col)
    print(mi.head(top_n).to_string())

    # 5. 可视化
    # 选取 Top 特征画热力图
    top_features = list(pearson.abs().head(top_n).index)
    if target_col not in top_features:
        top_features.append(target_col)
    plot_correlation_heatmap(
        df, top_features,
        title=f"Top-{top_n} 特征相关性热力图",
        save_name="correlation_heatmap_top.png",
    )

    # 三种方法对比
    plot_top_correlations(pearson, spearman, mi, top_n=top_n)

    # 6. 保存报告
    report = pd.DataFrame({
        "Pearson_r": pearson,
        "|Pearson_r|": pearson.abs(),
        "Spearman_rho": spearman,
        "|Spearman_rho|": spearman.abs(),
    })
    # 合并 MI（索引可能不完全一致）
    mi_df = mi.rename("MI")
    report = report.join(mi_df, how="left")
    report = report.sort_values("|Pearson_r|", ascending=False)

    report_path = os.path.join(REPORT_DIR, "correlation_report.csv")
    report.to_csv(report_path, encoding="utf-8-sig")
    print(f"\n[相关性分析] 报告已保存: {report_path}")

    return {"pearson": pearson, "spearman": spearman, "mi": mi}
