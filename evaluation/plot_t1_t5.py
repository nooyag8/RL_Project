"""
T1~T5 평가 결과 CSV(t1_t5_summary.csv, t1_t5_per_seed.csv)를 읽어
깔끔한 논문 스타일의 그래프(성공률, 성능 저하율, 평균 목표 거리)를 생성하는 스크립트.

실행 예:
    python -m evaluation.plot_t1_t5 --steps 500k
    python -m evaluation.plot_t1_t5 --steps 300k
    python -m evaluation.plot_t1_t5 --steps 500k --dist-col mean_final_distance
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_ORDER: List[str] = ["fixed", "mass_dr", "damping_dr", "combined_dr"]
EXP_LABELS: Dict[str, str] = {
    "fixed": "Fixed (A)",
    "mass_dr": "Mass DR (B)",
    "damping_dr": "Damping DR (C)",
    "combined_dr": "Combined DR (D)",
}
# 차분한 논문용 컬러 팔레트 (색각 보정 고려)
EXP_COLORS: Dict[str, str] = {
    "fixed": "#6C8EBF",
    "mass_dr": "#D9822B",
    "damping_dr": "#4C9A7A",
    "combined_dr": "#C44E52",
}

TEST_ORDER: List[str] = ["T1", "T2", "T3", "T4", "T5"]
TEST_LABELS: List[str] = [
    "T1\n(1.0, 1.0)",
    "T2\n(1.15, 1.0)",
    "T3\n(1.0, 1.4)",
    "T4\n(1.15, 1.4)",
    "T5\n(1.3, 1.75)",
]


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Malgun Gothic"],
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_aggregated_df(summary_csv: Path, per_seed_csv: Path, dist_col: str) -> pd.DataFrame:
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        required = {
            "experiment",
            "test_id",
            "success_rate_mean",
            "success_rate_std",
            "degradation_pct_mean",
            "degradation_pct_std",
        }
        dist_mean_col = f"{dist_col}_mean"
        dist_std_col = f"{dist_col}_std"
        if required.issubset(df.columns) and {dist_mean_col, dist_std_col}.issubset(df.columns):
            out = df.copy()
            out["dist_mean"] = out[dist_mean_col]
            out["dist_std"] = out[dist_std_col]
            return out

    if not per_seed_csv.exists():
        raise FileNotFoundError(f"Neither {summary_csv} nor {per_seed_csv} exists.")

    seed_df = pd.read_csv(per_seed_csv)
    return (
        seed_df.groupby(["experiment", "test_id"], as_index=False)
        .agg(
            success_rate_mean=("success_rate_pct", "mean"),
            success_rate_std=("success_rate_pct", "std"),
            degradation_pct_mean=("degradation_pct", "mean"),
            degradation_pct_std=("degradation_pct", "std"),
            dist_mean=(dist_col, "mean"),
            dist_std=(dist_col, "std"),
        )
    )


def extract_series(
    df: pd.DataFrame, exp: str, mean_col: str, std_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    sub = df[df["experiment"] == exp].set_index("test_id")
    means: List[float] = []
    stds: List[float] = []
    for tid in TEST_ORDER:
        if tid in sub.index:
            m_val = sub.loc[tid, mean_col]
            s_val = sub.loc[tid, std_col]
            if isinstance(m_val, pd.Series):
                m_val = m_val.iloc[0]
            if isinstance(s_val, pd.Series):
                s_val = s_val.iloc[0]
            means.append(float(m_val))
            stds.append(0.0 if pd.isna(s_val) else float(s_val))
        else:
            means.append(np.nan)
            stds.append(0.0)
    return np.array(means, dtype=np.float64), np.array(stds, dtype=np.float64)


def plot_grouped_metric(
    df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    ref_line: Optional[float] = None,
    ref_label: Optional[str] = None,
    show_t1_note: bool = False,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=300)

    x = np.arange(len(TEST_ORDER))
    n_exps = len(EXP_ORDER)
    width = 0.18
    gap = 0.015
    step = width + gap

    all_tops: List[float] = [0.0]

    for i, exp in enumerate(EXP_ORDER):
        if exp not in set(df["experiment"]):
            continue
        means, stds = extract_series(df, exp, mean_col, std_col)
        offset = (i - (n_exps - 1) / 2.0) * step
        ax.bar(
            x + offset,
            means,
            width=width,
            color=EXP_COLORS.get(exp, None),
            edgecolor="#2B2B2B",
            linewidth=0.6,
            alpha=0.92,
            yerr=stds,
            capsize=2.5,
            label=EXP_LABELS.get(exp, exp),
            error_kw={"elinewidth": 0.9, "ecolor": "#2B2B2B", "capthick": 0.9},
            zorder=3,
        )
        all_tops.append(float(np.nanmax(means + stds, initial=0.0)))
    if ref_line is not None:
        ax.axhline(
            ref_line,
            color="#444444",
            linewidth=1.0,
            linestyle="--",
            label=ref_label or f"Threshold ({ref_line})",
            zorder=4,
        )
        all_tops.append(ref_line)

    ax.axhline(0.0, color="#333333", linewidth=0.8, zorder=2)

    y_max = max(all_tops)
    y_min_candidates = [0.0]
    if show_t1_note:
        # 음수 저하율(성능 소폭 상승) 및 범례/주석 공간 확보
        sub_min = df[mean_col].min() - df[std_col].fillna(0).max()
        if pd.notna(sub_min) and sub_min < 0:
            y_min_candidates.append(float(sub_min) * 1.25)

    y_low = min(y_min_candidates)
    y_high = y_max * 1.22 if y_max > 0 else 1.0
    ax.set_ylim(y_low, y_high)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(TEST_LABELS, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=14, loc="left", color="#111111")

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, color="#888888", zorder=1)
    ax.legend(
        frameon=False,
        fontsize=8.5,
        ncol=2,
        loc="upper right",
    )

    if show_t1_note:
        ax.text(
            0.0,
            -0.23,
            "Note: Degradation is measured relative to T1 (nominal); negative values indicate slight improvement over T1.",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot T1-T5 evaluation metrics with seed mean ± std.")
    parser.add_argument(
        "--steps",
        type=str,
        default="300k",
        help="그래프를 그릴 학습 스텝 (예: 300k, 500k). results/evaluation_t1_t5_{steps}를 읽습니다.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="결과 폴더를 직접 지정할 때 사용 (지정하면 --steps 기반 경로보다 우선)",
    )
    parser.add_argument(
        "--dist-col",
        type=str,
        default="mean_min_distance",
        choices=["mean_min_distance", "mean_final_distance"],
        help="Distance metric column to plot",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Target distance success threshold (m) for reference line",
    )
    args = parser.parse_args()

    if args.results_dir is not None:
        res_dir = Path(args.results_dir)
    else:
        res_dir = Path("results") / f"evaluation_t1_t5_{args.steps}"

    summary_csv = res_dir / "t1_t5_summary.csv"
    per_seed_csv = res_dir / "t1_t5_per_seed.csv"
    plots_dir = res_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_aggregated_df(summary_csv, per_seed_csv, args.dist_col)

    plot_grouped_metric(
        df=df,
        mean_col="success_rate_mean",
        std_col="success_rate_std",
        title=f"Target Reaching Success Rate Across T1–T5 ({args.steps})",
        ylabel="Success Rate (%)",
        out_path=plots_dir / "fig1_t1_t5_success_rate.png",
    )

    plot_grouped_metric(
        df=df,
        mean_col="degradation_pct_mean",
        std_col="degradation_pct_std",
        title=f"Performance Degradation Relative to T1 ({args.steps})",
        ylabel="Degradation Rate (%)",
        out_path=plots_dir / "fig2_t1_t5_degradation.png",
        show_t1_note=True,
    )

    dist_label = (
        "Mean Minimum Distance to Target (m)"
        if args.dist_col == "mean_min_distance"
        else "Mean Final Distance to Target (m)"
    )
    plot_grouped_metric(
        df=df,
        mean_col="dist_mean",
        std_col="dist_std",
        title=f"{dist_label} Across T1–T5 ({args.steps})",
        ylabel="Distance (m)",
        out_path=plots_dir / f"fig3_t1_t5_{args.dist_col}.png",
        ref_line=args.threshold,
        ref_label=f"Success Threshold ({args.threshold:.2f} m)",
    )

    print("[DONE] Saved refined plots to:", plots_dir)


if __name__ == "__main__":
    main()