"""
4개 실험(fixed, mass_dr, damping_dr, combined_dr)의 학습 중 평가 로그(evaluations.npz)를
읽어 PPO 학습 보상 곡선을 그리는 스크립트. 시드가 여러 개면 시드 간 평균 ± 표준편차를 표시합니다.

실행 예:
    python -m evaluation.plot_training_all --eval-root results --steps 500k
    python -m evaluation.plot_training_all --eval-root results \
        --template "results/eval/{experiment}/{steps}/seed_{seed}/evaluations.npz" --steps 500k
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

EXP_ORDER: List[str] = ["fixed", "mass_dr", "damping_dr", "combined_dr"]
EXP_LABELS: Dict[str, str] = {
    "fixed": "Fixed (A)",
    "mass_dr": "Mass DR (B)",
    "damping_dr": "Damping DR (C)",
    "combined_dr": "Combined DR (D)",
}
EXP_COLORS: Dict[str, str] = {
    "fixed": "#6C8EBF",
    "mass_dr": "#D9822B",
    "damping_dr": "#4C9A7A",
    "combined_dr": "#C44E52",
}
SEEDS: List[int] = [42, 43, 44]


def find_eval_file(
    root: Path, experiment: str, seed: int, steps: Optional[str], template: Optional[str]
) -> Optional[Path]:
    """evaluations.npz 경로를 찾습니다. template이 있으면 그 경로를 그대로 사용합니다."""
    if template:
        p = Path(template.format(experiment=experiment, seed=seed, steps=steps))
        return p if p.exists() else None

    matches = []
    for p in root.rglob("evaluations.npz"):
        parts = set(p.parts)
        if experiment in parts and f"seed_{seed}" in parts and (steps is None or steps in parts):
            matches.append(p)
    matches = sorted(matches)
    if len(matches) > 1:
        print(f"[WARN] {experiment}/seed_{seed}: 후보 {len(matches)}개 -> {matches[0]} 사용")
    return matches[0] if matches else None


def load_curve(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """timesteps, 평가 시점별 에피소드 보상 (n_eval, n_episodes)을 반환합니다."""
    data = np.load(path)
    return np.asarray(data["timesteps"]), np.asarray(data["results"], dtype=np.float64)


def aggregate_experiment(
    root: Path, experiment: str, steps: Optional[str], template: Optional[str]
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """시드별 평균 곡선을 모아 (timesteps, mean, std, 시드 수)를 반환합니다."""
    curves: List[Tuple[np.ndarray, np.ndarray]] = []
    for seed in SEEDS:
        path = find_eval_file(root, experiment, seed, steps, template)
        if path is None:
            print(f"[WARN] evaluations.npz not found: experiment={experiment}, seed={seed}")
            continue
        print(f"[INFO] {experiment} seed {seed}: {path}")
        t, res = load_curve(path)
        curves.append((t, res))

    if not curves:
        return None

    if len(curves) == 1:
        t, res = curves[0]
        return t, res.mean(axis=1), res.std(axis=1), 1

    # 시드 간 공통 timestep만 사용
    common = curves[0][0]
    for t, _ in curves[1:]:
        common = np.intersect1d(common, t)
    per_seed = []
    for t, res in curves:
        idx = np.isin(t, common)
        per_seed.append(res[idx].mean(axis=1))
    stacked = np.stack(per_seed, axis=0)  # (n_seeds, n_t)
    return common, stacked.mean(axis=0), stacked.std(axis=0), len(curves)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PPO training reward curves for all experiments.")
    parser.add_argument("--eval-root", type=str, default="results", help="evaluations.npz를 탐색할 루트 폴더")
    parser.add_argument("--steps", type=str, default=None, help="학습 스텝 폴더명 (예: 300k, 500k)")
    parser.add_argument(
        "--template", type=str, default=None,
        help='경로 템플릿. 예: "results/eval/{experiment}/{steps}/seed_{seed}/evaluations.npz"',
    )
    parser.add_argument("--out-dir", type=str, default=None, help="그래프 저장 폴더")
    args = parser.parse_args()

    root = Path(args.eval_root)
    suffix = f"_{args.steps}" if args.steps else ""
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / "figures" / f"training_curves{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    agg: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for exp in EXP_ORDER:
        res = aggregate_experiment(root, exp, args.steps, args.template)
        if res is not None:
            agg[exp] = res

    if not agg:
        print("[ERROR] evaluations.npz를 하나도 찾지 못했습니다. --eval-root 또는 --template을 확인하세요.")
        return

    title_suffix = f" ({args.steps})" if args.steps else ""

    # 1) 한 그래프에 4개 겹치기
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp, (t, m, s, n) in agg.items():
        ax.plot(t, m, color=EXP_COLORS[exp], label=f"{EXP_LABELS[exp]} (n={n})")
        ax.fill_between(t, m - s, m + s, color=EXP_COLORS[exp], alpha=0.18)
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title(f"PPO Training Reward Curve{title_suffix}")
    ax.legend()
    ax.grid(True)
    fig.savefig(out_dir / "ppo_training_curve_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2) 모델별 2x2 (기존 스크린샷 스타일)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for ax, exp in zip(axes.ravel(), EXP_ORDER):
        if exp not in agg:
            ax.set_visible(False)
            continue
        t, m, s, n = agg[exp]
        ax.plot(t, m, color=EXP_COLORS[exp], label="PPO Mean Reward")
        ax.fill_between(t, m - s, m + s, color=EXP_COLORS[exp], alpha=0.2, label="Standard Deviation")
        ax.set_title(f"{EXP_LABELS[exp]} (n={n} seeds)")
        ax.grid(True)
        ax.legend(loc="lower right")
    for ax in axes[1]:
        ax.set_xlabel("Training Timesteps")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean Episode Reward")
    fig.suptitle(f"PPO Training Reward Curve{title_suffix}")
    fig.tight_layout()
    fig.savefig(out_dir / "ppo_training_curve_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3) 요약 출력
    print("\n===== Training Evaluation Summary =====")
    for exp, (t, m, s, n) in agg.items():
        print(f"{EXP_LABELS[exp]:<18} | seeds: {n} | initial: {m[0]:8.4f} | final: {m[-1]:8.4f} | best: {m.max():8.4f}")
    print(f"\n[DONE] Saved to: {out_dir}")


if __name__ == "__main__":
    main()