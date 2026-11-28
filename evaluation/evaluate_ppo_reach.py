"""
evaluate_ppo.py + evaluate_reach.py 통합 버전.
4개 실험(fixed, mass_dr, damping_dr, combined_dr) 모델을 기본 환경에서 평가합니다.
(보상 통계 + 최소/최종 거리 + 성공률)

실행 예:
    python -m evaluation.evaluate_ppo_reach --steps 300k
    python -m evaluation.evaluate_ppo_reach --steps 500k --episodes 100
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import mujoco
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

EXPERIMENTS: List[str] = ["fixed", "mass_dr", "damping_dr", "combined_dr"]
SEEDS: List[int] = [42, 43, 44]
MAX_STEPS = 50


def find_model_path(models_root: Path, experiment: str, seed: int, steps: str) -> Optional[Path]:
    """models/{experiment}/{steps}/seed_{seed}/ 아래의 모델 zip을 찾습니다."""
    exp_step_dir = models_root / experiment / steps
    step_dir = exp_step_dir / f"seed_{seed}"

    for name in [f"ppo_reacher_{experiment}_{seed}.zip", "best_model.zip", "final_model.zip", "model.zip"]:
        p = step_dir / name
        if p.exists():
            return p

    if step_dir.exists():
        zips = sorted(step_dir.rglob("*.zip"))
        if zips:
            return zips[0]

    if exp_step_dir.exists():
        zips = sorted(p for p in exp_step_dir.rglob("*.zip") if str(seed) in p.stem)
        if zips:
            return zips[0]

    return None


def get_goal_distance(env: gym.Env) -> float:
    """손끝(fingertip)과 목표(target) 사이의 거리를 계산합니다."""
    m = env.unwrapped
    tip_id = mujoco.mj_name2id(m.model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip")
    tgt_id = mujoco.mj_name2id(m.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    tip = m.data.geom_xpos[tip_id]
    tgt = m.data.geom_xpos[tgt_id]
    return float(np.linalg.norm(tip - tgt))


def evaluate_model(model: PPO, env_id: str, n_episodes: int, threshold: float, verbose: bool) -> pd.DataFrame:
    env = gym.make(env_id)
    rows = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            total_reward = 0.0
            distances: List[float] = []

            for _ in range(MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                distances.append(get_goal_distance(env))
                if terminated or truncated:
                    break

            min_d, final_d = min(distances), distances[-1]
            success = min_d <= threshold
            rows.append(
                {
                    "episode": ep + 1,
                    "reward": total_reward,
                    "min_distance": min_d,
                    "final_distance": final_d,
                    "success": int(success),
                }
            )
            if verbose:
                print(
                    f"  Episode {ep + 1:3d} | Reward: {total_reward:8.4f} | "
                    f"Min: {min_d:.4f}m | Final: {final_d:.4f}m | Success: {success}"
                )
    finally:
        env.close()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO reward + reach metrics for all experiments.")
    parser.add_argument("--steps", type=str, required=True, help="학습 스텝 폴더명 (예: 300k, 500k)")
    parser.add_argument("--env-id", type=str, default="Reacher-v5")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.05, help="성공 거리 기준 (m)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--verbose", action="store_true", help="에피소드별 로그 출력")
    args = parser.parse_args()

    models_root = Path(args.models_dir)
    out_dir = Path(args.results_dir) / f"evaluation_ppo_reach_{args.steps}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict] = []
    episode_dfs: List[pd.DataFrame] = []

    for exp in EXPERIMENTS:
        for seed in args.seeds:
            path = find_model_path(models_root, exp, seed, args.steps)
            if path is None:
                print(f"[WARN] Model not found: experiment={exp}, seed={seed}, steps={args.steps} -> skipped")
                continue

            print(f"\n===== {exp} | seed {seed} | {args.steps} =====")
            print(f"[INFO] Loading model: {path}")
            model = PPO.load(str(path), device="auto")

            df = evaluate_model(model, args.env_id, args.episodes, args.threshold, args.verbose)
            df.insert(0, "seed", seed)
            df.insert(0, "experiment", exp)
            df.insert(0, "train_steps", args.steps)
            episode_dfs.append(df)

            sr = df["success"].mean() * 100.0
            print(
                f"Mean Reward: {df['reward'].mean():.4f} | Std: {df['reward'].std(ddof=0):.4f} | "
                f"Max: {df['reward'].max():.4f} | Min: {df['reward'].min():.4f}"
            )
            print(
                f"Mean Min Dist: {df['min_distance'].mean():.4f}m | "
                f"Mean Final Dist: {df['final_distance'].mean():.4f}m | "
                f"Success: {int(df['success'].sum())}/{len(df)} ({sr:.2f}%)"
            )

            per_seed_rows.append(
                {
                    "train_steps": args.steps,
                    "experiment": exp,
                    "seed": seed,
                    "episodes": len(df),
                    "mean_reward": df["reward"].mean(),
                    "std_reward": df["reward"].std(ddof=0),
                    "max_reward": df["reward"].max(),
                    "min_reward": df["reward"].min(),
                    "mean_min_distance": df["min_distance"].mean(),
                    "mean_final_distance": df["final_distance"].mean(),
                    "success_count": int(df["success"].sum()),
                    "success_rate_pct": sr,
                }
            )

    if not per_seed_rows:
        print("[ERROR] 평가된 모델이 없습니다. models/ 경로 구조를 확인하세요.")
        return

    episodes_df = pd.concat(episode_dfs, ignore_index=True)
    per_seed_df = pd.DataFrame(per_seed_rows)
    summary_df = per_seed_df.groupby(["train_steps", "experiment"], as_index=False).agg(
        num_seeds=("seed", "count"),
        mean_reward_mean=("mean_reward", "mean"),
        mean_reward_std=("mean_reward", "std"),
        min_distance_mean=("mean_min_distance", "mean"),
        min_distance_std=("mean_min_distance", "std"),
        final_distance_mean=("mean_final_distance", "mean"),
        final_distance_std=("mean_final_distance", "std"),
        success_rate_mean=("success_rate_pct", "mean"),
        success_rate_std=("success_rate_pct", "std"),
    )
    summary_df["experiment"] = pd.Categorical(summary_df["experiment"], EXPERIMENTS, ordered=True)
    summary_df = summary_df.sort_values("experiment")

    episodes_df.to_csv(out_dir / "ppo_reach_episodes.csv", index=False, encoding="utf-8-sig")
    per_seed_df.to_csv(out_dir / "ppo_reach_per_seed.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "ppo_reach_summary.csv", index=False, encoding="utf-8-sig")

    print("\n===== Summary (seed mean ± std) =====")
    print(summary_df.to_string(index=False))
    print(f"\n[DONE] Saved to: {out_dir}")


if __name__ == "__main__":
    main()