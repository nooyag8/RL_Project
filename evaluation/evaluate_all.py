"""
12개 PPO 모델(fixed, mass_dr, damping_dr, combined_dr x seed 42, 43, 44)을
T1~T5 물리 테스트 환경에서 평가하고 결과를 CSV로 저장하는 스크립트.

실행 예:
    python -m evaluation.evaluate_all --episodes 100
    python -m evaluation.evaluate_all --episodes 10 --env-id Reacher-v4
"""

import argparse
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

EXPERIMENTS: List[str] = ["fixed", "mass_dr", "damping_dr", "combined_dr"]
SEEDS: List[int] = [42, 43, 44]

# T1: Nominal, T2: Mass shift, T3: Damping shift,
# T4: Combined in-distribution shift, T5: Out-of-distribution stress test
TEST_ENVS: List[Dict[str, Any]] = [
    {"test_id": "T1", "test_name": "nominal", "mass_scale": 1.0, "damping_scale": 1.0},
    {"test_id": "T2", "test_name": "mass_shift", "mass_scale": 1.15, "damping_scale": 1.0},
    {"test_id": "T3", "test_name": "damping_shift", "mass_scale": 1.0, "damping_scale": 1.4},
    {"test_id": "T4", "test_name": "combined_in_dist", "mass_scale": 1.15, "damping_scale": 1.4},
    {"test_id": "T5", "test_name": "ood_stress", "mass_scale": 1.3, "damping_scale": 1.75},
]


def find_model_path(models_root: Path, experiment: str, seed: int) -> Optional[Path]:
    """300k 학습 모델만 탐색하며 10k 경로는 절대 선택하지 않습니다."""
    exp_dir = models_root / experiment
    candidates = [
        exp_dir / "300k" / f"seed_{seed}" / f"ppo_reacher_{experiment}_{seed}.zip",
        exp_dir / "300k" / f"seed_{seed}" / "best_model.zip",
        exp_dir / "300k" / f"seed_{seed}" / "final_model.zip",
        exp_dir / "300k" / f"seed_{seed}" / "model.zip",
    ]
    for path in candidates:
        if path.exists() and "10k" not in path.parts:
            return path

    seed_300k_dir = exp_dir / "300k" / f"seed_{seed}"
    if seed_300k_dir.exists():
        zips = sorted(
            p for p in seed_300k_dir.rglob("*.zip")
            if "10k" not in p.parts
        )
        if zips:
            return zips[0]

    exp_300k_dir = exp_dir / "300k"
    if exp_300k_dir.exists():
        zips = sorted(
            p for p in exp_300k_dir.rglob("*.zip")
            if "10k" not in p.parts and str(seed) in p.stem
        )
        if zips:
            return zips[0]

    return None


def resolve_dr_setter() -> Optional[Callable[[Any, float, float], None]]:
    """envs/domain_randomization.py에 정의된 물리 파라미터 설정 함수를 탐색합니다."""
    try:
        dr_mod = importlib.import_module("envs.domain_randomization")
    except ModuleNotFoundError:
        return None

    fn_candidates = [
        "set_physics_params",
        "apply_physics_params",
        "set_domain_randomization",
        "apply_domain_randomization",
        "set_env_params",
    ]
    for fn_name in fn_candidates:
        fn = getattr(dr_mod, fn_name, None)
        if callable(fn):
            return fn
    return None


def apply_fixed_physics(
    env: gym.Env,
    mass_scale: float,
    damping_scale: float,
    base_masses: np.ndarray,
    base_inertias: np.ndarray,
    base_dampings: np.ndarray,
    custom_setter: Optional[Callable[[Any, float, float], None]] = None,
) -> None:
    """테스트 환경의 고정 mass/damping 배율을 MuJoCo 시뮬레이션에 반영합니다."""
    if custom_setter is not None:
        try:
            custom_setter(env, mass_scale, damping_scale)
            return
        except TypeError:
            pass

    unwrapped = env.unwrapped
    model = getattr(unwrapped, "model", None)
    if model is None:
        return

    # Reacher 링크 body 질량 및 관성(inertia) 일관 스케일링
    model.body_mass[:] = base_masses * mass_scale
    model.body_inertia[:] = base_inertias * mass_scale

    # 관절 감쇠(dof_damping) 스케일링
    model.dof_damping[:] = base_dampings * damping_scale

    try:
        import mujoco  # type: ignore

        data = getattr(unwrapped, "data", None)
        if data is not None:
            mujoco.mj_forward(model, data)
    except Exception:
        pass


def compute_target_distance(env: gym.Env, obs: np.ndarray) -> float:
    """관측값 또는 MuJoCo body 좌표에서 끝단(fingertip)과 목표(target) 간 거리를 계산합니다."""
    unwrapped = env.unwrapped
    data = getattr(unwrapped, "data", None)
    model = getattr(unwrapped, "model", None)

    if data is not None and model is not None:
        try:
            import mujoco  # type: ignore

            tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
            target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
            if tip_id >= 0 and target_id >= 0:
                diff = data.xpos[tip_id] - data.xpos[target_id]
                return float(np.linalg.norm(diff[:2]))
        except Exception:
            pass

    # Gymnasium Reacher 표준 관측 벡터 fallback:
    # [cos(q1), cos(q2), sin(q1), sin(q2), target_x, target_y, qdot1, qdot2, dx, dy]
    arr = np.asarray(obs, dtype=np.float64).ravel()
    if arr.size >= 10:
        return float(np.linalg.norm(arr[8:10]))
    return float("nan")


def evaluate_single_condition(
    model: PPO,
    env_id: str,
    mass_scale: float,
    damping_scale: float,
    num_episodes: int,
    success_threshold: float,
    base_seed: int,
    custom_setter: Optional[Callable[[Any, float, float], None]] = None,
) -> List[Dict[str, Any]]:
    """단일 (모델, 테스트 환경) 조건에서 num_episodes 롤아웃을 수행합니다."""
    env = gym.make(env_id)
    unwrapped = env.unwrapped
    base_masses = np.array(unwrapped.model.body_mass, copy=True)
    base_inertias = np.array(unwrapped.model.body_inertia, copy=True)
    base_dampings = np.array(unwrapped.model.dof_damping, copy=True)

    episode_rows: List[Dict[str, Any]] = []

    try:
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=base_seed + ep)
            apply_fixed_physics(
                env=env,
                mass_scale=mass_scale,
                damping_scale=damping_scale,
                base_masses=base_masses,
                base_inertias=base_inertias,
                base_dampings=base_dampings,
                custom_setter=custom_setter,
            )

            total_reward = 0.0
            distances: List[float] = []
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                distances.append(compute_target_distance(env, obs))

            min_dist = float(np.nanmin(distances)) if distances else float("nan")
            final_dist = float(distances[-1]) if distances else float("nan")
            success = bool(min_dist <= success_threshold)

            episode_rows.append(
                {
                    "episode": ep + 1,
                    "episode_reward": total_reward,
                    "min_distance": min_dist,
                    "final_distance": final_dist,
                    "success": int(success),
                }
            )
    finally:
        env.close()

    return episode_rows


def compute_degradation(nominal_sr: float, test_sr: float) -> Tuple[float, float]:
    """
    T1(기본 환경) 성공률 대비 성능 저하율(%)과 성공률 차이(%p)를 반환합니다.
    - degradation_pct: ((SR_nominal - SR_test) / SR_nominal) * 100 (SR_nominal == 0이면 NaN)
    - sr_diff_pp: SR_test - SR_nominal (단위: %p, 음수면 하락)
    """
    sr_diff_pp = test_sr - nominal_sr
    if nominal_sr <= 0.0:
        return float("nan"), sr_diff_pp
    degradation_pct = ((nominal_sr - test_sr) / nominal_sr) * 100.0
    return degradation_pct, sr_diff_pp


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate A-D PPO models on T1-T5 environments.")
    parser.add_argument("--env-id", type=str, default="Reacher-v5", help="Gymnasium Reacher env ID")
    parser.add_argument("--models-dir", type=str, default="models", help="Root models directory")
    parser.add_argument("--results-dir", type=str, default="results", help="Root results directory")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per test condition")
    parser.add_argument("--threshold", type=float, default=0.05, help="Success distance threshold (m)")
    args = parser.parse_args()

    models_root = Path(args.models_dir)
    results_root = Path(args.results_dir)
    out_dir = results_root / "evaluation_t1_t5"
    out_dir.mkdir(parents=True, exist_ok=True)

    custom_setter = resolve_dr_setter()
    all_episode_records: List[Dict[str, Any]] = []
    per_seed_records: List[Dict[str, Any]] = []

    for exp in EXPERIMENTS:
        for seed in SEEDS:
            model_path = find_model_path(models_root, exp, seed)
            if model_path is None:
                print(f"[WARN] Model not found for experiment='{exp}', seed={seed} -> skipped")
                continue

            print(f"[INFO] Loading model: {model_path}")
            model = PPO.load(str(model_path), device="auto")

            seed_env_summaries: List[Dict[str, Any]] = []
            for idx, t_cfg in enumerate(TEST_ENVS):
                ep_rows = evaluate_single_condition(
                    model=model,
                    env_id=args.env_id,
                    mass_scale=float(t_cfg["mass_scale"]),
                    damping_scale=float(t_cfg["damping_scale"]),
                    num_episodes=args.episodes,
                    success_threshold=args.threshold,
                    base_seed=seed * 1000 + idx * 100,
                    custom_setter=custom_setter,
                )

                for r in ep_rows:
                    all_episode_records.append(
                        {
                            "experiment": exp,
                            "seed": seed,
                            "test_id": t_cfg["test_id"],
                            "test_name": t_cfg["test_name"],
                            "mass_scale": t_cfg["mass_scale"],
                            "damping_scale": t_cfg["damping_scale"],
                            **r,
                        }
                    )

                rewards = np.array([r["episode_reward"] for r in ep_rows], dtype=np.float64)
                min_dists = np.array([r["min_distance"] for r in ep_rows], dtype=np.float64)
                final_dists = np.array([r["final_distance"] for r in ep_rows], dtype=np.float64)
                successes = np.array([r["success"] for r in ep_rows], dtype=np.float64)

                seed_env_summaries.append(
                    {
                        "experiment": exp,
                        "seed": seed,
                        "test_id": t_cfg["test_id"],
                        "test_name": t_cfg["test_name"],
                        "mass_scale": t_cfg["mass_scale"],
                        "damping_scale": t_cfg["damping_scale"],
                        "episodes": args.episodes,
                        "success_rate_pct": float(np.mean(successes) * 100.0),
                        "mean_reward": float(np.mean(rewards)),
                        "std_reward": float(np.std(rewards)),
                        "mean_min_distance": float(np.nanmean(min_dists)),
                        "mean_final_distance": float(np.nanmean(final_dists)),
                    }
                )

            nominal_row = next((s for s in seed_env_summaries if s["test_id"] == "T1"), None)
            nominal_sr = nominal_row["success_rate_pct"] if nominal_row is not None else float("nan")

            for s in seed_env_summaries:
                deg_pct, sr_diff_pp = compute_degradation(nominal_sr, s["success_rate_pct"])
                s["degradation_pct"] = deg_pct
                s["sr_diff_pp"] = sr_diff_pp
                per_seed_records.append(s)

    if not per_seed_records:
        print("[ERROR] No models were evaluated. Check models/ directory paths.")
        return

    episodes_df = pd.DataFrame(all_episode_records)
    per_seed_df = pd.DataFrame(per_seed_records)

    summary_df = (
        per_seed_df.groupby(
            ["experiment", "test_id", "test_name", "mass_scale", "damping_scale"],
            as_index=False,
        )
        .agg(
            num_seeds=("seed", "count"),
            success_rate_mean=("success_rate_pct", "mean"),
            success_rate_std=("success_rate_pct", "std"),
            mean_reward_mean=("mean_reward", "mean"),
            mean_reward_std=("mean_reward", "std"),
            mean_min_dist_mean=("mean_min_distance", "mean"),
            mean_min_dist_std=("mean_min_distance", "std"),
            mean_final_dist_mean=("mean_final_distance", "mean"),
            mean_final_dist_std=("mean_final_distance", "std"),
            degradation_pct_mean=("degradation_pct", "mean"),
            degradation_pct_std=("degradation_pct", "std"),
            sr_diff_pp_mean=("sr_diff_pp", "mean"),
            sr_diff_pp_std=("sr_diff_pp", "std"),
        )
    )

    episodes_csv = out_dir / "t1_t5_episodes.csv"
    per_seed_csv = out_dir / "t1_t5_per_seed.csv"
    summary_csv = out_dir / "t1_t5_summary.csv"

    episodes_df.to_csv(episodes_csv, index=False, encoding="utf-8-sig")
    per_seed_df.to_csv(per_seed_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n[DONE] Saved evaluation CSVs:")
    print(f"  - {episodes_csv}")
    print(f"  - {per_seed_csv}")
    print(f"  - {summary_csv}")


if __name__ == "__main__":
    main()