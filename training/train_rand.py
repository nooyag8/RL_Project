# ============================================================
# PPO with Domain Randomization
# Experiments: B (Mass DR), C (Damping DR), D (Combined DR)
# Environment: Gymnasium Reacher-v5
# ============================================================

import os
import argparse

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from configs.ppo_config import (
    ENV_NAME,
    TOTAL_TIMESTEPS,
    LEARNING_RATE,
    N_STEPS,
    BATCH_SIZE,
    N_EPOCHS,
    GAMMA,
    GAE_LAMBDA,
    CLIP_RANGE,
    NET_ARCH,
    SEED,
    MODEL_DIR,
    RESULT_DIR,
)

from envs.domain_randomization import DomainRandomizationWrapper

# ============================================================
# 2. Experiment Configuration
# ============================================================

EXPERIMENTS = {
    "mass_dr": {"randomize_mass": True, "randomize_damping": False},
    "damping_dr": {"randomize_mass": False, "randomize_damping": True},
    "combined_dr": {"randomize_mass": True, "randomize_damping": True},
}


# ============================================================
# 3. Main Training Function
# ============================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment", type=str, required=True, choices=EXPERIMENTS.keys()
    )

    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()

    experiment_name = args.experiment
    experiment_config = EXPERIMENTS[experiment_name]

    seed = args.seed

    # ----------------------------------------
    # Directory Setup
    # ----------------------------------------

    timestep_name = f"{TOTAL_TIMESTEPS // 1000}k"
    seed_name = f"seed_{seed}"

    model_dir = os.path.join(MODEL_DIR, experiment_name, timestep_name, seed_name)

    result_dir = os.path.join(RESULT_DIR, experiment_name, timestep_name, seed_name)

    best_model_dir = os.path.join(model_dir, "best")

    checkpoint_dir = os.path.join(model_dir, "checkpoints")

    eval_dir = os.path.join(result_dir, "eval")

    tensorboard_dir = os.path.join(result_dir, "tensorboard")

    for directory in [
        model_dir,
        best_model_dir,
        checkpoint_dir,
        eval_dir,
        tensorboard_dir,
    ]:

        os.makedirs(directory, exist_ok=True)

    # ----------------------------------------
    # Environment Setup
    # ----------------------------------------

    base_env = gym.make(ENV_NAME)

    env = DomainRandomizationWrapper(
        base_env,
        randomize_mass=experiment_config["randomize_mass"],
        randomize_damping=experiment_config["randomize_damping"],
    )

    env = Monitor(env)

    # Fixed evaluation environment
    eval_env = Monitor(gym.make(ENV_NAME))

    # ----------------------------------------
    # Evaluation Callback
    # ----------------------------------------

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_dir,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # ----------------------------------------
    # Checkpoint Callback
    # ----------------------------------------

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix=f"ppo_reacher_{experiment_name}",
    )

    # ----------------------------------------
    # PPO Model
    # ----------------------------------------

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        policy_kwargs={"net_arch": NET_ARCH},
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_dir,
    )

    # ----------------------------------------
    # Training
    # ----------------------------------------

    print("\n========================================")
    print(" PPO Domain Randomization Training")
    print("========================================")

    print(f"Experiment: {experiment_name}")
    print(f"Seed: {seed}")
    print(f"Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Mass DR: {experiment_config['randomize_mass']}")
    print(f"Damping DR: {experiment_config['randomize_damping']}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # ----------------------------------------
    # Save Final Model
    # ----------------------------------------

    final_model_path = os.path.join(model_dir, f"ppo_reacher_{experiment_name}_{seed}")

    model.save(final_model_path)

    print("\n===== Training Completed =====")
    print(f"Experiment: {experiment_name}")
    print(f"Seed: {seed}")
    print(f"Model saved: {final_model_path}.zip")
    print(f"Results: {result_dir}")

    # ----------------------------------------
    # Close Environments
    # ----------------------------------------

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
