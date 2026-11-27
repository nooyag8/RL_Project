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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    seed = args.seed

    # 1. 저장 경로 설정
    experiment_name = "fixed"
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

    # 2. 환경 생성
    env = Monitor(gym.make(ENV_NAME))
    eval_env = Monitor(gym.make(ENV_NAME))

    # 3. 평가 Callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_dir,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # 4. Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=checkpoint_dir, name_prefix="ppo_reacher_fixed"
    )

    # 5. PPO 모델 생성
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

    # 6. 학습
    print("\n===== PPO Fixed Training Started =====")
    print(f"Experiment: {experiment_name}")
    print(f"Seed: {seed}")
    print(f"Training Timesteps: {TOTAL_TIMESTEPS:,}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # 7. 최종 모델 저장
    final_model_path = os.path.join(model_dir, f"ppo_reacher_fixed_{seed}")

    model.save(final_model_path)

    print("\n===== PPO Fixed Training Completed =====")
    print(f"Model saved to: {final_model_path}.zip")
    print(f"Evaluation results: {eval_dir}")
    print(f"Checkpoints: {checkpoint_dir}")

    # 8. 환경 종료
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
