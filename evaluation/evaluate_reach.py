
import gymnasium as gym
import numpy as np
import mujoco
import os
import csv

from stable_baselines3 import PPO

# 설정
MODEL_PATH = "models/300k/ppo_reacher_fixed.zip" #학습량에 따라서 경로 변경 필요
N_EPISODES = 100
MAX_STEPS = 50
SUCCESS_THRESHOLD = 0.05


def get_goal_distance(env):
    """로봇 팔 끝부분과 목표 지점 사이의 거리 계산"""

    mujoco_env = env.unwrapped

    fingertip_id = mujoco.mj_name2id(
        mujoco_env.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "fingertip"
    )

    target_id = mujoco.mj_name2id(
        mujoco_env.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "target"
    )

    fingertip_pos = mujoco_env.data.geom_xpos[fingertip_id].copy()
    target_pos = mujoco_env.data.geom_xpos[target_id].copy()

    return float(np.linalg.norm(fingertip_pos - target_pos))


def main():

    print("===== PPO Goal Reaching Evaluation =====")

    # 모델 및 환경 생성
    env = gym.make("Reacher-v5")
    model = PPO.load(MODEL_PATH)

    episode_rewards = []
    min_distances = []
    final_distances = []

    success_count = 0

    for episode in range(N_EPISODES):

        obs, info = env.reset(seed=episode)

        total_reward = 0
        distances = []

        for step in range(MAX_STEPS):

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            distance = get_goal_distance(env)
            distances.append(distance)

            if terminated or truncated:
                break

        min_distance = min(distances)
        final_distance = distances[-1]

        success = min_distance <= SUCCESS_THRESHOLD

        if success:
            success_count += 1

        episode_rewards.append(total_reward)
        min_distances.append(min_distance)
        final_distances.append(final_distance)

        print(
            f"Episode {episode + 1:3d} | "
            f"Reward: {total_reward:8.4f} | "
            f"Min Distance: {min_distance:.4f}m | "
            f"Final Distance: {final_distance:.4f}m | "
            f"Success: {success}"
        )

    # 전체 결과
    print("\n===== Final Evaluation Results =====")

    print(f"Episodes: {N_EPISODES}")
    print(f"Mean Reward: {np.mean(episode_rewards):.4f}")
    print(f"Mean Minimum Distance: {np.mean(min_distances):.4f}m")
    print(f"Mean Final Distance: {np.mean(final_distances):.4f}m")
    print(f"Success Count: {success_count}/{N_EPISODES}")
    print(f"Success Rate: {success_count / N_EPISODES * 100:.2f}%")

    # CSV 저장
    os.makedirs("results", exist_ok=True)

    with open(
        "results/ppo_reach_evaluation.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Episode",
            "Reward",
            "Min_Distance",
            "Final_Distance",
            "Success"
        ])

        for i in range(N_EPISODES):

            writer.writerow([
                i + 1,
                episode_rewards[i],
                min_distances[i],
                final_distances[i],
                min_distances[i] <= SUCCESS_THRESHOLD
            ])

    env.close()

    print("\nResults saved to results/ppo_reach_evaluation.csv")


if __name__ == "__main__":
    main()
