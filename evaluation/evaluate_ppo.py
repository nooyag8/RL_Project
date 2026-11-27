
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

MODEL_PATH = "models/300k/ppo_reacher_fixed.zip" #학습량에 따라서 경로 변경 필요
N_EPISODES = 100

# 1. 환경 생성
env = gym.make("Reacher-v5")

# 2. 학습된 모델 불러오기
model = PPO.load(MODEL_PATH)

rewards = []

# 3. 100 Episode 평가
for episode in range(N_EPISODES):

    obs, info = env.reset(seed=episode)
    total_reward = 0

    for step in range(50):

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    rewards.append(total_reward)

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.4f}"
    )

# 4. 결과 출력
print("\n===== PPO Evaluation Results =====")
print(f"Mean Reward: {np.mean(rewards):.4f}")
print(f"Std Reward:  {np.std(rewards):.4f}")
print(f"Max Reward:  {np.max(rewards):.4f}")
print(f"Min Reward:  {np.min(rewards):.4f}")

env.close()