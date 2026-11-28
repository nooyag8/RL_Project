import gymnasium as gym
from stable_baselines3 import PPO

# 학습된 Fixed 모델 불러오기
MODEL_PATH = "models/fixed/300k/seed_42/ppo_reacher_fixed_42.zip"

# 시뮬레이션 창 띄우기
env = gym.make("Reacher-v5", render_mode="human")
model = PPO.load(MODEL_PATH)

obs, info = env.reset(seed=42)

for episode in range(5):
    obs, info = env.reset()

    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

env.close()
