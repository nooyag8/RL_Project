
import gymnasium as gym

# 1. Reacher 환경 생성
env = gym.make("Reacher-v5")

# 2. 환경 초기화
obs, info = env.reset(seed=42)

print("===== Environment Information =====")
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# 3. 랜덤 행동으로 10개 Episode 실행
for episode in range(10):
    obs, info = env.reset()
    total_reward = 0

    for step in range(50):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.4f}"
    )

env.close()

print("\nReacher environment test completed!")
