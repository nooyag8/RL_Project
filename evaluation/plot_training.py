
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 평가 데이터 불러오기
data = np.load("results/eval/evaluations.npz")

timesteps = data["timesteps"]
results = data["results"]

# 2. 각 평가 시점의 평균 및 표준편차 계산
mean_rewards = np.mean(results, axis=1)
std_rewards = np.std(results, axis=1)

# 3. 그래프 생성
plt.figure(figsize=(10, 6))

plt.plot(
    timesteps,
    mean_rewards,
    label="PPO Mean Reward"
)

plt.fill_between(
    timesteps,
    mean_rewards - std_rewards,
    mean_rewards + std_rewards,
    alpha=0.2,
    label="Standard Deviation"
)

plt.xlabel("Training Timesteps")
plt.ylabel("Mean Episode Reward")
plt.title("PPO Training Reward Curve")

plt.legend()
plt.grid(True)

# 4. 그래프 저장
os.makedirs("results/figures", exist_ok=True)

plt.savefig(
    "results/figures/ppo_training_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 5. 평가 결과 출력
print("\n===== Training Evaluation =====")

for step, reward in zip(timesteps, mean_rewards):
    print(f"Timesteps: {step:>7} | Mean Reward: {reward:.4f}")

print(f"\nTotal evaluations: {len(timesteps)}")
print(f"Initial reward: {mean_rewards[0]:.4f}")
print(f"Final reward: {mean_rewards[-1]:.4f}")
