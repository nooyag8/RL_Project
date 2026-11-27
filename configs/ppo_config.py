# PPO Configuration

ENV_NAME = "Reacher-v5"

# Training
TOTAL_TIMESTEPS = 300_000
# TOTAL_TIMESTEPS = 10_000 # 테스트용

# PPO Hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

# Network
NET_ARCH = [64, 64]

# Reproducibility
SEED = 42

# Paths
MODEL_DIR = "models"
RESULT_DIR = "results"
