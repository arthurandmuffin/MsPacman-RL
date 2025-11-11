# Configs for training variables that can't be dictated in path args

# Training loop / emulator settings
EPISODES = 50
FRAME_SKIP = 1
END_ON_LIFE_LOSS = False
REWARD_CLIP = False
MAX_STEPS = 1e5

# Exploration settings
INIT_Q = 5.0
# Epsilon greedy settings
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 5e4
DISCOUNT = 0.99
ALPHA = 0.1
# UCB settings
UCB_STRENGTH = 1