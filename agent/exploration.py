import math, random

# Epsilon [0,1] is chance to choose random actions instead of that with max q-value, 
# decreases w/ step count
class EpsilonGreedyExploration:
    def __init__(self, eps_start=1.0, eps_end=0.05, decay_steps=1e5):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = decay_steps

    def epsilon(self, total_steps) -> float:
        if self.decay_steps <= 0:
            return self.eps_end
        
        # Decrease epsilon linearly with total steps taken
        decay_ratio = max(0, 1 - (total_steps/float(self.decay_steps)))
        return self.eps_end + (self.eps_start - self.eps_end) * decay_ratio

    def select(self, q_values, total_steps, actions) -> int:
        if random.random() < self.epsilon(total_steps):
            # Randrange equivalent of picking random action index
            return random.randrange(actions)
        return q_values.index(max(q_values))

# UCB: score action w/ q-value + strength * sqrt(ln(1+t)/(1+Count(s,a))), inflates score if count low
class UCBPolicyExploration:
    def __init__(self, exploration_strength=1.0):
        self.exploration_strength = exploration_strength

    def select(self, q_values, t, counts):
        best_score, best_action = -math.inf, 0
        # Precompute ln, max(1, t) for first step to avoid 0 exploration incentive
        logt = math.log(1 + max(1, t))
        for action, q_value in enumerate(q_values):
            action_count = counts[action]
            score = q_value + self.exploration_strength * math.sqrt(logt / (1 + action_count))
            if score > best_score:
                best_score, best_action = score, action
        return best_action