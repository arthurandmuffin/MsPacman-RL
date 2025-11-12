from collections import defaultdict
import pickle, random

from agent import exploration

class QLearningAgent:
    def __init__(
        self,
        actions, # number of actions available
        discount=0.99,
        alpha=0.1, # learning rate
        init_q=0.0, # prior q, optimistic if > 0
        policy="eps_greedy",  # "eps_greedy" or "ucb" so far
        seed=17,
        # epsilon-greedy vars
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=1e5,
        # ucb vars
        ucb_strength=1.0,
    ):
        self.actions = actions
        self.discount = discount
        self.alpha = alpha
        self.init_q = init_q
        self.total_steps = 0
        
        random.seed(seed)

        # Initialize hash table, state -> q_values/action_counts, initialize w/ init_q/0 on first access
        self.q_by_state = defaultdict(lambda: [float(self.init_q)] * self.actions)
        self.count_by_state = defaultdict(lambda: [0] * self.actions)

        # Initialize exploration policy
        if policy == "eps_greedy":
            self._policy_name = "eps_greedy"
            self._policy = exploration.EpsilonGreedyExploration(eps_start, eps_end, eps_decay_steps)
        elif policy == "ucb":
            self._policy_name = "ucb"
            self._policy = exploration.UCBPolicyExploration(exploration_strength=ucb_strength)
        else:
            raise ValueError("unknown policy")

    # Pick action according to policy
    def select_action(self, state_key):
        qvals = self.q_by_state[state_key]
        if self._policy_name == "eps_greedy":
            return self._policy.select(qvals, self.total_steps, self.actions)
        elif self._policy_name == "ucb":
            return self._policy.select(qvals, self.total_steps, self.count_by_state[state_key])
        # add more policy?

    # Update q_values and count tables after action
    def update(self, init_state_key, action, reward, result_state_key, terminal: bool):
        self.count_by_state[init_state_key][action] += 1 # increase action count
        # td_target = reward of action taken + discounted future greedy reward (if not terminal)
        if terminal:
            td_target = reward
        else:
            td_target = reward + self.discount*max(self.q_by_state[result_state_key])
        # Update q_value of state+action w/ td_target
        self.q_by_state[init_state_key][action] += self.alpha * (td_target - self.q_by_state[init_state_key][action])
        self.total_steps += 1

    # Saves agent variables + state function used to train into a while
    def save(self, path, state_function):
        print("about to save; q_by_state len:", len(self.q_by_state))
        first_key = next(iter(self.q_by_state))
        print("example key:", first_key)
        payload = dict(
            actions=self.actions,
            discount=self.discount,
            alpha=self.alpha,
            init_q=self.init_q,
            policy=self._policy_name,
            policy_params=self.policy_params(),
            total_steps=self.total_steps,
            q_by_state=dict(self.q_by_state),
            count_by_state=dict(self.count_by_state),
            state_function_name=state_function.__name__
        )
        with open(path, "wb") as f: 
            pickle.dump(payload, f)

    # Loads previously saved file as a new agent object, extra attribute state_function
    @staticmethod
    def load(path: str):
        with open(path, "rb") as f: 
            payload = pickle.load(f)
            agent = QLearningAgent(
                actions=payload["actions"],
                discount=payload["discount"],
                alpha=payload["alpha"],
                init_q=payload["init_q"],
                policy=payload["policy"],
                **payload["policy_params"]
            )
            agent.total_steps = payload["total_steps"]
            agent.state_function_name = payload["state_function_name"]
            # Dict instead of defaultdict to recognize unseen states
            agent.q_by_state = dict(payload["q_by_state"])
            agent.count_by_state = dict(payload["count_by_state"])
            return agent

    def policy_params(self):
        if self._policy_name == "eps_greedy":
            return dict(eps_start=self._policy.eps_start, eps_end=self._policy.eps_end, eps_decay_steps=self._policy.decay_steps)
        elif self._policy_name == "ucb":
            return dict(ucb_strength=self._policy.exploration_strength)