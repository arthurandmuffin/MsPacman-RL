import time

from emulator.game_env import MsPacmanALE
from agent.q_agent import QLearningAgent
from agent import state_functions

# Makes agent play 1 game on emulator
def run_episode_ale(
        env: MsPacmanALE,
        agent: QLearningAgent,
        state_function, 
        training=True, 
        max_steps=10000, 
        reward_clip=False,
        render=False,
        fps=60
    ):  
        init_ram = env.reset()
        prev_action = 3 # Initialize as 3 as pacman faces left side
        # State function generalizes states to a state key
        init_state_raw = state_function(init_ram, init_ram, prev_action)
        init_state_key = encode_state(init_state_raw)
        prev_ram = init_ram.copy()
        total_reward = 0
        steps = 0
        
        if render:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((160, 210))
            pygame.display.set_caption("Ms. Pac-Man ALE")

        while True:
            # If training, agent uses exploration policy; else, be greedy and take largest q
            if training:
                action = agent.select_action(init_state_key)
            else:
                q_vals = agent.q_by_state.get(init_state_key)
                if q_vals is None:
                    approximation_function = getattr(state_functions, state_function.__name__ + "_approximation")
                    closest_state_q_vals = approximation_function(agent, init_state_key)
                    action = int(max(range(agent.actions), key=lambda i: closest_state_q_vals[i]))
                else:
                    action = int(max(range(agent.actions), key=lambda i: agent.q_by_state[init_state_key][i]))

            # take action in emulator
            cur_ram, reward, isTerminal = env.step(action)
            if reward_clip:
                reward = max(-1.0, min(1.0, reward))
            
            print("prev_x: ", prev_ram[10], " prev_y: ", prev_ram[16], end="")
            print("cur_x: ", cur_ram[10], " cur_y: ", cur_ram[16])
            
            # Optionally render w/ pygames
            if render:
                # Closing window kills process
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        isTerminal = True
                frame = env.ale.getScreenRGB()
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                time.sleep(1/fps)

            result_state_key = encode_state(state_function(cur_ram, prev_ram, prev_action))

            if training:
                agent.update(init_state_key, action, reward, result_state_key, isTerminal)
            init_state_key = result_state_key
            prev_ram = cur_ram.copy()
            prev_action = action

            total_reward += reward
            steps += 1
            if isTerminal or steps >= max_steps:
                break

        if render:
            pygame.quit()
        return total_reward, steps

def encode_state(state):
    return tuple(sorted(state.items()))

# Training loop
def train_loop(
    seed,
    # Training vars
    episodes,
    reward_clip,
    state_function,
    max_steps,
    # Emulator vars
    frame_skip,
    end_when_life_lost,
    # Agent vars
    discount,
    alpha,
    init_q,
    policy,
    eps_start, eps_end, eps_decay_steps, # Epsilon
    ucb_strength, # UCB

    filename="q_ale.pkl",
):
    env = MsPacmanALE(seed=seed, frame_skip=frame_skip, end_when_life_lost=end_when_life_lost)
    actions = env.actions_count

    agent = QLearningAgent(
        actions=actions, discount=discount, alpha=alpha, init_q=init_q,
        policy=policy, eps_start=eps_start, eps_end=eps_end, eps_decay_steps=eps_decay_steps,
        ucb_strength=ucb_strength, seed=seed,
    )

    history = {"reward": [], "steps": []}
    for ep in range(episodes):
        reward, steps = run_episode_ale(
            env, agent, state_function, training=True, max_steps=max_steps, reward_clip=reward_clip,
        )
        history["reward"].append(reward); 
        history["steps"].append(steps)

    # final save
    if filename:
        agent.save(filename, state_function)
        print(f"Saved file")

    return agent, history