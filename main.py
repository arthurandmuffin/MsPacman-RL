import argparse

from agent import runner, state_functions, config
from agent.q_agent import QLearningAgent
from emulator.game_env import MsPacmanALE

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["training", "play"], required=True)
parser.add_argument("--policy", choices=["eps_greedy", "ucb"])
parser.add_argument("--state_function", choices=["coarse_manhattan_distance", "sector_distance_state"])
parser.add_argument("--display", choices=["true", "false"])
parser.add_argument("--file")
args = parser.parse_args()

if args.mode == "training":
    agent, history = runner.train_loop(
        seed=42, #ARGPATH,
        
        episodes=config.EPISODES,
        reward_clip=config.REWARD_CLIP,
        max_steps=config.MAX_STEPS,
        policy=args.policy,
        
        frame_skip=config.FRAME_SKIP,
        end_when_life_lost=config.END_ON_LIFE_LOSS,
        
        init_q=5.0,
        discount=config.DISCOUNT,
        alpha=config.ALPHA,
        eps_start=config.EPS_START, eps_end=config.EPS_END, eps_decay_steps=config.EPS_DECAY_STEPS,
        ucb_strength=config.UCB_STRENGTH,
        
        filename=args.file,
        state_function=getattr(state_functions, args.state_function),
    )
    print(history["reward"])
elif args.mode == "play":
    agent = QLearningAgent.load(args.file)
    env = MsPacmanALE(seed=0, frame_skip=config.FRAME_SKIP, end_when_life_lost=config.END_ON_LIFE_LOSS)
    print(len(agent.q_by_state))
    print(agent.state_function_name)
    runner.run_episode_ale(
        env=env, 
        agent=agent, 
        state_function=getattr(state_functions, agent.state_function_name), 
        training=False,
        max_steps=config.MAX_STEPS,
        reward_clip=False,
        render=True if args.display == "true" else False,
        fps=60
    )
