**MsPacman RL**
Trains RL agent (tabular Q-learning) to play Ms. Pacman using Atari simulator ALE.

Built-in heuristics: Coarse manhatttan distance, sector distance states, directional positionless. See agent/state_functions.py for more detail.

For self-implemented heuristics, add name to --state_function in main.py

Mode-specific arguments are in their own section
--mode training/play
--seed int


Training:
--policy eps_greedy/ucb
--state_function coarse_manhattan_distance/sector_distance_state/directional_positionless
--file string

e.g. python main.py --mode training --seed 17 --policy eps_greedy --state_function sector_distance_state --file training1.pkl
This would seed the emulator with 17, train a model using Epsilon-Greedy as the action selection policy, 
the sector_distance_state as the generalizing state function, and save the resulting agent in training1.pkl

Play:
--file string
--display true/false

e.g. python main.py --mode play --file training1.pkl --display true
This would load the agent in file training1.pkl, and play a game with said aget, using pygame for display.
Pygame is not needed if --display false


For graph plotting related functionality, see analytics.py