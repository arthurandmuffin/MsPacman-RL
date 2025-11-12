import math

from statics.ram_annotations import MS_PACMAN_RAM_INFO
from agent.q_agent import QLearningAgent

"""
State functions that generalize similar situations to same state key (tuple)
"""

"""State Function 1"""
# State is identified by (player_position, dots_eaten, distance to closest ghost/fruit, if there is fruit, lives remaining)
def coarse_manhattan_distance(ram, prev_ram, prev_action):
    r = MS_PACMAN_RAM_INFO
    player_x, player_y = ram[r["player_x"]], ram[r["player_y"]]
    ghosts_coords = [
        (ram[r["enemy_blinky_x"]], ram[r["enemy_blinky_y"]]),
        (ram[r["enemy_pinky_x"]], ram[r["enemy_pinky_y"]]),
        (ram[r["enemy_inky_x"]], ram[r["enemy_inky_y"]]),
        (ram[r["enemy_sue_x"]], ram[r["enemy_sue_y"]]),
    ]
    closest_ghost_distance = min(abs(int(player_x) - int(ghost_x)) + abs(int(player_y) - int(ghost_y)) for ghost_x, ghost_y in ghosts_coords)

    # fruit distance if present (heuristic: coords > 0 indicate visible)
    fruit_x, fruit_y = ram[r["fruit_x"]], ram[r["fruit_y"]]
    fruit_flag = True if fruit_x > 0 or fruit_y > 0 else False
    if fruit_flag:
        # fruit present
        d_fruit = (abs(int(player_x) - int(fruit_x)) + abs(int(player_y) - int(fruit_y)))
    else:
        d_fruit = 0

    dots_eaten = ram[r["dots_eaten_count"]]
    lives = ram[r["num_lives"]]

    return dict(
        # Round positions and dots eaten
        prev_action = prev_action,
        px=player_x // 2,
        py=player_y // 2,
        dots=(dots_eaten // 5),
        distance_ghost=round_distances(closest_ghost_distance),
        distance_fruit=round_distances(d_fruit),
        fruit=fruit_flag,
        lives=lives,
    )

def round_distances(d):
    if d <= 5: return 0
    if d <= 10: return 1
    if d <= 15: return 2
    return 3

def coarse_manhattan_distance_approximation(agent: QLearningAgent, cur_state):
    min_distance, closest_q_vals = math.inf, None
    for state, q_vals in agent.q_by_state.items():
        distance = coarse_manhattan_state_distance(cur_state, state)
        if distance < min_distance:
            min_distance = distance
            closest_q_vals = q_vals
    return closest_q_vals

def coarse_manhattan_state_distance(state1, state2):
    state1 = dict(state1)
    state2 = dict(state2)
    diff_px, diff_py = (state1["px"] - state2["px"]), (state1["py"] - state2["py"])
    diff_ghost = (state1["distance_ghost"] - state2["distance_ghost"])
    return diff_px + diff_py + diff_ghost


"""State Function 2"""

def sector_distance_state(ram, prev_ram, prev_action):
    r = MS_PACMAN_RAM_INFO

    # Current positions of player & ghosts
    player_x, player_y = int(ram[r["player_x"]]), int(ram[r["player_y"]])
    ghosts = [
        (int(ram[r["enemy_blinky_x"]]), int(ram[r["enemy_blinky_y"]])),
        (int(ram[r["enemy_pinky_x"]]), int(ram[r["enemy_pinky_y"]])),
        (int(ram[r["enemy_inky_x"]]), int(ram[r["enemy_inky_y"]])),
        (int(ram[r["enemy_sue_x"]]), int(ram[r["enemy_sue_y"]])),
    ]

    ghosts_info = []
    for ghost_x, ghost_y in ghosts:
        dx, dy = ghost_x - player_x, ghost_y - player_y
        ghosts_info.append((ghost_x, ghost_y, dx, dy, euclid_distance(dx, dy)))

    # Get distance + direction of nearest ghost
    gx, gy, dxg, dyg, dg = min(ghosts_info, key=lambda t: t[4])
    ghost_sector = round_distance_sector(dg)
    ghost_direction = relative_direction(dxg, dyg)

    # Fruit features
    fruit_x, fruit_y = int(ram[r["fruit_x"]]), int(ram[r["fruit_y"]])
    fruit_flag = True if fruit_x > 0 or fruit_y > 0 else False
    if fruit_flag:
        dxf, dyf = fruit_x - player_x, fruit_y - player_y
        df = euclid_distance(dxf, dyf)
        fruit_sector = round_distance_sector(df)
        fruit_direction = relative_direction(dxf, dyf)
    else:
        fruit_sector, fruit_direction = 4, 8

    # Velocity signs from previous frame (if any)
    if prev_ram is None:
        vxs, vys = 0, 0
    else:
        previous_x, previous_y = int(prev_ram[r["player_x"]]), int(prev_ram[r["player_y"]])
        dxp, dyp = player_x - previous_x, player_y - previous_y
        vxs = 0 if dxp == 0 else (1 if dxp > 0 else -1)
        vys = 0 if dyp == 0 else (1 if dyp > 0 else -1)

    # Coarse position to reduce state count
    px_bin = player_x // 4
    py_bin = player_y // 4

    # Heading (unknown exact encoding; keep 0..3)
    heading = int(ram[r["player_direction"]]) % 4

    return dict(
        px=px_bin, py=py_bin,
        vx=vxs, vy=vys,
        heading=heading,
        ghost_sector=ghost_sector,     # 0..4
        ghost_direction=ghost_direction, # 0..8
        fruit_flag=fruit_flag,
        fruit_sector=fruit_sector, 
        fruit_direction=fruit_direction,
        dots=int(ram[r["dots_eaten_count"]]) // 5,
        lives=int(ram[r["num_lives"]]),
        prev_action=int(prev_action),
    )
    
# Calculates euclidean distance given distance on 2 axes
def euclid_distance(dx, dy):
    return math.sqrt(dx*dx + dy*dy)

# Relative direction of object given dx dy to obj, 0=E 1=NE 2=N 3=NW 4=W 5=SW 6=S 7=SE
def relative_direction(dx, dy):
    # Special case, on top of one another
    if dx == 0 and dy == 0: 
        return 8
    
    if abs(dx) > abs(dy):
        if dx > 0:
            return 0
        return 4
    elif abs(dy) > abs(dx):
        if dy > 0:
            return 2
        return 6
    else:
        if dx > 0 and dy > 0: return 1
        if dx < 0 and dy > 0: return 3
        if dx < 0 and dy < 0: return 5
        if dx > 0 and dy < 0: return 7

# Rounds distances into buckets
def round_distance_sector(d, edges=(8, 16, 32, 64)):
    # 0: very close, 1: close, 2: mid, 3: far, 4: very far
    if d <= 8: return 0
    if d <= 16: return 1
    if d <= 32: return 2
    if d <= 64: return 3
    return 4
    
def sector_distance_state_approximation(agent: QLearningAgent, cur_state):
    min_distance, closest_q_vals = math.inf, None
    for state, q_vals in agent.q_by_state.items():
        distance = sector_distance_state_distance(cur_state, state)
        if distance < min_distance:
            min_distance = distance
            closest_q_vals = q_vals
    return closest_q_vals

def sector_distance_state_distance(state1, state2):
    state1 = dict(state1)
    state2 = dict(state2)

    # coarse position
    diff_px = abs(int(state1["px"]) - int(state2["px"]))
    diff_py = abs(int(state1["py"]) - int(state2["py"]))
    # velocity signs
    diff_vx = 0.5 * abs(int(state1["vx"]) - int(state2["vx"]))
    diff_vy = 0.5 * abs(int(state1["vy"]) - int(state2["vy"]))
    # heading
    diff_heading = heading_dist(state1["heading"], state2["heading"])
    # nearest ghost (band + sector)
    diff_ghost_sector = abs(int(state1.get('ghost_sector', 4)) - int(state2.get('ghost_direction', 4)))
    diff_ghost_direction = relative_direction_dist(state1.get('ghost_sector', 8), state2.get('ghost_direction', 8))

    diff_progress = 0.05 * abs(int(state1.get('dots', 0)) - int(state2.get('dots', 0)))
    diff_prev_act = 0.1  * (int(state1.get('prev_action', -1)) != int(state2.get('prev_action', -1)))
    
    distance = diff_px + diff_py + diff_vx + diff_vy + diff_heading + diff_ghost_sector + diff_ghost_direction + diff_progress + diff_prev_act
    return float(distance)

def direction_dist(mod, a, b):
    a, b = int(a) % mod, int(b) % mod
    diff = abs(a - b) % mod
    return min(diff, mod - diff)

def relative_direction_dist(a, b):
    # sectors 0..7, and 8 means "none"
    a, b = int(a), int(b)
    if a == 8 and b == 8: return 0
    if a == 8 or b == 8:  return 4
    return direction_dist(8, a, b)

def heading_dist(a, b):
    return direction_dist(4, a, b)