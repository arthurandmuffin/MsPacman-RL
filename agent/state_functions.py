from statics.ram_annotations import MS_PACMAN_RAM_INFO

"""
State functions that generalize similar situations to same state key (tuple)
"""

# State is identified by (player_position, dots_eaten, distance to closest ghost/fruit, if there is fruit, lives remaining)
def coarse_manhattan_distance(ram, _):
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