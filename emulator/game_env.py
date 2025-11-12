from ale_py import ALEInterface, LoggerMode, roms
from statics.ram_annotations import MS_PACMAN_RAM_INFO

class MsPacmanALE:
    def __init__(self, seed=0, frame_skip=4, end_when_life_lost=False):
        self.ale = ALEInterface()
        self.ale.setLoggerMode(LoggerMode.Error) # Stops "sending reset..." spam
        self.ale.setInt("random_seed", seed)
        self.ale.setInt("frame_skip", 1)
        self.ale.loadROM(roms.get_rom_path("ms_pacman"))
        #self.ale.loadROM("./bin/MSPACMAN.BIN")
        self.ale.setMode(0)
        self.ale.setDifficulty(0)

        # less options then legalactionset, better learning?
        self.action_set = self.ale.getMinimalActionSet()
        self.actions_count = 4 # Only allow 4 actions
        self.frame_skip = max(1, frame_skip)
        self.end_when_life_lost = end_when_life_lost
        self.lives = self.ale.lives()

    def reset(self):
        self.ale.reset_game()
        self._lives = self.ale.lives()
        ram = self.read_ram()
        player_x, player_y = ram[MS_PACMAN_RAM_INFO["player_x"]], ram[MS_PACMAN_RAM_INFO["player_y"]]
        # Spins to avoid long waiting period at game start to affect learning
        while True:
            self.ale.act(0)
            new_ram = self.read_ram()
            new_x, new_y = new_ram[MS_PACMAN_RAM_INFO["player_x"]], new_ram[MS_PACMAN_RAM_INFO["player_y"]]
            if (player_x != new_x or player_y != new_y):
                break
        return self.read_ram()
    
    # Return an independent copy of RAM
    def read_ram(self):
        return self.ale.getRAM()

    def step(self, a_index):
        ale_action = self.action_set[a_index+1]
        reward = 0
        # loop through frames
        for _ in range(self.frame_skip):
            reward += self.ale.act(int(ale_action))
            if self.end_when_life_lost and self.ale.lives() < self._lives:
                break
            if self.ale.game_over():
                break
        done = self.ale.game_over() or (self.end_when_life_lost and self.ale.lives() < self._lives)
        self._lives = self.ale.lives()
        ram = self.read_ram()
        return ram, reward, done