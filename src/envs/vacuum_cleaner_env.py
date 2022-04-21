import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete


#Custom vacuum cleaner gym environment with pyGame
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
SUCK = 4 #Action code for sucking


MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
    "8x8s": [
        "SFFFFFFF",
        "FFFUFFFF",
        "FFUHUUFF",
        "FFFUUHUF",
        "FUUHUUUF",
        "UHHUUUHU",
        "UHUUHUHU",
        "FUUHUFUG",
    ],
    "5x5s": [
        "SFFFF",
        "FUUUF",
        "FUHUF",
        "FUUUF",
        "FFFFG",
    ],
    "5x5": [
        "SFFFF",
        "FFFFF",
        "FFHFF",
        "FFFFF",
        "FFFFG",
    ],
    "6x6s": [
        "SFUFFF",
        "FUHUUF",
        "FUHUHU",
        "FFUUFF",
        "FFUHUF",
        "FFFUFG",
    ],
    "6x6": [
        "SFFFFF",
        "FFHFFF",
        "FFHFHF",
        "FFFFFF",
        "FFFHFF",
        "FFFFFG",
    ],
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == "G":
                        return True
                    if res[r_new][c_new] != "H":
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        res[0][0] = "S"
        res[-1][-1] = "G"
        valid = is_valid(res)
    
    print(res)
    # Set all tiles close to holes as unsafe ("U")
    # for r in range(res.shape[0]):
    #     for c in range(res.shape[1]):
    #         print(r,c)
    #         if res[r][c] == "F" and (res[r-1][c-1] == "H" or res[r-1][c+1] == "H" or res[r+1][c-1] == "H" or res[r+1][c+1] == "H"):
    #             res[r][c] = "U"    
        
    return ["".join(x) for x in res]


class VacuumCleanerEnv(discrete.DiscreteEnv):
    """
    The House is dirty again (as always). And it is time to clean it, but you are very tired, for this reason you have decided
    to spend much more time to build a robot that can clean the house instead of yourself. 
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point
    F : Flor surface, safe
    H : Human, the human is in the house and will kick you if you hit they.
    G : Goal(s), where the dirt is located
    The episode ends when you reach all the goals.
    """

    metadata = {"render.modes": ["human", "ansi","gui"],"render_fps": 4}

    def __init__(self, env_config=None):
        
        if env_config != None:
            desc = env_config["desc"]
            self.map_name = env_config["map_name"]
        else:
            desc=None, 
            self.map_name="5x5s", 
        if desc is None and self.map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[self.map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-1000, 1000)
        
        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b"S").astype("float64").ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b"GH"
            if (newletter == b"G"):
                reward = +1.1
            elif (newletter == b"H"):
                reward = -1
            elif (newletter == b"F"):
                reward = 0.0
            elif (newletter == b"U"):
                reward = -0.25
            else:
                reward = 0.0
            reward -= 0.1
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        li.append((1.0, *update_probability_matrix(row, col, a)))

        super(VacuumCleanerEnv, self).__init__(nS, nA, P, isd)

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.window_surface = None
        self.clock = None
        self.human_img = None
        self.cracked_human_img = None
        self.ice_img = None
        self.vacuum_cleaner_images = None
        self.goal_img = None
        self.start_img = None
        
        
    def render(self, mode="gui"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        
        if mode == "human":
            desc = [[c.decode("utf-8") for c in line] for line in desc]
            desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
            if self.lastaction is not None:
                outfile.write(
                    "  ({})\n".format(["Left", "Down", "Right", "Up", "Suck"][self.lastaction])
                )
            else:
                outfile.write("\n")
            outfile.write("\n".join("".join(line) for line in desc) + "\n")
        if mode == "gui":
            return self._render_gui(desc)
        if mode != "human" and mode != "gui":
            with closing(outfile):
                return outfile.getvalue()
                    
    def _render_gui(self, desc):
        import pygame
        from pygame.constants import SRCALPHA
        from os import path
        
        if self.window_surface is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Vacuum Cleaner")
            self.window_surface = pygame.display.set_mode(self.window_size)
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.human_img is None:
            humans = [
                path.join(path.dirname(__file__), "../../img/elf_down.png"),
                path.join(path.dirname(__file__), "../../img/elf_up.png"),
                path.join(path.dirname(__file__), "../../img/elf_left.png"),
                path.join(path.dirname(__file__), "../../img/elf_right.png"),
            ]
            self.human_img = [pygame.image.load(file_name) for file_name in humans]
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "../../img/ice.png")
            self.ice_img = pygame.image.load(file_name)
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "../../img/dirt.png")
            self.goal_img = pygame.image.load(file_name)
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "../../img/stool.png")
            self.start_img = pygame.image.load(file_name)
        if self.vacuum_cleaner_images is None:
            elfs = [
                path.join(path.dirname(__file__), "../../img/vc.png"),
                path.join(path.dirname(__file__), "../../img/vc.png"),
                path.join(path.dirname(__file__), "../../img/vc.png"),
                path.join(path.dirname(__file__), "../../img/vc.png"),
            ]
            self.vacuum_cleaner_images = [pygame.image.load(f_name) for f_name in elfs]

        cell_width = self.window_size[0] // self.ncol
        cell_height = self.window_size[1] // self.nrow
        smaller_cell_scale = 0.6
        small_cell_w = int(smaller_cell_scale * cell_width)
        small_cell_h = int(smaller_cell_scale * cell_height)

        # prepare images
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.vacuum_cleaner_images[last_action]
        elf_scale = min(
            small_cell_w / elf_img.get_width(),
            small_cell_h / elf_img.get_height(),
        )
        elf_dims = (
            elf_img.get_width() * elf_scale,
            elf_img.get_height() * elf_scale,
        )
        elf_img = pygame.transform.scale(elf_img, elf_dims)
        
        
        human_img = pygame.transform.scale(self.human_img, (cell_width, cell_height))
        ice_img = pygame.transform.scale(self.ice_img, (cell_width, cell_height))
        goal_img = pygame.transform.scale(self.goal_img, (cell_width, cell_height))
        start_img = pygame.transform.scale(self.start_img, (small_cell_w, small_cell_h))

        for y in range(self.nrow):
            for x in range(self.ncol):
                rect = (x * cell_width, y * cell_height, cell_width, cell_height)
                if desc[y][x] == b"H":
                    self.window_surface.blit(human_img, (rect[0], rect[1]))
                elif desc[y][x] == b"G":
                    self.window_surface.blit(ice_img, (rect[0], rect[1]))
                    goal_rect = self._center_small_rect(rect, goal_img.get_size())
                    self.window_surface.blit(goal_img, goal_rect)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(ice_img, (rect[0], rect[1]))
                    stool_rect = self._center_small_rect(rect, start_img.get_size())
                    self.window_surface.blit(start_img, stool_rect)
                else:
                    self.window_surface.blit(ice_img, (rect[0], rect[1]))

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (
            bot_col * cell_width,
            bot_row * cell_height,
            cell_width,
            cell_height,
        )
        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(cracked_human_img, (cell_rect[0], cell_rect[1]))
        else:
            elf_rect = self._center_small_rect(cell_rect, elf_img.get_size())
            self.window_surface.blit(elf_img, elf_rect)

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        # else:  # rgb_array
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
        #     )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )


from ray.tune.registry import register_env

register_env("vacuum-cleaner-v1", lambda config: VacuumCleanerEnv(config))

if __name__ == "__main__":
    import gym
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.tune.logger import pretty_print
    
    config = {
        "env": VacuumCleanerEnv,
        "env_config": {
        "desc": None,
        "map_name":"5x5",
        },
        "num_workers": 2,
        "framework": "torch",
        "model": {
            "fcnet_hiddens": [32, 32],    # Number of hidden layers
            "fcnet_activation": "relu", # Activation function
        },
        "evaluation_num_workers": 1,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": True,
            "render_mode": "gui",
        },
    }

    trainer = PPOTrainer(config=config)

    for i in range(2):
        result=trainer.train()
        print(pretty_print(result))
        
        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
        
    trainer.evaluate()


