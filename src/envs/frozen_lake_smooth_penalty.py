import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

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


class FrozenLakeSmoothPenaltyEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the lake. The water is mostly frozen, but there are a few holes where the
    ice has melted. If you step into one of those holes, you'll fall into the
    freezing water. At this time, there's an international frisbee shortage, so
    it's absolutely imperative that you navigate across the lake and retrieve
    the disc. However, the ice is slippery, so you won't always move in the
    direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, env_config=None):
        
        if env_config != None:
            desc = env_config["desc"]
            self.map_name = env_config["map_name"]
            self.is_slippery = env_config["is_slippery"]
        else:
            desc=None, 
            self.map_name="5x5s", 
            self.is_slippery=True
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
                        if self.is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        super(FrozenLakeSmoothPenaltyEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction])
            )
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()



from ray.tune.registry import register_env

register_env("frozenlake-smooth-penalty-v1", lambda config: FrozenLakeSmoothPenaltyEnv(config))

if __name__ == "__main__":
    import gym
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.tune.logger import pretty_print
    
    config = {
        "env": "frozenlake-smooth-penalty-v1",
        "env_config": {
        "desc": None,
        "map_name":"5x5",
        "is_slippery":False,
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
        },
    }

    trainer = PPOTrainer(config=config)

    for i in range(100):
        result=trainer.train()
        print(pretty_print(result))
        
        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
        
    trainer.evaluate()

