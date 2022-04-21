#!/usr/bin/env python3

from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print
from envs.frozen_lake_smooth_penalty import FrozenLakeSmoothPenaltyEnv

config = {
    "env": "frozenlake-smooth-penalty-v1",
    "env_config": {
        "desc": None,
        "map_name":"6x6s",
        "is_slippery":False,
    },
    "num_workers": 4,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [64, 64],    # Number of hidden layers
        "fcnet_activation": "relu", # Activation function
    },
    "evaluation_num_workers": 2,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    }
}

trainer = PPOTrainer(config=config)

for i in range(100):
    result=trainer.train()
    print(pretty_print(result))
    
    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
    
trainer.evaluate()
