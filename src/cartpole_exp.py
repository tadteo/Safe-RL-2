#!/usr/bin/env python3

import ray
import gym 

# Train SAC agent in gym  cartpole environement
def train_SAC_agent():
    ray.init()
    from ray.rllib.agents.sac import SACTrainer
    from ray.tune.logger import pretty_print
    from ray.tune.registry import register_env
    
    config = {
        "env": "CartPole-v0",
        "num_workers": 2,
        "framework": "torch",
        "model": {
            "fcnet_hiddens": [64, 64],    # Number of hidden layers
            "fcnet_activation": "relu", # Activation function
        },
        
        "evaluation_num_workers": 1,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": True,
        },
    }
    trainer = SACTrainer(config=config)
    for i in range(100):
        result = trainer.train()
        print(pretty_print(result))
        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
    trainer.stop()
    ray.shutdown()

if __name__ == "__main__":
    train_SAC_agent()
