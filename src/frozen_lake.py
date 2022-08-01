#!/usr/bin/env python3

import argparse
from traceback import print_tb
import gym
from pyparsing import replaceWith

import ray
from ray import tune
from importlib.resources import path
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print
from envs.frozen_lake_smooth_penalty import FrozenLakeSmoothPenaltyEnv

parser = argparse.ArgumentParser()
parser.add_argument("--stop-iters", type=int, default=100)


config = {
    "env": FrozenLakeSmoothPenaltyEnv,
    "env_config": {
        "desc": None,
        "map_name":"8x8s",
        "is_slippery":False,
    },
    "horizon": 500,
    "num_workers": 15,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [64, 64],    # Number of hidden layers
        "fcnet_activation": "relu", # Activation function
    },
    "evaluation_num_workers": 6,
    "evaluation_interval": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },
}

class MyExperiment():
    def __init__(self):
        ray.shutdown()
        ray.init()
        self.config = config
        self.trainer = PPOTrainer
        self.env_class = config["env"]

    def train(self, stop_criteria):
        results = tune.run(
            self.trainer,
            stop=stop_criteria,
            config=self.config,
            # local_dir="~/ray_results",
            keep_checkpoints_num=3,
            checkpoint_freq=5,
            checkpoint_at_end=True,
            verbose=1,
        )

        checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean'),
                                                          metric="episode_reward_mean")
        checkpoint_path = checkpoints[-1][0]
        print("Checkpoint path:", checkpoint_path)
        return checkpoint_path, results
    
    def load(self,path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        """
        self.agent = PPOTrainer(config=self.config)
        self.agent.restore(path)

    def test(self):
        """
        Test a trained agent for an episode. Return the episode reward.
        """
        
        env = self.env_class(env_config=self.config["env_config"])
        
        episode_reward = 0
        done = False
        obs = env.reset()
        
        while not done:
            env.render()
            action = self.agent.compute_action(obs)
            print(action)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        return episode_reward
        
def main(debug_mode=False, stop_criteria=None, use_safe_env=False):
    exp = MyExperiment()
    
    checkpoint_path, results = exp.train(stop_criteria={"training_iteration": stop_criteria})
    print("Finished training!")
    
    print("Testing trained agent!\nLoading checkpoint:", checkpoint_path)    
    exp.load(checkpoint_path)
    
    print("Starting testing")
    #TODO: visualize environment
    
    r= exp.test()
    print("Finished testing! Cumulative Episode Reward:",r)
    
# train = True


# if train:
#     trainer = PPOTrainer(config=config)

#     for i in range(100):
#         result=trainer.train()
#         print('\n--- Result ---\n')
#         print(pretty_print(result))
        
#         # if i % 10 == 0:
#         #     checkpoint = trainer.save()
#         #     print("checkpoint saved at", checkpoint)
        
#     trainer.evaluate()
# else:
#     trainer = PPOTrainer(config=config, path=path("ray_results", "frozenlake-smooth-penalty-v1_2019-07-24_16-51-27_0"))

if __name__ == "__main__":
    args = parser.parse_args()
    debug_mode = True
    use_safe_env = False
    main(debug_mode, args.stop_iters, use_safe_env)
