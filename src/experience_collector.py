#!/usr/bin/env python3

import ray.rllib.evaluation.rollout_worker as rollout_worker
import ray.rllib.policy.sample_batch as sample_batch
import ray.rllib.policy.ppo
from envs.vacuum_cleaner_env import VacuumCleanerEnv


worker = rollout_worker.RolloutWorker(
    lambda _: VacuumCleanerEnv(desc=None,map_name= "6x6",is_slippery= False),
    policy_spec=PPOPolicy)
