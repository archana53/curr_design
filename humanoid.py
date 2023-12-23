import os
from copy import deepcopy

import numpy as np
import PIL
from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import cmu_humanoid
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecVideoRecorder,
)
from tqdm import tqdm
from wandb.integration.sb3 import WandbCallback

import wandb
from chat_gpt_generated_functions import (
    cmu_humanoid_run_gaps_step1,
    cmu_humanoid_run_gaps_step2,
    cmu_humanoid_run_gaps_step3,
    cmu_humanoid_run_gaps_step4,
)
from feature_extractor import CustomCombinedExtractor
from video import VideoRecorder
from wrapper import DMCGym


def cmu_humanoid_run_gaps_stepwise(step):
    if step == 1:
        print("Step1 invoked!")
        env = cmu_humanoid_run_gaps_step1()
    elif step == 2:
        print("Step2 invoked!")
        env = cmu_humanoid_run_gaps_step2()
    elif step == 3:
        print("Step3 invoked!")
        env = cmu_humanoid_run_gaps_step3()
    elif step == 4:
        print("Step4 invoked!")
        env = cmu_humanoid_run_gaps_step4()
    return env


def cmu_humanoid_run_gaps(random_state=None, gap_lengths=(1, 2)):
    """Requires a CMU humanoid to run down a corridor with gaps."""

    # Build a position-controlled CMU humanoid walker.
    walker = cmu_humanoid.CMUHumanoidPositionControlled()

    # Build a corridor-shaped arena with gaps, where the sizes of the gaps and
    # platforms are uniformly randomized.
    arena = corr_arenas.GapsCorridor(
        platform_length=distributions.Uniform(0.3, 2.5),
        gap_length=distributions.Uniform(gap_lengths[0], gap_lengths[1]),
        corridor_width=10,
        corridor_length=100,
    )

    # Build a task that rewards the agent for running down the corridor at a
    # specific velocity.
    task = corr_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(0.5, 0, 0),
        target_velocity=3.0,
        physics_timestep=0.005,
        control_timestep=0.03,
    )

    return composer.Environment(
        time_limit=30, task=task, random_state=random_state, strip_singleton_obs_buffer_dim=True
    )


def evaluate(env, agent, video, num_episodes, eval_mode, adapt=False):
    episode_rewards = []
    for i in tqdm(range(num_episodes)):
        vec_env = env
        ep_agent = agent
        obs, _ = vec_env.reset()
        video.init(enabled=True)
        done = False
        episode_reward = 0
        while not done:
            action, _ = ep_agent.predict(obs)
            next_obs, reward, done, _, _ = env.step(action)
            video.record(env, eval_mode)
            if reward is not None:
                episode_reward += reward
            else:
                break
            obs = next_obs

        video.save(f"eval_{eval_mode}_{i}.mp4")
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


def make_env(seed, rank, gap_lengths=(1, 2), step=None):
    def _init():
        try:
            print(f"Initializing environment for rank {rank}")
            if step is not None:
                env = cmu_humanoid_run_gaps_stepwise(step)
            else:
                env = cmu_humanoid_run_gaps(rank, gap_lengths=gap_lengths)
            env = DMCGym(env)
            print(f"Environment initialized for rank {rank}")
            return env
        except Exception as e:
            print(f"Error in environment initialization: {e}")
            raise

    return _init


if __name__ == "__main__":
    # Create gym environment

    no_curr_config = {
        "name": "No-Curriculum-1e6-new",
        "learning_steps": 1e6,
        "num_cpus": 4,
        "gap_curriculum": [(1, 5), (1, 5), (1, 5), (1, 5)],
    }
    gap_config = {
        "name": "Gap-Curriculum-1e6-new",
        "learning_steps": 1e6,
        "num_cpus": 4,
        "gap_curriculum": [(1, 2), (2, 3), (3, 4), (4, 5)],
    }
    chatgpt_config = {
        "name": "ChatGPT-Curriculum-1e6-new",
        "learning_steps": 1e6,
        "num_cpus": 4,
        "gap_curriculum": [(1, 2), (2, 3), (3, 4), (4, 5)],
    }
    # SELECT WHICH CONFIG TO USE!
    config = gap_config

    run = wandb.init(
        project="curr_learning",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
    )
    run.name = config["name"]
    num_steps_per_curriculum = config["learning_steps"] // len(config["gap_curriculum"])
    model = None
    for it, gap_lengths in enumerate(config["gap_curriculum"]):
        print("Starting curriculum: ", it + 1)
        num_cpu = config["num_cpus"]
        if "ChatGPT" in config["name"]:
            vec_env = SubprocVecEnv(
                [make_env(seed=0, rank=i, step=it + 1) for i in range(num_cpu)],
                start_method="spawn",
            )
        else:
            vec_env = SubprocVecEnv(
                [make_env(seed=0, rank=i) for i in range(num_cpu)], start_method="spawn"
            )
        vec_env = VecMonitor(vec_env)
        vec_env = VecVideoRecorder(
            vec_env,
            f"videos/{run.id}",
            record_video_trigger=lambda x: x % 2000 == 0,
            video_length=200,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=100000, save_path=f"./model/{config['name']}"
        )
        # Create the callback list
        wandb_callback = WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
        callback = CallbackList([checkpoint_callback, wandb_callback])
        if os.path.exists(f"saved_models/{config['name']}_{it}.zip"):
            print("Loading existing model available")
            model = PPO.load(f"saved_models/{config['name']}_{it}.zip")
            continue
        elif model is None:
            policy_kwargs = dict(
                features_extractor_class=CustomCombinedExtractor,
                features_extractor_kwargs=dict(),
            )
            model = PPO(
                "MultiInputPolicy",
                vec_env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"runs/{run.id}",
            )
        else:
            model.set_env((vec_env))
        model.learn(total_timesteps=num_steps_per_curriculum, progress_bar=True, callback=callback)
        model.save(f"saved_models/{config['name']}_{it}")
        vec_env.close()
