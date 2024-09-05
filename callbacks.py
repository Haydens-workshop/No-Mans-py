# callbacks.py

import os
import numpy as np
import tensorflow as tf
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Figure
import matplotlib.pyplot as plt
from typing import Union, List
import gym
import torch
from utils import VideoRecorder, capture_screen, process_image

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.training_env = None

    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 steps
        self.training_env = self.model.get_env()
        self.win_size = 100
        self.ep_reward_buffer = []
        self.ep_len_buffer = []

    def _on_step(self) -> bool:
        if self.n_calls % self._log_freq == 0:
            if isinstance(self.training_env, VecEnv):
                if hasattr(self.training_env, 'envs'):
                    env = self.training_env.envs[0]
                    if hasattr(env, 'env'):
                        env = env.env
            else:
                env = self.training_env

            # Log scalar values
            self.logger.record("train/learning_rate", self.model.learning_rate)
            self.logger.record("train/n_updates", self.model.n_updates)
            self.logger.record("train/clip_range", self.model.clip_range)
            self.logger.record("train/clip_range_vf", self.model.clip_range_vf)

            # Log game-specific metrics
            self.logger.record("game/health", env.health)
            self.logger.record("game/oxygen", env.oxygen)
            self.logger.record("game/units", env.units)
            self.logger.record("game/discovered_species", len(env.discovered_species))

            # Log episode statistics
            if len(self.ep_reward_buffer) >= self.win_size:
                self.logger.record("rollout/ep_rew_mean", np.mean(self.ep_reward_buffer[-self.win_size:]))
                self.logger.record("rollout/ep_len_mean", np.mean(self.ep_len_buffer[-self.win_size:]))

            # Log current game screen
            current_screen = env.render(mode='rgb_array')
            figure = plt.figure(figsize=(8, 6))
            plt.imshow(current_screen)
            plt.axis('off')
            self.logger.record("game/current_screen", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()

        return True

    def _on_rollout_end(self):
        self.ep_reward_buffer.append(np.sum(self.model.ep_info_buffer[-1]["r"]))
        self.ep_len_buffer.append(self.model.ep_info_buffer[-1]["l"])

class ProgressBarCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.pbar = None

    def _on_training_start(self):
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.locals['total_timesteps'])
        except ImportError:
            self.pbar = None

    def _on_step(self):
        if self.pbar:
            self.pbar.update(1)
        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()
            self.pbar = None

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model = self.model
            env = self.training_env
            if isinstance(env, VecEnv):
                env = env.envs[0]

            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
            if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                self.model.save(self.save_path)

        return True

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: Union[gym.Env, VecEnv], render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        super().__init__()
        self.eval_env = eval_env
        self.render_freq = render_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.video_recorder = None

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.video_recorder = VideoRecorder(width=self.eval_env.width, height=self.eval_env.height)
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=False,
                callback=self._video_record_callback,
                return_episode_rewards=True,
            )

            self.video_recorder.close()
            self.logger.record("eval/mean_reward", np.mean(episode_rewards))
            self.logger.record("eval/mean_ep_length", np.mean(episode_lengths))

            video_path = f"videos/video_step_{self.n_calls}.mp4"
            self.logger.record("videos/video", Video(video_path, fps=30), exclude=("stdout", "log", "json", "csv"))

        return True

    def _video_record_callback(self, locals_: dict, globals_: dict) -> None:
        screen = self.eval_env.render(mode='rgb_array')
        self.video_recorder.add_frame(screen)

class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.eval_idx = 0

    def _on_step(self) -> bool:
        result = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            # Log additional metrics or create visualizations here
            self.logger.record(f"eval/exploration_rate_{self.eval_idx}", self.model.exploration_rate)
            
            # Create and log a visualization
            fig, ax = plt.subplots()
            ax.plot(self.results)
            ax.set_title("Evaluation Rewards Over Time")
            ax.set_xlabel("Evaluation Number")
            ax.set_ylabel("Mean Reward")
            self.logger.record(f"eval/reward_plot_{self.eval_idx}", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()

        return result

class HyperparamSchedulerCallback(BaseCallback):
    def __init__(self, schedule, verbose=0):
        super(HyperparamSchedulerCallback, self).__init__(verbose)
        self.schedule = schedule

    def _on_step(self) -> bool:
        for param, schedule_fn in self.schedule.items():
            current_value = schedule_fn(self.num_timesteps)
            setattr(self.model, param, current_value)
        return True

# Example usage:
# schedule = {
#     "learning_rate": lambda t: 1e-4 * np.exp(-0.5 * t / 1e6),
#     "clip_range": lambda t: 0.2 * np.exp(-0.5 * t / 1e6)
# }
# hyperparam_callback = HyperparamSchedulerCallback(schedule)

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience=5, min_delta=0, verbose=0):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) < 100:  # Wait for at least 100 episodes
            return True

        mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
        
        if mean_reward > self.best_mean_reward + self.min_delta:
            self.best_mean_reward = mean_reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                if self.verbose > 0:
                    print("Stopping training early due to no improvement in mean reward.")
                return False
        return True

class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomTensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log custom metrics
        self.logger.record("custom/exploration_rate", self.model.exploration_rate)
        self.logger.record("custom/value_loss", self.model.value_loss)
        self.logger.record("custom/policy_loss", self.model.policy_loss)

        # Log gradients
        if hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
            for name, param in self.model.policy.named_parameters():
                if param.grad is not None:
                    self.logger.record(f"gradients/{name}", param.grad.norm().item())

        # Log weight histograms
        for name, param in self.model.policy.named_parameters():
            self.logger.record(f"weights/{name}", param.data.cpu().numpy(), exclude=("stdout", "log", "json", "csv"))

        return True

class EnvironmentStepsCallback(BaseCallback):
    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super(EnvironmentStepsCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            env = self.training_env
            if isinstance(env, VecEnv):
                env = env.envs[0]
            
            # Log environment-specific metrics
            self.logger.record("environment/position", env.position)
            self.logger.record("environment/inventory_size", len(env.inventory))
            self.logger.record("environment/health", env.health)
            self.logger.record("environment/oxygen", env.oxygen)
            
            # Log a snapshot of the game screen
            screen = capture_screen()
            processed_screen = process_image(screen)
            self.logger.record("environment/screen", Figure(plt.imshow(processed_screen)), exclude=("stdout", "log", "json", "csv"))

        return True