# No Mans AI.py

import gym
import numpy as np
import torch
import logging
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from nms_env import NoMansSkyEnv
from seed_manager import SeedManager
from reward_function import RewardFunction
from callbacks import TensorboardCallback, ProgressBarCallback
from utils import setup_logger, create_video_recorder

def parse_arguments():
    parser = argparse.ArgumentParser(description="No Man's Sky AI Training and Evaluation")
    parser.add_argument('--mode', choices=['train', 'test', 'optimize'], default='train', help='Mode to run the script in')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Number of timesteps to train for')
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of episodes to test for')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Logging level')
    return parser.parse_args()

def create_env(seed=None, log_dir="./logs/"):
    seed_manager = SeedManager(seed)
    reward_function = RewardFunction()
    env = NoMansSkyEnv(seed_manager, reward_function)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env

def create_model(env, seed=None):
    return PPO("CnnPolicy", 
               env, 
               verbose=1, 
               tensorboard_log="./nms_ai_tensorboard/",
               learning_rate=3e-4,
               n_steps=2048,
               batch_size=64,
               n_epochs=10,
               gamma=0.99,
               gae_lambda=0.95,
               clip_range=0.2,
               ent_coef=0.01,
               device='cuda' if torch.cuda.is_available() else 'cpu',
               seed=seed)

def train_model(model, env, total_timesteps, seed=None):
    stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=10000, verbose=1)
    eval_env = create_env(seed)
    eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path='./models/',
                                 log_path='./logs/',
                                 eval_freq=10000,
                                 deterministic=True, 
                                 render=False,
                                 callback_after_eval=stop_train_callback)
    
    model.learn(total_timesteps=total_timesteps, 
                callback=[TensorboardCallback(), ProgressBarCallback(), eval_callback],
                tb_log_name="nms_ai_ppo")

    model.save("nms_ai_final_model")

def test_model(model, env, episodes):
    video_recorder = create_video_recorder(env, "test_episodes")
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            video_recorder.capture_frame()

        logging.info(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")
    
    video_recorder.close()

def optimize_hyperparameters(env, total_timesteps, n_trials=50):
    import optuna

    def objective(trial):
        model = PPO("CnnPolicy", 
                    env,
                    learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
                    n_steps=trial.suggest_int('n_steps', 16, 2048),
                    batch_size=trial.suggest_int('batch_size', 8, 256),
                    n_epochs=trial.suggest_int('n_epochs', 3, 30),
                    gamma=trial.suggest_uniform('gamma', 0.9, 0.9999),
                    gae_lambda=trial.suggest_uniform('gae_lambda', 0.9, 1.0),
                    clip_range=trial.suggest_uniform('clip_range', 0.1, 0.3),
                    ent_coef=trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
                    verbose=0)
        
        model.learn(total_timesteps=total_timesteps)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        return mean_reward

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    logging.info(f"Best hyperparameters: {study.best_params}")
    return study.best_params

def main():
    args = parse_arguments()
    setup_logger(args.log_level)
    
    env = create_env(args.seed)
    
    if args.mode == 'train':
        model = create_model(env, args.seed)
        train_model(model, env, args.timesteps, args.seed)
    elif args.mode == 'test':
        model = PPO.load("nms_ai_final_model")
        test_model(model, env, args.test_episodes)
    elif args.mode == 'optimize':
        best_params = optimize_hyperparameters(env, args.timesteps)
        model = create_model(env, args.seed, **best_params)
        train_model(model, env, args.timesteps, args.seed)

if __name__ == "__main__":
    main()