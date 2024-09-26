import os
import gymnasium as gym
import argparse

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, version: str, check_freq: int, save_path: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.version = version
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -float('inf')

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate policy
            mean_reward, _ = evaluate_policy(self.model, self.training_env, n_eval_episodes=100)

            # Save model if it is better than the previous ones
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                model_path = os.path.join(self.save_path, f"{self.version}_{mean_reward:.2f}.zip")
                self.model.save(model_path)
                if self.verbose > 0:
                    print(f"Saving new best model | mean reward : {mean_reward:.2f}")

        return True

class ExpertTrainer():
    def __init__(self):
        self.args = self.parse_args()
        self.get_save_path()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
        parser.add_argument("--env_name", type=str, help="Environment name.", default='MountainCar-v0')
        parser.add_argument("--method", type=str, help="Expert's method.", default='PPO')
        parser.add_argument("--steps", type=int, help="Steps of training", default=1e6)
        parser.add_argument("--version", type=str, help="Version of expert", default='model')
        args = parser.parse_args()
        return args

    def get_save_path(self):
        self.save_path = os.path.join(os.getcwd(), 'runs', 'experts', self.args.env_name)
        self.name = f"{self.args.version}_{self.args.method}"
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def train(self):
        self.env = gym.make(self.args.env_name, render_mode='rgb_array')

        if self.args.method == 'PPO':
            method = PPO
        elif self.args.method == 'DQN':
            method = DQN
        
        self.expert = method("MlpPolicy", self.env, verbose=1, tensorboard_log=f"logs/experts/{self.args.env_name}/{self.name}")

        callback = SaveOnBestTrainingRewardCallback(self.name, check_freq=10000, save_path=self.save_path)

        self.expert.learn(total_timesteps=int(self.args.steps), progress_bar=True, callback=callback)
        
        best_mean_reward = float('inf')
        mean_reward, std_reward = evaluate_policy(self.expert, self.expert.get_env(), n_eval_episodes=10)
        
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            self.expert.save(os.path.join(self.save_path, f'{self.name}_{best_mean_reward}'))
            print('Model saved!')

if __name__=='__main__':
    trainer = ExpertTrainer()
    trainer.train()