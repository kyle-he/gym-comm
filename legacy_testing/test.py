import gym
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent
from stable_baselines3.common.evaluation import evaluate_policy
import gym_comm

env = gym.make('OvercookedMultiCommEnv-v0')
model = PPO.load("ppo_overcooked")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()