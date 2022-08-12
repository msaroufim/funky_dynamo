from stable_baselines3 import PPO
import torchdynamo 



with torchdynamo.optimize("eager"):
    model = PPO("MlpPolicy", "CartPole-v1").learn(10_000)