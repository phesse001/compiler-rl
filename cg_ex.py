import gym
import compiler_gym
env = gym.make("llvm-autophase-ic-v0")
env.require_dataset("npb-v0")
env.reset(benchmark="benchmark://npb-v0/50")
#env.render()
episode_reward = 0

test = env.observation
print(test)

for i in range(1, 101):
    observation, reward, done, info = env.step(env.action_space.sample())
    print(info)
    if done:
        break
    episode_reward += reward
    print(f"Step {i}, quality={episode_reward:.3%}")

env.close()