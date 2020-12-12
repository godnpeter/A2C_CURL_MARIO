import gym

if __name__=='__main__':

    env = gym.make("PongNoFrameskip-v4")
    observation = env.reset()
    for _ in range(100000):
        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

    env.close()

