import gym
env = gym.make('CartPole-v0')
for i_episode in range(2000):
    observation = env.reset()
    initial_reward = 0
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        initial_reward += reward 
        if done:
            print(initial_reward)
            print("Episode finished after {} timesteps".format(t+1))
            break