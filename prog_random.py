## plays 1000 games to find the avg time the we survive the game

import gym
env = gym.make('CartPole-v0').env

EPISODES = 1000
avg_time = 0


for i_episode in range(EPISODES):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        if done:
            avg_time = avg_time + t
            print("Episode finished after {} timesteps".format(t+1))
            break

avg_time = avg_time/EPISODES
print('avg time network survives : ', avg_time)