## plays 1000 games to find the avg time the we survive the game

import gym
from gym import wrappers

EPISODES = 1000
avg_time = 0
max_time = -1
env = gym.make('CartPole-v0').env
env= wrappers.Monitor(env, 'random_files',force=True)

for i_episode in range(EPISODES):
    # instansiating the environment
    observation = env.reset()
    for t in range(1000):
        # uncomment this is you want to see the rendering 
        #env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            avg_time = avg_time + t
            if t >max_time:
                max_time = t
                print(max_time)
            #print("Episode finished after {} timesteps".format(t+1))
            break
    # resetting the enviroment
    env.reset()
        

# printing the avg time the game lasted
avg_time = avg_time/EPISODES
print('avg time network survives : ', avg_time)
