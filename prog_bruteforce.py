# make a random vector which gives good result

# importing the dependencies
import gym
import random
import numpy as np 
from gym import wrappers

env = gym.make('CartPole-v1').env
bestLength = 0
episode_length =[]
best_weights = np.zeros(4)
flag = 0
max_life = 1000

for i in range(10):
    new_weights = np.random.uniform(-1, 1, 4)
    length = []
    for j in range(500):
        observation = env.reset()
        done = False
        count = 0
        while not done:
            count = count +1
            action = 1 if np.dot(observation,new_weights) >0 else 0
            observation,reward,done,_ = env.step(action)
            if done:
                break
            elif count > max_life:
                flag =1
                break
        length.append(count)
    avg_length = float(sum(length) / len(length))

    if avg_length >bestLength:
        bestLength = avg_length
        best_weights = new_weights 
    episode_length.append(avg_length)
    if flag ==1:
        break

print(best_weights)


## testing
env= wrappers.Monitor(env, 'brute_force_files', force = True)
done=  False
count = 0
observation = env.reset()

while not done:
    count = count +1
    action = 1 if np.dot(observation,best_weights) >0 else 0
    observation,reward,done,_ = env.step(action)
    if done:
        break
print('with best weights, game lasted ',count , ' moves')
