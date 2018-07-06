# importing the dependencies
import gym
import random
import numpy as np 
from gym import wrappers

env = gym.make('CartPole-v0')
bestLength = 0
episode_length =[]
best_weights = np.zeros(4)

for i in range(10):
    if i%10 ==0:
        print('i: ',i)
    new_weights = np.random.uniform(-1, 1, 4)
    length = []
    for j in range(100):
        print(j)
        observation = env.reset()
        done = False
        count = 0
        while not done:
            count = count +1
            action = 1 if np.dot(observation,new_weights) >0 else 0
            observation,reward,done,_ = env.step(action)
            if done:
                break
        length.append(count)
    avg_length = float(sum(length) / len(length))

    if avg_length >bestLength:
        bestLength = avg_length
        best_weights = new_weights 
    episode_length.append(avg_length)
    if i%10 ==0:
        print('best length is: ',bestLength)
    

print(best_weights)

done=  False
count = 0
env = wrappers.Monitor(env,'files',force = True)
observation = env.reset()

while not done:
    count = count +1
    action = 1 if np.dot(observation,new_weights) >0 else 0
    observation,reward,done,_ = env.step(action)
    if done:
        break
print('with best weights, game lasted ',count , ' moves')
