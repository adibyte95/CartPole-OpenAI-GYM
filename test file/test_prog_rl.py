import random
import gym
from gym import wrappers
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
               
model = load_model("model/cartpole-dqn.h5")
#model.load_weights("model/cartpole-ddqn.h5")

env = gym.make('CartPole-v0').env
env= wrappers.Monitor(env, 'reinforcement files', force = True)
state = env.reset()
print('state is : ',state)
done = False
count = 0
count_left=0
count_right = 0

while count < 500:
    count = count_left + count_right 
    env.render()
    state = np.reshape(state, [1, 4])
    action = model.predict(state)
    print(action)
    if action[0][0] > action[0][1]:
        action = 0
        count_left +=1
    else:
        action = 1
        count_right +=1
    next_state,reward,done,info = env.step(action)
    state = next_state
    count = count + 1
    print('count: ', count)
print('count left: ',count_left)
print('count right: ',count_right)
print(count)
