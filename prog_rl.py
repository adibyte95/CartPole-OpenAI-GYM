import gym
import numpy as np
import random
import keras
from keras import backend as k 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import load_model
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def replay(self, batch_size,episodes ):
        
        # if the length of the memorgy is less than the batch size then simple return 
        if len(self.memory) < batch_size:
            return
        
        # else
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma *np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            # saving the model
            if episodes %100 == 0:
                self.model.save('model_rl.h5')
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

## training portion


# initialize gym environment and the agent
env = gym.make('CartPole-v0').env
agent = DQNAgent(4,2)
episodes = 1000
score = []
# Iterate the game
for e in range(episodes):
    # reset state in the beginning of each game
    state = env.reset()
    state = np.reshape(state, [1, 4])
    # time_t represents each frame of the game
    # Our goal is to keep the pole upright as long as possible until score of 500
    # the more time_t the more score
    for time_t in range(5000):
        # turn this on if you want to render
        # env.render()
        # Decide action
        action = agent.act(state)
        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived and -10 if the pole falls
        next_state, reward, done, _ = env.step(action)
        reward  = reward  if not done else - 100
        next_state = np.reshape(next_state, [1, 4])
        # Remember the previous state, action, reward, and done
        agent.remember(state, action, reward, next_state, done)
        # make next_state the new current state for the next frame.
        state = next_state
        # done becomes True when the game ends
        # ex) The agent drops the pole
        if done:
            # print the score and break out of the loop
            print("episode: {}/{}, score: {}"
                    .format(e, episodes, time_t))
            score.append(time_t)
            break
    # train the agent with the experience of the episode
    agent.replay(32,e)

print('avg score : ', sum(score)/1000)
print('maximum score: ', max(score))
print('min score: ', min(score))