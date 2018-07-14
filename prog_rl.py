import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt

EPISODES = 1000


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
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=4))
        model.add(Activation('relu'))

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future  = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(Q_future)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env._max_episode_steps = None
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("model/cartpole-ddqn.h5")
    done = False
    batch_size = 128

    scores = []
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        flag = 0
        for time in range(1200):
            # uncomment this to see the actual rendering 
            # env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                flag = 1
                scores.append(time)
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if flag == 0:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))      
        if e % 100 == 0:
            print('saving the model')
            agent.save("model/cartpole-dqn.h5")
            # saving the figure
            plt.plot(scores)
            plt.savefig('score_plot')
