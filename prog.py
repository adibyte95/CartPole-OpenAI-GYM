# importing the dependencies
import gym
import numpy as np 
import random
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from sklearn.model_selection import train_test_split

'''
NOTE:
action:
0 for left 
1 for right
'''

def generate_training_data(no_of_episodes):
    # initize the environment
    env = gym.make('CartPole-v0')
    X = []
    y =[]
    for i_episode in range(no_of_episodes):
        observation = env.reset()
        for t in range(100):
            env.render()
            action  = random.randint(0, 1)
            prev_observation = observation
            observation, reward, done, info = env.step(action)
            if done:
                # if the game or episode is over
                print('episode number: ', i_episode)
                print("Episode finished after {} timesteps".format(t+1))
                break
            else:
                # if the episode is not over
                X.append(prev_observation)
                y.append(action)

    # converting them into numpy array
    X = np.asarray(X)
    y =np.asarray(y) 

    # saving the numpy array
    np.save('X',X)
    np.save('y',y)
    
    # printing the size
    print('shape of X: ',X.shape)
    print('shape of target labels', y.shape)

# defines the model to be trained
def get_model():
    model = Sequential()
    model.add(Dense(512, input_dim=4))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(1))
    model.add(Activation('softmax'))
    
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    

def train_model(model):
    # loading the training data from the disk
    X= np.load('X.npy')
    y = np.load('y.npy')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2, random_state = 42)
    model.fit(X_train,y_train,validation_data = [X_test,y_test],verbose = 1,
    epochs= 100, batch_size = 1000)
    return model

model = get_model()
mdoel = train_model(model)

#generate_training_data(5000)