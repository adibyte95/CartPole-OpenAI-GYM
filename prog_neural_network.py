# importing the dependencies
import gym
from gym import wrappers

import numpy as np 
import random
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

'''
NOTE
action:
0 for left 
1 for right
'''
checkpoint = ModelCheckpoint('model/model_dnn.h5', monitor='val_loss',verbose=1, save_best_only=True)
no_of_observations = 500
min_score = 100

# generate the training data 
def generate_training_data(no_of_episodes):
    print('generating training data')
    # initize the environment
    env = gym.make('CartPole-v1').env
    X = []
    y =[]
    left = 0
    right = 0

    for i_episode in range(no_of_episodes):
        prev_observation = env.reset()
        score = 0
        X_memory  = []
        y_memory = []
        for t in range(no_of_observations):
            action = random.randrange(0,2)
            
            ## debugging code
            '''
            if action == 0:
                left = left + 1
            else:
                right = right + 1
            '''
            new_observation,reward,done,info = env.step(action)
            score = score + reward
            X_memory.append(prev_observation)
            y_memory.append(action)
            prev_observation = new_observation
            if done:
                if score >min_score:
                    for data in X_memory:
                        X.append(data)
                    for data in y_memory:
                        y.append(data)
                    print('episode : ',i_episode, ' score : ',score)
                break
        env.reset()
    #debugging code
    '''
    print('left : ', left)
    print('right: ',right)
    '''
    # converting them into numpy array
    X = np.asarray(X)
    y =np.asarray(y) 

    # saving the numpy array
    np.save('data/X',X)
    np.save('data/y',y)
    
    # printing the size
    print('shape of X: ',X.shape)
    print('shape of target labels', y.shape)

# defines the model to be trained
def get_model():
    model = Sequential()
    model.add(Dense(128, input_dim=4))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
     
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return model
    

# trains the model
def train_model(model):
    # loading the training data from the disk
    X= np.load('data/X.npy')
    y = np.load('data/y.npy')
    # making train test split 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2, random_state = 42)
    print('X_train: ',X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)
    # training the model
    model.fit(X_train,y_train,validation_data = [X_test,y_test],verbose = 1,
    callbacks=[checkpoint],
    epochs= 20, batch_size = 10000,shuffle =True)
    # returns the model
    return model

# testing the model 
def testing(model):
    #model = load_model('model/model.h5')
    env = gym.make('CartPole-v1').env
    env= wrappers.Monitor(env, 'nn_files', force = True)
    observation = env.reset()
    no_of_rounds = 10
    max_rounds = no_of_rounds
    min_score = 1000000
    max_score = -1
    avg_score = 0

    # playing a number of games
    while (no_of_rounds > 0):
        # initial score
        score =0
        action = 0
        prev_obs = []
        while (True):
            env.render()
            if len(prev_obs) == 0:
                action = random.randrange(0,2)
            else:
                data = np.asarray(prev_obs)
                data = np.reshape(data, (1,4))
                output = model.predict(data)
                # checking if the required action is left or right
                if output[0][0] >= .5:
                    action = 1
                elif output[0][0] < .5:
                    action = 0
            
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            # calculating total reward
            score = score  + reward 
            
            if done:
                # if the game is over
                print('game over!! your score is :  ',score)
                if score > max_score:
                    max_score = score
                elif score < min_score:
                    min_score = score
                avg_score +=score 
                env.reset()
                break
        no_of_rounds = no_of_rounds - 1
        # stats about scores 
        if no_of_rounds == 0:
            print('avg score : ',avg_score/max_rounds)
            print('max score: ', max_score)
            print('min score: ',min_score)

# calling the functions
generate_training_data(50000)
model = get_model()
model = train_model(model)
testing(model)
