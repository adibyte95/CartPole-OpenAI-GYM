# importing the dependencies
import gym
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
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss',verbose=1, save_best_only=True)
no_of_observations = 500
min_score = 100


# generate the training data 
def generate_training_data(no_of_episodes):
    # initize the environment
    env = gym.make('CartPole-v0').env
    X = []
    y =[]
    prev_X = []
    prev_y= []
    left = 0
    right = 0

    for i_episode in range(no_of_episodes):
        observation = env.reset()
        score = 0
        for t in range(no_of_observations):
            action  =env.action_space.sample()
            observation, reward, done, info = env.step(action)
            prev_observation = observation
            score = score + reward
            if done:
                if score > min_score:
                    prev_X = X
                    prev_y = y
                    # if the game or episode is over
                    print('episode number: ', i_episode)
                    print("Episode finished after {} timesteps".format(t+1))
                else:
                    X= prev_X
                    y = prev_y
                break
            else:
                # if the episode is not over
                X.append(prev_observation)
                if action ==0:
                    left = left + 1
                    y.append([1,0])
                elif action ==1:
                    right = right +1
                    y.append([0,1])
        env.reset()
    
    print('left : ', left)
    print('right: ',right)
    # converting them into numpy array
    X = np.asarray(prev_X)
    y =np.asarray(prev_y) 

    # saving the numpy array
    np.save('X',X)
    np.save('y',y)
    
    # printing the size
    print('shape of X: ',X.shape)
    print('shape of target labels', y.shape)

# defines the model to be trained
def get_model():
    model = Sequential()
    model.add(Dense(128, input_dim=4))
    model.add(Activation('relu'))
    model.add(Dropout(.2))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(.2))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(.2))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(.2))

    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    

# trains the model
def train_model(model):
    # loading the training data from the disk
    X= np.load('X.npy')
    y = np.load('y.npy')
    # making train test split 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2, random_state = 42)
    # training the model
    model.fit(X_train,y_train,validation_data = [X_test,y_test],verbose = 1,
    callbacks=[checkpoint],
    epochs= 20, batch_size = 10000)
    # returns the model
    return model

# testing the model 
def testing():
    model = load_model('model.h5')
    env = gym.make('CartPole-v0').env
    observation = env.reset()
    # initial score
    score =0

    action = 0
    while (True):
        env.render()
        input('aa')
        data = np.asarray(observation)
        data = np.reshape(data, (1,4))
        output = model.predict(data)
        '''
        if output[0][0] >= output[0][1]:
            action = 1
        elif output[0][0] < output[0][1]:
            action = 0
        print(action)
        '''
        observation, reward, done, info = env.step(action)
        score = score  + reward 
        
        if done:
            print('game over!! your score is :  ',score)
            env.reset()
            break

#generate_training_data(1000)
#model = get_model()
#model = train_model(model)
testing()
