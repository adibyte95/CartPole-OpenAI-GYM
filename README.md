## TOPIC [![HitCount](http://hits.dwyl.io/adibyte95/CartPole-OpenAI-GYM.svg)](http://hits.dwyl.io/adibyte95/CartPole-OpenAI-GYM) 
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.



## Approaches

## 1. random movements 
in this approach we choose an random action (left or right) given a paticular state of the enviroment. needles to say this approach performs very poorly
because it does not take into consideration the present state.<br/>
this approach because of its random nature is quite un predictable. On 10 trails runs the max time of survival is 118 timesteps and
acg survival time of about 21 time steps which is pretty bad. <br/>

![ramdom rendering gif ](https://github.com/adibyte95/OpenAI-GYM/blob/master/gif%20images/random.gif)
<br/>

## 2. using weight vector
in this approach we take a random weight vecotor of size 4 which is equal to the dimension of the state of the enviroment. A dot product is taken between the weight vector and state and depending upon the value of the output we take an action i.e either left or right. we see that this method outperforms the previous method but this method does not uses any machine learning algorithm. 
Resutls of this approach is very impressive. with proper number of games played this approach can last for more than 1000  time steps. <br/>
on 10 trail run of this algorithm max score achieved was 762 and avg score of about315.<br/>
Note that these can change with trail run and we can get even better results than this with appropriate parameter tuining
<br/>
![bruteforce rendering gif ](https://github.com/adibyte95/OpenAI-GYM/blob/master/gif%20images/brute_force.gif)


## 3. using deep neural networks
in this approach we take generate training data by randomly taking actions on the enviromnent . if the run is succesful that is the pole is balanced on the cart from more than 100 time steps we add this example to out training set. this approach aims that we can learn how to balance the pole by learning from good training examples. we then fit the model to this training data and try to predict the outcome that is action for any new observation.

![neural network rendering gif ](https://github.com/adibyte95/OpenAI-GYM/blob/master/gif%20images/nn.gif)


## 4. using deep Q networks
this uses a technique in which the model is rewarded is if makes correct action given the observations of a state and penalty otherwise. initially the model will not be very good at guessing the output but slowly it will become good at predicting the output. exploration and exploitation is carried simaltaneouly to find new improved solutions and to find the good solution in explored search space 

comparison how model performs in the begining and after a few epochs <br/>
<img src="https://github.com/adibyte95/OpenAI-GYM/blob/master/images/dqn_initial.png"> &nbsp; <img src = "https://github.com/adibyte95/OpenAI-GYM/blob/master/images/dqn_final.png">
<br/>
we can see that initialy the model was not able to perform very good, but eventually it learns from its mistakes and performs very good( 1199 is the upper time limit ...after this game is forcefully closed).even higher avg score can be achieved by training longer and increasing the time limit 

plot of score during various episodes
<img src = "https://github.com/adibyte95/OpenAI-GYM/blob/master/images/score_plot.png">

the pole was balanced on the cart for more than 2000 timeframes and outperforms all the approaches used above
![reinforcement rendering gif ](https://github.com/adibyte95/OpenAI-GYM/blob/master/gif%20images/reinforcement-gif.gif)

## references
<a href ="https://www.youtube.com/watch?v=3zeg7H6cAJw&t=3s">Sentdex</a><br/>
<a href = "https://www.youtube.com/watch?v=ZipAjLSNlQc">Machine Learning with Phil</a><br/>
<a href = "https://keon.io/deep-q-learning/">Medium blog</a><br/>

# Link to other OpenAI-GYM Enviroments
<a href='https://github.com/adibyte95/Mountain_car-OpenAI-GYM'>mountain car</a>
