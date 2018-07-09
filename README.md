## TOPIC 
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

