import numpy
import gym #OpenAI's gym handles pretty much all the RL-related processes 
import random2

env = gym.make("FrozenLake-v0") #from the gym's library

action_size = env.action_space.n
state_size = env.observation_space.n

qtable = numpy.zeroes((state_size, action_size))
print(qtable)

#Parameters for each game and the learning rate/decay of the agent
total_episodes = 10000
learning_rate = 0.8 #epsilon
max_steps = 99 #Max steps per episode
gamma = 0.95 #Discounting rate 

epsilon = 0.1
max_epsilon = 1.0
min_epsilon = 0.01 #I wonder what the lowest possible epsilon value is i.e. I should RTFM
decay_rate = 0.005 #Rate of change for the discounting rate

#Reward list
rewards = []

#Rules for continuous learning in FrozenLake 
for episode in range(total_episodes):
    #i.e. every episode is fresh
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps): 
        exp_tradeoff = random.uniform(0,1)
        
        #defining exploitation
        if exp_tradeoff > epsilon:
        action = numpy.argmax(qtable[state,:])
        
        #defining exploration
        else:
        action = env.action_space.sample()
    
        new_state, reward, done, info = env.step(action) #RTFM OpenAI gym

        #Using the Temporal Difference equation below
        #Q(s,a):= Q(s,a) + learning_rate[reward + discounting_rate * Q_max(s,a) - Q(s,a)]
        
        #Weirdly enough epsilon is never explicitly defined in any of Simonini's texts and equations, so
        #the relationship between the learning rate and the epsilon is another mystery to solve for me
    
        qtable[state,action] = qtable[state,action] + learning_rate * (reward + gamma * numpy.max(qtable[new_state, :]) - qtable[state,action])
    
        total_rewards += reward
    
        state = new_state # Placing this anywhere but near the bottom is most likely gonna break the code
    
        #Done = dead
        if done == True:
            break
  
print("Score over time: " + str(sum(rewards)/total_episodes))
print(qtable)

env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = Flase
    print("FROZENLAKEFROZENLAKEFROZENLAKEFROZENLAKEFROZENLAKEFROZENLAKEFROZENLAKE")
    print("EPISODE ", episode)
    
    #Exploit o'clock
    for step in range(max_steps):
        action = numpy.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            print("Number of steps: ", step)
            break
            
        state = new_state
env.close()
