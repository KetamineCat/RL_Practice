import numpy
import gym #OpenAI's gym handles pretty much all the RL-related processes 
import random2

env = gym.make("FrozenLake-v0") #from the gym's library

action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeroes((state_size, action_size))
print(qtable)

#Hyperparameters (
total_episodes = 10000
learning_rate 0.8
max_steps = 99 #Max steps nper episode
gamma = 0.95 #Discounting rate (comes in at the gamma^n * R_(t+n+..))

#Exploration parameters
epsilon = 0.1
max_epsilon = 1.0
min_epsilon = 0.01 #I wonder what the lowest possible epsilon value is, read gym sourcecode)
decay_rate = 0.005 #Rate of change form exploration to exploitation over episodes

#Reward list
rewards = []

#Rules for continuous learning in FrozenLake 
for episode in range(total_episodes):
    #Resets the environment (?)
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
    [Continue from here]
