import game
import numpy as np
import random
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import math

N0 = 100
ITERATIONS = 10000000

hit = True
stick = False
actions = [hit, stick]

Q_matrix = np.zeros((10, 21, 2))
N_matrix = np.zeros((10, 21, 2))

def Q(state, action):
    return Q_matrix[state.dealer-1][state.player-1][int(action)]

def N(state, action):
    return N_matrix[state.dealer-1][state.player-1][int(action)]

def allQ(state):
    return Q_matrix[state.dealer-1][state.player-1]
    
def allN(state):
    return N_matrix[state.dealer-1][state.player-1]

def V(q):
    return np.max(q, axis=2) 
    
def epsilon_greedy(q, n):
    epsilon = N0 / (N0 + sum(n))
    
    if np.random.random() < epsilon:
        return random.choice(actions)
    else:
        return bool(np.argmax(q))

if __name__ == "__main__":
 
    for k in range(1, ITERATIONS):
        terminal = False
        
        state = game.initialise_state()
        action = epsilon_greedy(allQ(state), allN(state))
        
        history = [state, action]
        
        while not terminal:
            state, reward = game.step(state, action)
            action = epsilon_greedy(allQ(state), allN(state))
            
            terminal = state.terminal
            
            if terminal:
                state_action_pairs = zip(history[0::3], history[1::3])
                
                history.append(reward)
                history.append(state)
                
                Gt = sum(history[2::3])
                
                for s, a in state_action_pairs:
                    allN(s)[int(a)] += 1
                    alpha = 1/N(s,a)
                    allQ(s)[int(a)] += alpha * (Gt - Q(s, a))
          
            else:
                history.append(reward)
                history.append(state)
                history.append(action)
    
    #np.save('Q_star.npy',Q_matrix)    
    game.visualise(V(Q_matrix))