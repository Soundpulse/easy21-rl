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
ITERATIONS = 100000

hit = True
stick = False
actions = [hit, stick]

lmd = 0.8

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

def allE(state):
    return E_matrix[state.dealer-1][state.player-1]

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
        
        E_matrix = np.zeros_like(Q_matrix)
        
        state = game.initialise_state()
        action = epsilon_greedy(allQ(state), allN(state))

        while not terminal:
            next_state, reward = game.step(state, action)

            terminal = state.terminal
            
            if not terminal:
                next_action = epsilon_greedy(allQ(state), allN(state))
                delta = reward + Q(next_state, next_action) - Q(state, action)
            else:
                delta = reward - Q(state, action)
            
            allE(state)[int(action)] += 1
            allN(state)[int(action)] += 1
            
            alpha = 1/N(state,action)
            
            Q_matrix += alpha * delta * E_matrix
            E_matrix *= lmd
            
            if not terminal:
                state = next_state
                action = next_action
     
    game.visualise(V(Q_matrix))