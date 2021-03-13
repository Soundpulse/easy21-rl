import game
import numpy as np
import random
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

ITERATIONS = 300000

#LFA for full model (10*21*2)

hit = True
stick = False
actions = [hit, stick]

alpha = 0.01
epsilon = 0.05
lmd = 0.8
theta = np.random.randn(420).reshape((420,1))

def psi(state, action):
    
    if state.player < 1 or state.player > 21:
        return np.zeros((420, 1))
    
    dealers = [int(state.dealer == x + 1) for x in range(0, 10)]
    players = [int(state.player == x + 1) for x in range(0, 21)]
    actions = [int(action == True), int(action == False)]
    
    psi = [1 if (i == 1 and j == 1 and k == 1) else 0
           for i in dealers for j in players for k in actions]

    return np.array(psi).reshape((420, 1))
    
def Q(state, action, theta):
    return np.matmul(psi(state, action).T, theta)

def V(q):
    return np.max(q, axis=2) 
    
def epsilon_greedy(state, theta):
    
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    else:
        return bool(np.argmax([Q(state, a, theta) for a in actions]))
        
def generate_Q(weight):
    
    Q_matrix = np.zeros((10, 21, 2))
    
    for i in range(0, 10, 1):
        for j in range(0, 21, 1):
            for k in range(0, 2, 1):
                Q_matrix[i][j][k] = Q(game.State(i+1, j+1, True), bool(k), weight)
                
    return Q_matrix
    
if __name__ == "__main__":    

    Q_star = np.load('Q_star.npy')

    for k in range(1, ITERATIONS):

        terminal = False
        
        state = game.initialise_state()
        action = epsilon_greedy(state, theta)
        
        E_matrix = np.zeros_like(theta)
        
        while not terminal:  
            # take action a, observe r, s'
            next_state, reward = game.step(state, action)
            # choose a' from s' using policy from Q
        
            terminal = next_state.terminal
            
            if not terminal:
                next_action = epsilon_greedy(state, theta)
                delta = reward + Q(next_state, next_action, theta) - Q(state, action, theta)
            else:
                delta = reward - Q(state, action, theta)
            
            E_matrix = np.add(lmd * E_matrix, psi(state, action))

            theta += alpha * delta * E_matrix
            
            if not terminal:
                state = next_state
                action = next_action
                
        if k % 10000 == 0:
            print("MSE: " + str(round(np.sum((Q_star - generate_Q(theta)) ** 2),2)))

    game.visualise(V(generate_Q(theta)))       