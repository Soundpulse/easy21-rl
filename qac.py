import game
import numpy as np
import random
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import math

ITERATIONS = 100000

#QAC for full model (10*21*2)

hit = True
stick = False
actions = [hit, stick]

alpha = 0.01
beta = 0.01
theta = np.random.randn(420).reshape((420,1))
w = np.random.randn(420).reshape((420,1))

def psi(state, action):
    
    if state.player < 1 or state.player > 21:
        return np.zeros((420, 1))
    
    dealers = [int(state.dealer == x + 1) for x in range(0, 10)]
    players = [int(state.player == x + 1) for x in range(0, 21)]
    actions = [int(action == True), int(action == False)]
    
    psi = [1 if (i == 1 and j == 1 and k == 1) else 0
           for i in dealers for j in players for k in actions]

    return np.array(psi).reshape((420, 1)) 
    
def Q(state, action, weight):
    return np.matmul(psi(state, action).T, weight)

def V(q):
    return np.max(q, axis=2) 
    
def softmax(state, weight):
    
    allQ = [Q(state, a, weight) for a in actions]
    
    probs = np.exp(allQ) / np.sum(np.exp(allQ))
    
    if np.random.random() < np.sum(probs[0]):
        return hit
    else:
        return stick
        
def expectation_psi(state, weight):
    allQ = [Q(state, a, weight) for a in actions]
    
    probs = np.exp(allQ) / np.sum(np.exp(allQ))
    
    return probs[0] * psi(state, hit) + probs[1] * psi(state, stick)

def generate_Q(weight):
    
    Q_matrix = np.zeros((10, 21, 2))
    
    for i in range(0, 10, 1):
        for j in range(0, 21, 1):
            for k in range(0, 2, 1):
                Q_matrix[i][j][k] = Q(game.State(i, j, True), bool(k), weight)
                
    return Q_matrix
    
if __name__ == "__main__":    

    Q_star = np.load('Q_star.npy')

    for k in range(1, ITERATIONS):

        terminal = False
        
        state = game.initialise_state()
        action = softmax(state, theta)
        
        while not terminal:  
            # take action a, observe r, s'
            next_state, reward = game.step(state, action)
            # choose a' from s' using policy from Q
        
            terminal = next_state.terminal
            
            if not terminal:
                next_action = softmax(state, theta)
                delta = reward + Q(next_state, next_action, w) - Q(state, action, w)
            else:
                delta = reward - Q(state, action, w)
            
            gradient = psi(state, action) - expectation_psi(state, theta)
            theta += alpha * gradient * Q(state, action, w)
            w += beta * delta * psi(state, action)
            
            if not terminal:
                state = next_state
                action = next_action
                
        if k % 10000 == 0:
            print("MSE: " + str(round(np.sum((Q_star - generate_Q(theta)) ** 2),2)))

    game.visualise(V(generate_Q(theta)))       