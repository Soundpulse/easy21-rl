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
    
def Q(state, action, weight):
    return np.matmul(psi(state, action).T, weight)

def V(q):
    return np.max(q, axis=2) 
    
def softmax(state, weight):
    
    allQ = [Q(state, a, weight) for a in actions]
    
    probs = np.exp(allQ) / np.sum(np.exp(allQ))
    
    return probs.reshape((2,))

def score_function(state, action, weight):

    probs = softmax(state, weight)
    
    expected_score = (probs[0] * psi(state, hit)) + (probs[1] * psi(state, stick))
    
    return psi(state, action) - expected_score

def softmax_policy(state, weight):
    
    probs = softmax(state, weight)
    
    if np.random.random() < probs[1]:
        return stick
    else:
        return hit

def generate_Q(weight):
    
    Q_matrix = np.zeros((10, 21, 2))
    
    for i in range(0, 10, 1):
        for j in range(0, 21, 1):
            for k in range(0, 2, 1):
                Q_matrix[i][j][k] = Q(game.State(i, j, True), bool(k), weight)
                
    return Q_matrix
    
def generate_EV(weight):
    
    V_matrix = np.zeros((10, 21))
    
    for i in range(0, 10, 1):
        for j in range(0, 21, 1):
            probs = softmax(game.State(i, j, True), weight)
            V_matrix[i][j] = (probs[0] * Q(game.State(i, j, True), hit, weight)) + (probs[1] * Q(game.State(i, j, True), stick, weight))
                
    return V_matrix
    
    
if __name__ == "__main__":    

    Q_star = np.load('Q_star.npy')

    for k in range(1, ITERATIONS):
        terminal = False
        
        state = game.initialise_state()
        action = softmax_policy(state, theta)
        
        history = [state, action]
        
        while not terminal:
            state, reward = game.step(state, action)
            action = softmax_policy(state, theta)
            
            terminal = state.terminal
            
            if terminal:
                state_action_pairs = zip(history[0::3], history[1::3])
                
                history.append(reward)
                history.append(state)
                
                Gt = sum(history[2::3])
                
                for s, a in state_action_pairs:
                    theta += alpha * Gt * score_function(s, a, theta)
          
            else:
                history.append(reward)
                history.append(state)
                history.append(action)
                
        if k % 10000 == 0:
            print("MSE: " + str(round(np.sum((V(Q_star) - generate_EV(theta)) ** 2),2)))

    game.visualise(generate_EV(theta))       