import game
import numpy as np
import random
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

ITERATIONS = 100000

#QAC for full model (10*21*2)

hit = True
stick = False
actions = [hit, stick]

alpha = 0.01
lmd = 0.6
theta = np.zeros((420, ))
w = np.zeros((420, ))
N_matrix = np.zeros((10,21,2))

def psi(state, action):
    
    if state.player < 1 or state.player > 21:
        return np.zeros((420, ))
    
    dealers = [int(state.dealer == x + 1) for x in range(0, 10)]
    players = [int(state.player == x + 1) for x in range(0, 21)]
    actions = [int(action == True), int(action == False)]
    
    psi = [1 if (i == 1 and j == 1 and k == 1) else 0
           for i in dealers for j in players for k in actions]

    return np.array(psi).reshape((420, ))
    
def Q(state, action, weight):
    return np.dot(psi(state, action), weight)

def N(state, action):
    return N_matrix[state.dealer - 1][state.player - 1][int(action)]
	
def increment_n(state, action):
	N_matrix[state.dealer - 1][state.player - 1][int(action)] += 1

def V(q):
    return np.max(q, axis=2) 
    
def softmax(state, weight):
    
    allQ = [Q(state, a, weight) for a in actions]

    probs = np.exp(allQ) / np.sum(np.exp(allQ))
    
    return probs

def score_function(state, action, weight):
    
    probs = softmax(state, weight)
	
    expected_score = (probs[0] * psi(state, hit)) + (probs[1] * psi(state, stick))
    
    return psi(state, action) - expected_score

def softmax_policy(state, weight):
    
    probs = softmax(state, weight)
    
    if np.random.random() < probs[int(hit)]:
        return hit
    else:
        return stick

def generate_Q(weight):
    
    Q_matrix = np.zeros((10, 21, 2))
    
    for i in range(0, 10, 1):
        for j in range(0, 21, 1):
            for k in range(0, 2, 1):
                Q_matrix[i][j][k] = Q(game.State(i+1, j+1, True), bool(k), weight)
                
    return Q_matrix
    
def generate_EV(weight):
    
    V_matrix = np.zeros((10, 21))
    
    for i in range(0, 10, 1):
        for j in range(0, 21, 1):
            probs = softmax(game.State(i+1, j+1, True), weight)
            V_matrix[i][j] = (probs[0] * Q(game.State(i+1, j+1, True), hit, weight)) + (probs[1] * Q(game.State(i+1, j+1, True), stick, weight))
                
    return V_matrix
    
    
if __name__ == "__main__":    

    Q_star = np.load('Q_star.npy')

    for k in range(1, ITERATIONS):
        terminal = False
        
        state = game.initialise_state()
        action = softmax_policy(state, theta)
		
        E_matrix = np.zeros_like(theta)
		
        while not terminal:
            next_state, reward = game.step(state, action)
            terminal = state.terminal
			
            if not terminal:
                next_action = softmax_policy(next_state, theta)
                delta = reward + Q(next_state, next_action, w) - Q(state, action, w)
            else:
                delta = reward - Q(state, action, w)
			
            increment_n(state, action)
            alpha = min(1/N(state, action), 0.01)
			
            advantage = reward - Q(state, action, w)
            theta += alpha * score_function(state, action, theta) * advantage

            E_matrix = np.add(lmd * E_matrix, psi(state, action))
            w += alpha * delta * E_matrix

            if not terminal:
                state = next_state
                action = next_action

        if k % 10000 == 0:
            print("MSE: " + str(round(np.sum((Q_star - generate_Q(theta)) ** 2),2)))

    game.visualise(V(generate_Q(theta)))       