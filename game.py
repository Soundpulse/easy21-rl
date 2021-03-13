import numpy as np
import random
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

# Initialise Deck
suit = ['r', 'b', 'b']
value = range(1,11,1)

deck = []
hit = True
stick = False
actions = [hit, stick]

for s in suit:
    for v in value:
        deck.append((s, v))
        

class State:
    def __init__(self, dealer, player, terminal):
        self.dealer = dealer
        self.player = player
        self.terminal = terminal
    
    def __repr__(self):
        return "(" + str(self.dealer) + "," + str(self.player) + "," + str(self.terminal) + ")"
        
def draw_card(draw_black_only = False):
    if not draw_black_only:
        return random.choice(deck)
    else:
        while True:
            card = random.choice(deck)
            if card[0] == 'b':
                return card
                
def initialise_state():
    return State(draw_card(True)[1], draw_card(True)[1], False)
    
def step(s, a):

    dealer = s.dealer
    player = s.player

    if a == hit:
        card = draw_card()
        if card[0] == 'b': player += card[1]
        elif card[0] == 'r': player -= card[1]
        
        if player < 1 or player > 21:
            return State(s.dealer, s.player, True), -1
        else:
            return State(dealer, player, False), 0
    elif a == stick:
        while dealer < 17:
            card = draw_card()
            if card[0] == 'b': dealer += card[1]
            elif card[0] == 'r': dealer -= card[1]
        
            if dealer < 1:
                return State(s.dealer, s.player, True), 1
        
        if dealer > 21:
            return State(s.dealer, s.player, True), 1
        elif dealer > player:
            return State(s.dealer, s.player, True), -1
        elif dealer == player:
            return State(s.dealer, s.player, True), 0
        elif dealer < player:
            return State(s.dealer, s.player, True), 1
            
def visualise(matrix):
    M = pd.DataFrame(matrix)

    df=M.unstack().reset_index()
    df.columns=["X","Y","Z"]
     
    # And transform the old column name in something numeric
    df['X']=pd.Categorical(df['X'])
    df['X']=df['X'].cat.codes
     
    # Make the plot
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    ax = fig.gca(projection='3d')

    df['X'] += 1
    df['Y'] += 1

    ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_yticks(np.arange(1, 22, 2))
    ax.set_xlabel("Dealer Showing", fontsize=14)
    ax.set_ylabel("Player Sum", fontsize=14)
    ax.set_zlabel("Expected Reward", fontsize=14)
    plt.show()
     
    # to Add a color bar which maps values to colors.
    surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    fig.colorbar( surf, shrink=0.5, aspect=5)
    plt.show()
     
    # Rotate it
    ax.view_init(30, 45)
    plt.show()
     
    # Other palette
    ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
    plt.show()