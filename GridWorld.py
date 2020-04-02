# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:24:00 2020

@author: joser
"""

import numpy as np

class GridWorldEnv():
    
    def __init__(self, gridsize=4, startState='00', terminalStates=['33'], ditches=['12'],
                ditchPenalty=-10, turnPenalty=-1, winReward=100, mode='prod'):
        
        self.mode=mode
        self.gridSize=min(gridsize, 9)
        self.create_stateSpace()
        self.actionSpace = [0, 1, 2, 3]
        self.actionDict = {0: 'UP', 1:'DOWN', 2:'LEFT', 3:'RIGHT'}
        self.startState = startState
        self.terminalStates = terminalStates
        self.ditches = ditches
        self.winReward = winReward
        self.ditchPenalty = ditchPenalty
        self.turnPenalty = turnPenalty
        self.stateCount = self.get_stateSpace_len()
        self.actionCount = self.get_actionSpace_len()
        self.stateDict = {k: v for k, v in zip(self.stateSpace, range(self.stateCount))}
        self.currentState = self.startState
        
        if self.mode == 'debug':
            print("State Space", self.stateSpace)
            print("State Dict", self.stateDict)
            print("Action Space", self.actionSpace)
            print("Action Dict", self.actionDict)
            print("Start State", self.startState)
            print("Terminal States", self.terminalStates)
            print("Ditches", self.ditches)
            print("WinReward:{}, TurnPenalty:{}, DitchPenalty:{}".format(self.winReward, self.turnPenalty, self.ditchPenalty))
        
    def create_stateSpace(self):
        
        self.stateSpace = []
        for row in range(self.gridSize):
            for col in range(self.gridSize):
                self.stateSpace.append(str(row)+ str(col))
                
    def set_mode(self, mode):
        self.mode = mode
    
    def get_stateSpace(self):
        return self.stateSpace
    
    def get_actionSpace(self):
        return self.actionSpace
    
    def get_actionDict(self):
        return self.actionDict
    
    def get_stateSpace_len(self):
        return len(self.stateSpace)
    
    def get_actionSpace_len(self):
        return len(self.actionSpace)
    
    def next_state(self, current_state, action):
        s_row = int(current_state[0])
        s_col = int(current_state[1])
        next_row = s_row
        next_col = s_col

        if action == 0: next_row = max(0, s_row - 1)
        if action == 1: next_row = min(self.gridSize-1, s_row+1)
        if action == 2: next_col = max(0, s_col - 1)
        if action == 3: next_col = min(self.gridSize - 1, s_col + 1)
            
        new_state = str(next_row) + str(next_col)
        if new_state in self.stateSpace:
            if new_state in self.terminalStates: self.isGameEnd = True
            if self.mode == 'debug':
                
                print("CurrentState:{}, Action:{}, NextState:{}".format(current_state, action, new_state))
            return new_state
        else:
            return current_state
    
    def compute_reward(self, state):
        
        reward = 0
        reward += self.turnPenalty
        if state in self.ditches: reward += self.ditchPenalty
        if state in self.terminalStates: reward += self.winReward
        return reward
    
    def reset(self):
        
        self.isGameEnd = False
        self.totalAccumulatedReward = 0
        self.totalTurns = 0
        self.currentState = self.startState
        return self.currentState
    
    def step(self, action):
        
        if self.isGameEnd:
            raise("Game is Over Exception")
        if action not in self.actionSpace:
            raise("Invalid Action Exception")
        self.currentState = self.next_state(self.currentState, action)
        obs = self.currentState
        reward = self.compute_reward(obs)
        done = self.isGameEnd
        if self.mode=='debug':
            print("Obs:{}, Reward:{}, Done:{}, TotalTurns:{}".format(obs, reward, done, self.totalTurns))
        return obs, reward, done, self.totalTurns