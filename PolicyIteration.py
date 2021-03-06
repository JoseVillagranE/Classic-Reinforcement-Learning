# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:45:01 2020

@author: joser
"""


from GridWorld import GridWorldEnv
import numpy as np

class PolicyIteration:
    
    def __init__(self, env = GridWorldEnv(), discountingFactor=0.9, convergenceThreshold=1e-4,
                 iterationThresholdValue=1000, iterationThresholdPolicy=100, mode='prod'):
        
        self.env = env
        self.gamma = discountingFactor
        self.th = convergenceThreshold
        self.maxIterValue = iterationThresholdValue
        self.maxIterPolicy = iterationThresholdPolicy
        self.stateCount = self.env.get_stateSpace_len()
        self.actionCount = self.env.get_actionSpace_len()
        self.uniformActionProbability = 1.0/self.actionCount
        self.stateDict = env.stateDict
        self.mode = mode
        self.stateCount = self.env.get_stateSpace_len()
        self.V = np.zeros(self.stateCount)
        self.Q = [np.zeros(self.actionCount) for s in range(self.stateCount)]
        self.Policy = np.zeros(self.stateCount)
        self.totalReward = 0
        self.totalSteps = 0
        
    def reset_episode(self):
        self.totalReward = 0
        self.totalSteps = 0
        
    def compute_value_under_policy(self):
        self.V = np.zeros(self.stateCount)
        for i in range(self.maxIterValue):
            last_V = np.copy(self.V)
            for state_index in range(self.stateCount):
                current_state = self.env.stateSpace[state_index]
                for action in self.env.actionSpace:
                    next_state = self.env.next_state(current_state, action)
                    reward = self.env.compute_reward(next_state)
                    next_state_index = self.env.stateDict[next_state]
                    self.Q[state_index][action] = reward + self.gamma*last_V[next_state_index]
                
                if self.mode == "debug":
                    print("Q(s={}):{}".format(current_state, self.Q[state_index]))
                self.V[state_index] = max(self.Q[state_index])
            
            if np.sum(np.fabs(last_V - self.V)) <= self.th:
                print("Convergene Achieved in {}th iteration. Breaking V_Iteration loop!".format(i))
                break
            
    def iterate_policy(self):
        self.Policy = [np.random.choice(self.actionCount) for s in range(self.stateCount)]
        for i in range(self.maxIterPolicy):
            self.compute_value_under_policy()
            old_policy = self.Policy
            self.improve_policy()
            new_policy = self.Policy
            if np.all(old_policy==new_policy):
                print("Policy Convergence achieved in step ", i)
                break
            self.Policy = new_policy
        
        return self.Policy
    
    def improve_policy(self):
        self.Policy = np.argmax(self.Q, axis=1)
        if self.mode == "debug":
            print("Optimal Policy: ", self.Policy)
        
    
    def run_episode(self):
        
        self.reset_episode()
        obs = self.env.reset()
        while True:
            action = self.Policy[self.env.stateDict[obs]]
            new_obs, reward, done, _ = self.env.step(action)
            if self.mode=="debug":
                print("PrevObs:{}, Action:{}, Obs:{}, Reward:{}, Done:{}".format(obs, action, new_obs, reward, done))
            
            self.totalReward += reward
            self.totalSteps += 1
            if done:
                break
            else:
                obs = new_obs
        return self.totalReward
        
    
    def evaluate_policy(self, n_episodes=100):
        
        episodes_scores = []
        if self.mode == "debug": print("Running {} episodes!".format(n_episodes))
        for e, episode in enumerate(range(n_episodes)):
            score = self.run_episode()
            episodes_scores.append(score)
            if self.mode == "debug": print("Score in {} episode = {}".format(e, score))
        return np.mean(episodes_scores)
    
    def solve_mdp(self, n_episode=100):
        
        if self.mode == "debug":
            print("Initializing variables and setting environment...")
        self.iterate_policy()
        
        if self.mode=="debug":
            print("Scoring Policy...")
        
        return self.evaluate_policy(n_episode)
    

if __name__=="__main__":
    
    policyIteration = PolicyIteration(mode="debug")
    print("Policy Evaluation Score = ", policyIteration.solve_mdp())
