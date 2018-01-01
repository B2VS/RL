import numpy as np
import random

states = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 0]]

class Environment:
    def __init__(self, x):
        self.probs = [[] for _ in range(x)]
        self.rewards = [[] for _ in range(x)]
        self.states = x
        
    def defS(self, curState, actions):
        self.probs[curState] = [{} for _ in range(actions)]
        self.rewards[curState] = [{} for _ in range(actions)]
        
    def defSA(self, curState, action, prob, reward, nextState):
        self.probs[curState][action][nextState] = prob
        self.rewards[curState][action][nextState] = reward

    def validate(self):
        for state in range(self.states):
            for action in range(len(self.probs[state])):
                sum = 0
                #print("iterating")
                #print(len(self.probs[state][action]))
                for nextState in self.probs[state][action]:
                    #print(nextState)
                    sum += self.probs[state][action][nextState]
                #print(sum)
                if sum != 1.0:
                    return False
        return True

    def printer(self):
        print("there are " + str(self.states) + " states")
        print()
        for state in range(self.states):
            print("state" + str(state))
            for action in range(len(self.probs[state])):
                print("  action" + str(action), end = '    ')
                for nextState in self.probs[state][action]:
                    print(str(nextState), end = ', ')
                print()

    def uniformPolicy(self): #returns policy that is unbiased in any action(pi)
        policy = []
        for state in range(self.states):
            l = []
            for action in range(len(self.probs[state])):
                l.append(1/len(self.probs[state]))
            policy.append(l)
        return policy

    def randomPolicy(self): #deterministic policy
        policy = []
        for state in range(self.states):
            l = [0 for _ in range(len(self.probs[state]))]
            if len(self.probs[state]) > 0:
                l[random.randint(0, len(self.probs[state]) - 1)] = 1
            policy.append(l)
        return policy

    def evaluatePolicy(self, policy, gamma = 0.5, theta = 0.001): #iterative policy evaluation (returns V)
        V = [0 for _ in range(self.states)]
        delta = 1
        while (delta > theta):
            print([round(i, 2) for i in V])
            delta = 0
            for s in range(self.states):
                if len(self.probs[s]) == 0:
                    continue
                v = V[s]
                t1 = 0
                for a in range(len(self.probs[s])):
                    t2 = 0
                    for sd in self.probs[s][a]:
                        t2 += self.probs[s][a][sd] * (self.rewards[s][a][sd] + gamma * V[sd])
                    t1 += policy[s][a] * t2
                V[s] = t1
                delta = max(delta, abs(v - V[s]))
        return V

    def improvePolicy(self, policy, V, gamma = 0.5): #policy improvement
        stable = True
        for s in range(self.states):
            if len(self.probs[s]) == 0:
                continue
            b = np.argmax(policy[s])
            val = 0
            for a in range(len(self.probs[s])):
                temp = 0
                for sd in self.probs[s][a]:
                    temp += self.probs[s][a][sd] * (self.rewards[s][a][sd] + gamma * V[sd])
                if a == 0 or temp > val:
                    policy[s][np.argmax(policy[s])] = 0
                    policy[s][a] = 1
                    val = temp
            if b != np.argmax(policy[s]):
                stable = False
        return policy, stable
                
#make environment

env = Environment(15)
for row in range(len(states)):
    for col in range(len(states[row])):
        if states[row][col] == 0: #terminal state
            continue
        env.defS(states[row][col], 4)
        for action in range(4):
            if action == 0: #left
                nextState = states[row][max(col - 1, 0)]
            elif action == 1: #up
                nextState = states[max(row - 1, 0)][col]
            elif action == 2: #right
                nextState = states[row][min(col + 1, 3)]
            else: #down
                nextState = states[min(row + 1, 3)][col]
            env.defSA(states[row][col], action, 1, -1, nextState)

#policy iteration

policy = env.randomPolicy()
print(policy)
print()
stable = False
while not stable:
    V = env.evaluatePolicy(policy)
    policy, stable = env.improvePolicy(policy, V)
    print(policy)
