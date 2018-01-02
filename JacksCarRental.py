import numpy as np
import random
import math

class Environment:
    def __init__(self, x):
        self.probs = [{} for _ in range(x)]
        self.rewards = [{} for _ in range(x)]
        self.states = x
        
    def defS(self, curState, actions):
        if type(actions) == int: #specify the number of actions (key value 0 to n - 1)
            actions = [i for i in range(actions)]
        for a in range(len(actions)):
            self.probs[curState][actions[a]] = {}
            self.rewards[curState][actions[a]] = {}
        
    def defSA(self, curState, action, prob, reward, nextState):
        self.probs[curState][action][nextState] = prob
        self.rewards[curState][action][nextState] = reward

    def validate(self):
        for state in range(self.states):
            for action in self.probs[state]:
                #print(type(action))
                sum = 0
                #print("iterating")
                #print(len(self.probs[state][action]))
                for nextState in self.probs[state][action]:
                    #print(nextState)
                    sum += self.probs[state][action][nextState]
                #print(sum)
                if sum  - 1.0 > 0.0001:
                    return False, state, action, sum
        return True

    def printer(self):
        print("there are " + str(self.states) + " states")
        print()
        for state in range(self.states):
            print("state" + str(state))
            for action in self.probs[state]:
                print("  action" + str(action), end = '    ')
                print(np.sum([self.probs[state][action][i] for i in self.probs[state][action]]))
                #print()

    def uniformPolicy(self): #returns policy that is unbiased in any action(pi)
        policy = []
        for state in range(self.states):
            l = {}
            for action in self.probs[state]:
                l[action] = (1/len(self.probs[state]))
            policy.append(l)
        return policy

    def randomPolicy(self): #deterministic policy
        policy = []
        for state in range(self.states):
            l = {}
            counter = 0
            x = random.randint(0, len(self.probs[state]) - 1)
            for action in self.probs[state]:
                l[action] = 1 if counter == x else 0
                counter += 1
            policy.append(l)
        return policy

    def fixedPolicy(self, x):
        policy = []
        for state in range(self.states):
            l = {}
            for action in self.probs[state]:
                l[action] = 1 if action == x else 0
            policy.append(l)
        return policy

    def evaluatePolicy(self, policy, gamma = 0.5, theta = 0.001): #iterative policy evaluation (returns V)
        V = [0 for _ in range(self.states)]
        delta = 1
        while (delta > theta):
            #print(V)
            delta = 0
            for s in range(self.states):
                if len(self.probs[s]) == 0:
                    continue
                v = V[s]
                t1 = 0
                for a in self.probs[s]:
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
            b = max(policy[s], key=policy[s].get)
            val = 0
            for a in self.probs[s]:
                temp = 0
                for sd in self.probs[s][a]:
                    temp += self.probs[s][a][sd] * (self.rewards[s][a][sd] + gamma * V[sd])
                if a == next(iter(self.probs[s])) or temp > val:
                    policy[s][max(policy[s], key=policy[s].get)] = 0
                    policy[s][a] = 1
                    val = temp
            if b != max(policy[s], key=policy[s].get):
                stable = False
        return policy, stable
                
#make environment

    
sx = 21

def printx(policy):
    for i in range(sx - 1, -1, -1):
        print("[", end = '')
        for j in range(sx):
            print(5 - max(policy[sx*i + j], key=policy[sx*i + j].get), end = ', ')
        print("]")


print("Making Env")
env = Environment(sx ** 2)
for state in range(sx ** 2):
    print("state" + str(state))
    actions = [i for i in range(11) if int(state / sx) + i - 5 >= 0 and int(state / sx) + i - 5 < sx
               and state % sx + 5 - i >= 0 and state % sx + 5 - i < sx]
    #actions = [i for i in range(max(0, 5 - int(state / sx)), min (11, 6 + state % sx))]
    env.defS(state, actions)
    for action in actions:
        nextState = state + (action - 5) * (sx - 1)
        for req1 in range(int(nextState / sx) + 1):
            for req2 in range(nextState % sx + 1):
                for ret1 in range(sx + req1 - int(nextState / sx)):
                    for ret2 in range(sx + req2 - nextState % sx):
                        
                        if req1 != int(nextState / sx):
                            prob = ((3 ** req1) / math.factorial(req1)) * np.exp(-3)
                        else:
                            prob = 1 - np.sum([((3 ** n) / math.factorial(n)) * np.exp(-3) for n in range(req1)])
                        if req2 != nextState % sx:
                            prob *= ((4 ** req2) / math.factorial(req2)) * np.exp(-4)
                        else:
                            prob *= 1 - np.sum([((4 ** n) / math.factorial(n)) * np.exp(-4) for n in range(req2)])

                        if ret1 != sx + req1 - int(nextState / sx) - 1:
                            prob *= ((3 ** ret1) / math.factorial(ret1)) * np.exp(-3)
                        else:
                            prob *= 1 - np.sum([((3 ** n) / math.factorial(n)) * np.exp(-3) for n in range(ret1)])
                            
                        if ret2 != sx + req2 - nextState % sx - 1:
                            prob *= ((2 ** ret2) / math.factorial(ret2)) * np.exp(-2)
                        else:
                            prob *= 1 - np.sum([((2 ** n) / math.factorial(n)) * np.exp(-2) for n in range(ret2)])
                            
                        endState = sx * (int(nextState / sx) - req1 + ret1) + nextState % sx - req2 + ret2
                        rew = 10 * (req1 + req2) - 2 * abs(5 - action)
                        if endState in env.probs[state][action]:
                            env.rewards[state][action][endState] = (env.probs[state][action][endState] * env.rewards[state][action][endState]
                                                                    + prob * rew) / (prob + env.probs[state][action][endState])
                            env.probs[state][action][endState] += prob
                        else:
                            env.probs[state][action][endState] = prob
                            env.rewards[state][action][endState] = rew
                            

#policy iteration

print(env.validate())

policy = env.fixedPolicy(5)
stable = False
printx(policy)
print()
while not stable:
    V = env.evaluatePolicy(policy, 0.9)
    policy, stable = env.improvePolicy(policy, V, 0.9)
    printx(policy)
    print()
