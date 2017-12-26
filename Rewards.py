import numpy
import matplotlib.pyplot as plt
import random

tasks = 100
plays = 100
ways = 5
tau = 1
alpha = 0.2

qstar = numpy.random.randn(tasks, 10)

def updater(q, cnt, v, e, method):
    if method == 1: #weighted average
        ep = [numpy.exp(i / tau) for i in q]
        prob = [i / sum(ep) for i in ep]
        choice = numpy.random.choice(10, 1, p = prob).tolist()[0]
    else:
        if random.uniform(0, 1) < e:
            choice = random.randint(0, 9)
        else:
            options = [j for j, k in enumerate(q) if k == max(q)]
            choice = random.choice(options)
    rew = numpy.random.randn() + qstar[v][choice]
    if method == 2: #stochastic update
        q[choice] = q[choice] + alpha * (rew - q[choice])
    else:
        q[choice] = (q[choice] * cnt[choice] + rew) / (cnt[choice] + 1)
    cnt[choice] += 1
    return rew

reward = numpy.zeros((ways, plays))
correct = numpy.zeros((ways, plays))

for i in range(tasks):
    print(i)
    q = numpy.zeros((ways, 10))
    q += 5
    cnt = numpy.zeros((ways, 10))
    l2 = qstar[i].tolist()
    for j in range(plays):
        reward[0][j] += updater(q[0], cnt[0], i, 0, 0)
        reward[1][j] += updater(q[1], cnt[1], i, 0.01, 0)
        reward[2][j] += updater(q[2], cnt[2], i, 0.1, 0)
        #reward[3][j] += updater(q[3], cnt[3], i, 1, 1)
        reward[4][j] += alpha * (updater(q[4], cnt[4], i, 0.1, 2) - reward[4][j])
        #for stochastic environment
        #qstar = qstar + numpy.random.randn(tasks, 10) * 0.1
        for k in range(ways):
            l1 = q[k].tolist()
            if l1.index(max(l1)) == l2.index(max(l2)):
                correct[k][j] += 1

for i in range(plays - 1):
    for j in range(ways):
        reward[j][i + 1] += reward[j][i]

for i in range(plays):
    for j in range(ways):
        reward[j][i] /= tasks*(i + 1)
        

for i in range(plays):
    for j in range(ways):
        correct[j][i] /= tasks

plt.plot(range(plays), reward[0].tolist(), 'r', range(plays), reward[1].tolist(),
         'b', range(plays), reward[2].tolist(), 'g', range(plays), reward[3].tolist(), 'c',
         range(plays), reward[4].tolist(), 'm')
plt.show()

plt.plot(range(plays), correct[0].tolist(), 'r', range(plays), correct[1].tolist(),
         'b', range(plays), correct[2].tolist(), 'g', range(plays), correct[3].tolist(), 'c',
         range(plays), correct[4].tolist(), 'm')
plt.show()
    

        



