import numpy as np
import random
def maxminevi(states, actions, gamma, alpha, I, total_rewards, Pk): # I是總共有I個vi
    # Pk={(s, a):[s'1, s'2...]}
    v = np.zeros(len(I, states)) # 計算v0

    # 計算v1
    for s in states: # states資料結構?
        Max = 0
        for key, value in Pk:
            if key[0] != s:
                continue
            Sum = 0
            for next_s in value:
                Sum += next_s * v[0][next_s] # v[0, next_s]
            if Sum > Max:
                Max = Sum
        val = total_rewards[s, Max_a] + Max
        v[1][s] = (1-alpha) * val + alpha * v[0][s]

    i = 1
    while ((max(v[i]) - min(v[i-1])) - (min(v[i]) - max(v[i-1]))) <= (1 - alpha) * gamma: 
        i += 1
        for s in states:
            Max = 0
            Max_a = {}
            for key, value in Pk:
                if key[0] != s:
                    continue
                Sum = 0
                for next_s in value:
                    Sum += next_s * v[i-1][next_s] # v[i-1, next_s]
                if Sum > Max:
                    Max = Sum
                    Max_a[key[0]] = key[1]
            val = total_rewards[s, Max_a] + Max
            v[i][s] = (1-alpha) * val + alpha * v[i-1][s]

    Max_vi = 0
    for s in states:
        if v[i][s] > Max_vi:
            Max_vi = v[i][s]
            choose_action = Max_a[s]
    
    ran = random.randint(1, 6)
    if ran == 1:
        action_i = random.randint(0, len(actions)-1) # actions資料結構?
        return actions[action_i]
    else:
        return choose_action
    
    