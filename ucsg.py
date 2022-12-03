import itertools
import math
from tkinter import N
import numpy as np
import sympy 
from sympy import *
import gfootball.env as football_env

env = football_env.create_environment(env_name='5_vs_5', representation='pixels', render='True', number_of_left_players_agent_controls=5 )

# initial phase
# Update the confidence set
# Optimistic Planning
# Execute Policies

#inner maximization for EVI
def inner_maximization(p_sa_hat, confidence_bound_p_sa, rank): 

    # print('rank', rank)
    p_sa = np.array(p_sa_hat)
    p_sa[rank[0]] = min(1, p_sa_hat[rank[0]] + confidence_bound_p_sa / 2)
    rank_dup = list(rank)
    last = rank_dup.pop()
    # Reduce until it is a distribution (equal to one within numerical tolerance)
    while sum(p_sa) > 1 + 1e-9:
        # print('inner', last, p_sa)
        p_sa[last] = max(0, 1 - sum(p_sa) + p_sa[last])
        last = rank_dup.pop()
    # print('p_sa', p_sa)
    return p_sa



def UCSG(n_states, n_actions, T, delta):
    t = 1
    # Initial state
    vk = np.zeros((n_states, n_actions)) #vk(s,a)
    total_numbers = np.zeros((n_states, n_actions, n_states)) # nk(s,a,s')
    total_rewards = np.zeros(n_states, n_actions) #maximin
    nk = np.ones((n_states, n_actions)) #nk(s,a)
    #initial state
    st = env.reset()
    # Initialize phase k
    for k in itertools.count():
        t_k = t
        #Per-phase visitations
        #在之後的execute policy做係數update
        #compute estimates
        # delta:我們自己設定的容許誤差值    
        delta_1 = delta/(2*n_states*n_states*n_actions*np.log2(T))
        p_hat = vk.reshape((n_states, n_actions, 1))/n_states  #拆解
        r_hat = total_rewards / np.clip(vk, 1, None)
        upper_conf = 1
        lower_conf = 0

        # Update the confidence set
        #confidence_bound_1 = np.clip(np.sqrt((2*n_states*np.log(1/delta)/n_states))+ p_hat, 0, 1)         
        confidence_bound_1 = np.sqrt((2*n_states*np.log(1/delta)/n_states))+ p_hat
        confidence_bound_2_2 = np.clip(np.sqrt(np.log(6/delta_1)/(2*n_states))+p_hat, None, np.sqrt(2*p_hat*(1-p_hat)*np.log(6/delta_1)/n_states) + 7*np.log(6/delta_1)/(3*(n_states-1))+p_hat)
        #confidence_bound_2_2 = np.clip(confidence_bound_2_2, 0, 1)
        a = np.sqrt(2*np.log(6/delta_1)/(n_states-1))
        b = np.sqrt(p_hat*(1-p_hat))
        c1 = (a+b)**2
        c2 = (a-b)**2
        xr = np.clip((1+np.sqrt(1-4*c1))/2, (1-np.sqrt(1-4*c1))/2, 1)
        xl = np.clip((1+np.sqrt(1-4*c2))/2, 0 ,(1-np.sqrt(1-4*c2))/2)
        confidence_bound_2_1 = np.clip(xr, 0 ,xl)
        tmp = confidence_bound_1 + confidence_bound_2_2
        #print('confidencebound1', confidence_bound_1)
        #print('confidencebound2', confidence_bound_2_2)
        confidencebound = [ (tmp[i]) for i in range(0, len(tmp)) if tmp[i] not in tmp[:i] ]
        #print('confbound', confidencebound)        
        

        # alpha, gamma
        #Optimistic Planning
        pi1_k, m_k = evi()
        
        #execute policy
        ac = pi1_k[st]
        while(vk[st,ac] != total_numbers[st,ac]):
            next_st, reward, done, info= env.step()
            yield(t, st, ac, next_st, reward)

            #update
            vk[st, ac] =  vk[st, ac] + 1 #update vk(s,a)
            total_rewards[st,ac] += reward # 即時reward
            nk[st, ac] = max(1,  nk[st, ac] + 1) #update nk(s,a)
            t += 1 
            st = next_st
            ac = pi1_k[st]
            total_numbers[st, ac, next_st] += 1 #update nk(s, a, s')








'''if __name__ == '__main__':
    eps = 0.1  # epsilon
    alpha = 0.1  # learning rate
    '''
