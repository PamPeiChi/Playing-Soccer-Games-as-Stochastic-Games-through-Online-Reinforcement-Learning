
# import gfootball.env as football_env
# import numpy as np
# import gym
# from gfootball.env import football_action_set


# """
# env = football_env.create_environment(env_name='5_vs_5', representation='raw', render='True',channel_dimensions=(10,15), number_of_left_players_agent_controls=5 )
# state = env.reset()
# steps = 0
# while True:
#   obs, rew, done, info = env.step(env.action_space.sample())
#   steps += 1
#   if steps % 100 == 0:
#     # print(obs, rew, done, info)
#     break
#   if done:
#     break
# # print("Steps: %d Reward: %.2f" % (steps, rew))
# # before 11/5
# import numpy as np
# R = {}
# Pk = {}
# S = []
# st = 1
# V = {}
# A1 = []
# A2 = []
# nstates = 100
# M = {}
# Lambda = 0.01
# alpha = 0.01
# nsteps = 100
# def value_iteration(st, Pk, V, t):
#   for a1 in A1:
#     for a2 in A2:
#       V_t = []
#       for st_1 in S:
#         V_t = Pk[st,a1,a2] + V[t-1,st_1]
#       v = max(V_t)
#       V[t,st] = v + R[st,a1,a2]
#   return V
# def maxmin_EVI(M, Lambda, alpha):
#   V[0] = np.zeros(nstates)
#   t = 0
#   while True:
#     for st in S:
#       val = maxmin_EVI(st,Pk,V,t)
#       V[t,st] = (1-alpha) * val[t,st] + alpha * V[t-1, st]
#     SP = max(V[t]) - min(V[t])
#     if SP <= ((1-alpha) * Lambda):
#       break
#     t = t + 1
# """

# from gfootball.env import observation_preprocessing
# import tensorflow as tf
# import sonnet as snt

# print("tensorflow version: ",tf.__version__)
# tf.disable_v2_behavior()
# # observation[0]['frame'].shape = (720,1280,3)

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# def network_fn(frame):
#   # Convert to floats.
#     print(frame.shape) # pixles: shape(72,96,3), frame: shape(720,960,3,1)
#     tf.reset_default_graph()
#     frame = tf.cast(frame, dtype= tf.float16)
#     frame /= 255
#     frame = tf.expand_dims(frame, axis = -1) # [720, 1280, 3, 1]
#     print(frame.shape.as_list())
 
#   #with tf.variable_scope('convnet'):

  
#     conv_out = frame
#     conv_layers = [(16,2), (16, 2), (32, 2), (32, 2)]
#     for i, (num_ch, num_blocks) in enumerate(conv_layers):
#       # Downscale.
#       print("num_channels", num_ch)
#       print("num_blocks",num_blocks)
#       conv_out = snt.Conv2D(num_ch, 4, stride=1, padding='SAME')(conv_out) # 4th dim = num_ch
#       print(conv_out.shape.as_list()) # [720, 1280, 3, 16], [720, 640, 2, 16], [720, 320, 1, 32]
#       conv_out = tf.nn.pool(
#           conv_out,
#           window_shape=[4, 4],
#           pooling_type='MAX',
#           padding='SAME',
#           strides=[3, 3]) # 2nd_dim =
#       print("for loop: ") 
#       print(conv_out.shape.as_list()) # [720, 640, 2, 16], [720,320, 1, 16], [720, 160, 1, 32]
#       # Residual block(s).
#       for j in range(num_blocks):
#         #with tf.variable_scope('residual_%d_%d' % (i, j)):
#           block_input = conv_out
#           conv_out = tf.nn.relu(conv_out) 
#           print("relu1:")
#           print(conv_out.shape.as_list()) # 
#           conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
#           print("Conv2D1:")
#           print(conv_out.shape.as_list()) # 
#           conv_out = tf.nn.relu(conv_out)
#           print("relu2:")
#           print(conv_out.shape.as_list()) # 
#           conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
#           print("Conv2D1:")
#           print(conv_out.shape.as_list()) # [720, 640, 2, 16], [720, 160, 1, 32], [720, 80, 1, 32], 
#           conv_out += block_input

#     conv_out = tf.nn.relu(conv_out)
#     conv_out = snt.BatchFlatten()(conv_out)
#     print("batchFlatten:")
#     print(conv_out.shape.as_list()) # [720, 2560]
#     conv_out = snt.Linear(256)(conv_out) # define 2nd dim size
#     print("Linear:")
#     print(conv_out.shape.as_list())
#     conv_out = tf.nn.relu(conv_out) # [720, 256]

#     return conv_out


# class ObservationStacker(object):
#     def __init__(self, stacking):
#       self._stacking = stacking
#       self._data = []

#     def get(self, observation):
#       sess = tf.compat.v1.Session()
#       sess.run(tf.global_variables_initializer())
#       sess.run(tf.local_variables_initializer())
#       observation = observation.eval(session = sess)

#       if self._data:
#         self._data.append(observation)
#         self._data = self._data[-self._stacking:]
#       else:
#         self._data = [observation] * self._stacking
#       return np.concatenate(self._data, axis=-1)

#     def reset(self):
#       self._data = []
# class DummyEnv(object):
#       # We need env object to pass to build_policy, however real environment
#   # is not there yet.

#   def __init__(self, action_set, stacking):
#     self.action_space = gym.spaces.Discrete(
#         len(action_set))
#     print(self.action_space)
#     # pixel size: 
#     #self.observation_space = gym.spaces.Box(
#     #    0, 255, shape=[72, 96, 4 * stacking], dtype=np.uint8)

#     # observation["frame"]  size:
#     self.observation_space = gym.spaces.Box(
#       0, 255, shape=[720, 1280, 3], dtype=np.uint8
#     )
#     print(self.observation_space)

# class policy_network(object):

#     def __init__(self, build_policy):
#         #Discrete(19)
#         #Box(720, 1280, 3)
      
#         self.policy = np.asarray([list(build_policy.observation_space.shape), list(build_policy.action_space.shape)])
#         print("policy", self.policy)
#     def step(self, stateid):
#         actions = self.policy[stateid] # 這裡不知道怎麼寫

# # 1. 本身維度就小的, 但具體如何downgrade部清楚的input: pixels: shape(72,96)
    
# # 2. 仿造ppo寫法使用raw裡面的frame (1280,720,3)    


# stacking = 1
# env = football_env.create_environment(env_name="5_vs_5", 
# representation='raw', render = True)
# state = env.reset()
# # convert frame to nn
# nn_network = network_fn(state[0]['frame'])
# action_set = football_action_set.get_action_set({'action_set': 'default'})
# print("action_set",action_set)
# # create a policy network, state = observation["frame"] ?? maybe should be nn_network
# build_policy = DummyEnv(action_set, stacking)
# #policy = np.array(build_policy.observation_space, build_policy.action_space)
# #print("policy_shape",policy.shape)
# policy = policy_network(build_policy)
# # (start) 初始化policy的值...??

# # (end) 初始化policy的值...??


# steps = 0

# # store observation into a stack
# stacker = ObservationStacker(stacking)
# while True:
#   # take an action, observe env and receive reward
#   next_state, rew, done, info = env.step(env.action_space.sample())
#   print(next_state[0]['frame'])
#   # convert raw observation into frame
#   # frame = observation_preprocessing.generate_smm(next_state)
#   new_nn_network = network_fn(next_state[0]['frame'])
#   # store new frame
#   frame_stack = stacker.get(new_nn_network)
#   # print("frame_stack", frame_stack.shape) # frame_stack (1, 72, 96, 4) => pixel


#   steps += 1
#   if steps % 100 == 0:
#     # print(obs, rew, done, info)
#     break
#   if done:
#     break













# ucsg.py
import itertools
import math
from tkinter import N
import numpy as np
import gfootball.env as football_env
import random
from gfootball.env import football_action_set
from Env_discrete import simplify
import matplotlib.pyplot as plt
from itertools import permutations


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


def maxminevi(states, actions, gamma, alpha, I, total_rewards, Pk, st_i): # I是總共有I個vi, Pk是三維的array, st_i是現在state的index
    v = np.zeros((I, len(states))) # 計算v0

    # 計算v1
    for s in range(len(states)):
        Max = -1
        for a in range(len(actions)):
            Sum = sum(Pk[s][a])
            if Sum >= Max:
                Max = Sum
                Max_a_1 = a
        val = total_rewards[s, Max_a_1] + Max
        v[1][s] = (1-alpha) * val + alpha * v[0][s]

    i = 1
    # print((max(v[i]) - min(v[i-1])) - (min(v[i]) - max(v[i-1])))
    while ((max(v[i]) - min(v[i-1])) - (min(v[i]) - max(v[i-1]))) > (1 - alpha) * gamma: 

        i += 1
        if i == I:
            break
        # 選一個Pk(s,a)讓值最大。Pk(s,a)是一個一維array
        Max_in = -1
        for s in range(len(states)):
            for a in range(len(actions)):
                Sum = 0
                for next_s in range(len(states)):
                    Sum += Pk[s][a][next_s] * v[i-1][next_s]
                if Sum >= Max_in:
                    Max_in = Sum
                    Max_in_Pk_s = s
                    Max_in_Pk_a = a
        
        # 每個state都選一個action讓值最大
        for s in range(len(states)):
            Max_out = -1
            for a in range(len(actions)):
                val = total_rewards[s, a] + Max_in
                if val >= Max_out:
                    Max_out = val
                    if st_i == s:
                        Max_out_a = a
                        choose_action = Max_out_a
            v[i][s] = (1-alpha) * val + alpha * v[i-1][s]

    try:
        return actions[choose_action]
    except: # 一開始可能不會進到前面的while迴圈，所以直接回傳random的action
        print('!')
        action_i = random.randint(0, len(actions)-1)
        return actions[action_i]

    # ran = random.randint(1, 6)
    # if ran == 1:
    #     action_i = random.randint(0, len(actions)-1)
    #     return actions[action_i]
    # else:
    #     return actions[choose_action]

def find_state_id(all_state, state):
    bins_width = np.round(np.arange(-1.2,1.3,0.6),3)
    bins_height = np.round(np.arange(-1,1.1,2),3)
    for i in range(len(state)):
        if (state[0] in bins_width) == False:
            if -1.2 <= state[0] and state[0] < -0.4:
                state[0] = -1.2
            elif -0.4 <= state[0] and state[0] <= 0.4:
                state[0] = 0
            elif 0.4 < state[0] and state[0] <= 1.2:
                state[0] = 1.2
            else:
                print('width:', state[0])
                state[0] = 0
        if (state[1] in bins_height) == False:
            if -0.5 <= state[1] and state[1] < -0.2:
                state[1] = -0.5
            elif -0.2 <= state[1] and state[1] <= 0.1:
                state[1] = 0
            elif 0.1 < state[1] and state[1] <= 0.5:
                state[1] = 0.5
            else:
                print('height:', state[1])
                state[1] = 0

    for i in range(len(all_state)):
        if (np.array(all_state[i]) == np.array(state)).all():
            return i
    print('st_id error')
    random_st_i = random.randint(0, len(all_state)-1)
    return random_st_i





def UCSG(states, actions, T, delta):
    t = 1
    # Initial state
    vk = np.zeros((len(states), len(actions)), dtype='float16') #vk(s,a)
    total_numbers = np.zeros((len(states), len(actions), len(states)), dtype='float16') # nk(s,a,s')
    total_rewards = np.zeros((len(states), len(actions)), dtype='float16') #maximin
    nk = np.ones((len(states), len(actions)), dtype='float16') #nk(s,a)
    #initial state
    #只能控制一人!!
    env = football_env.create_environment(env_name='1_vs_1_easy', representation='raw', render='True',channel_dimensions=(10,15), number_of_left_players_agent_controls=1 , rewards = "checkpoints,scoring")
    st = env.reset()
    bin_st = simplify(st) #  [[-1, -1], [-0.0, -0.0], [-1.2, -0.0]]
    new_st = bin_st[0]
    st_i = find_state_id(states, new_st)
    # st_i = 0 #!!
    accu_reward = 0
    draw_reward = []
    draw_step = []
    draw_step_all = []
    draw_reward_all = []
    # Initialize phase k
    for k in range(T):
        t_k = t
        #Per-phase visitations
        #在之後的execute policy做係數update
        #compute estimates
        # delta:我們自己設定的容許誤差值    
        delta_1 = delta/(2*len(states)*len(states)*len(actions)*np.log2(T))
        p_hat = total_numbers/np.clip(nk.reshape(len(states), len(actions), 1), 1, None)  #拆解
        r_hat = total_rewards / np.clip(vk, 1, None)
        upper_conf = 1
        lower_conf = 0

        # Update the confidence set
        #confidence_bound_1 = np.clip(np.sqrt((2*n_states*np.log(1/delta)/n_states))+ p_hat, 0, 1)         
        confidence_bound_1 = np.sqrt((2*len(states)*np.log(1/delta)/len(states)))+ p_hat
        confidence_bound_2_2 = np.clip(np.sqrt(np.log(6/delta_1)/(2*len(states)))+p_hat, None, np.sqrt(2*p_hat*(1-p_hat)*np.log(6/delta_1)/len(states)) + 7*np.log(6/delta_1)/(3*(len(states)-1))+p_hat)
        #confidence_bound_2_2 = np.clip(confidence_bound_2_2, 0, 1)
        a = np.sqrt(2*np.log(6/delta_1)/(len(states)-1))
        b = np.sqrt(p_hat*(1-p_hat))
        c1 = (a+b)**2
        c2 = (a-b)**2
        xr = np.clip((1+np.sqrt(1-4*c1))/2, (1-np.sqrt(1-4*c1))/2, 1)
        xl = np.clip((1+np.sqrt(1-4*c2))/2, 0 ,(1-np.sqrt(1-4*c2))/2)
        confidence_bound_2_1 = np.clip(xr, 0 ,xl)
        tmp = confidence_bound_1 + confidence_bound_2_2
        # print('confidencebound1', confidence_bound_1)
        # print('confidencebound2', confidence_bound_2_2)
        confidencebound = [ (tmp[i]) for i in range(0, len(tmp)) if tmp[i] not in tmp[:i] ]
        confidencebound = np.minimum(confidence_bound_1, confidence_bound_2_2)
        #print('confbound', confidencebound)        

        # alpha, gamma
        #Optimistic Planning
        # print('maxminevi')
        ac = maxminevi(states, actions, 0.01, 0.9, 10000, total_rewards, confidencebound, st_i)
        # print('ac', ac)

        #execute policy
        next_st, reward, done, info= env.step(ac)
        # next_st, reward, done, info= env.step([0,1,2,3,4]) # 5人
        bin_next_st = simplify(next_st)
        new_next_st = bin_next_st[0]
        if done:
            print('step', k)
            break
        # next_st_i = random.randint(0, len(states)-1) #!!
        next_st_i = find_state_id(states, new_next_st)
        # yield(t, st, ac, next_st, reward)


        #update
        for a in range(len(actions)):
            if ac == actions[a]:
                ac_i = a

        # 5人
        # for a in range(len(actions)):
        #     flag = 0
        #     for b in range(len(ac)):
        #         if ac[b] != actions[a][b]:
        #             flag = 1
        #             break
        #     if flag == 0:
        #         ac_i = a

        vk[st_i, ac_i] =  vk[st_i, ac_i] + 1 #update vk(s,a)
        total_rewards[st_i,ac_i] += reward # 即時reward
        # total_rewards[st_i,ac_i] += sum(reward) # 5人
        accu_reward += reward
        # print('st_i, ac_i:', st_i, ac_i)
        # print('vk:', vk[st_i][ac_i])
        # print('nk:', nk[st_i][ac_i])
        nk[st_i, ac_i] = max(1,  nk[st_i, ac_i] + 1) #update nk(s,a)
        total_numbers[st_i, ac_i, next_st_i] += 1 #update nk(s, a, s')
        st_i = next_st_i
        # t += 1 
        if(k % 100 == 0):
            print('step:', k)
            draw_step.append(k)
            draw_reward.append(accu_reward)
        draw_step_all.append(k)
        draw_reward_all.append(accu_reward)

    print('accumulate reward:', accu_reward)

    # print("steps", draw_step)
    # plt.plot(draw_step,draw_reward)
    # plt.title("UCSG Reward Convergence Rate") # title
    # plt.xlabel("Steps") # y label
    # plt.ylabel("Reward") # x label
    # plt.show()

    path = 'ucsg_reward.txt'
    f = open(path, 'w')
    lines = []
    for i in draw_reward_all:
        lines.append(str(i) + "\n")
    f.writelines(lines)
    f.close()


    plt.plot(draw_step_all,draw_reward_all)
    plt.title("UCSG Reward Convergence Rate") # title
    plt.xlabel("Steps") # y label
    plt.ylabel("Reward") # x label
    plt.show()



if __name__ == '__main__':
    bins_width = np.round(np.arange(-1.2,1.3,0.6),3)
    bins_height = np.round(np.arange(-1,1.1,2),3)
    # print(bins_width)
    # print(bins_height)

    all_position = []
    for i in range(len(bins_width)):
        for j in range(len(bins_height)):
            all_position.append([bins_width[i],bins_height[j]])
    # print('all_position:', all_position)


    # states = ['s1', 's2', 's3', 's4', 's5']
    # states = np.zeros(10)
    actions = football_action_set.get_action_set({'action_set': 'default'})
    actions_del = [18, 16, 15, 14, 13, 11, 10, 9, 0]
    for idx in actions_del:
        actions.pop(idx)


    T = 1000
    delta = 0.5
    UCSG(all_position, actions, T, delta)

    # 寫reward
    # path = 'ucsg_reward.txt'
    # f = open(path, 'w')
    # lines = []
    # for i in range(10):
    #     lines.append(str(i) + "\n")
    # f.writelines(lines)
    # f.close()

    # 讀reward
    # r = []
    # with open('ucsg_reward.txt', 'r') as f:
    #     for line in f:
    #         r.append(int(float(line.strip('\n'))))
    # # print(r)

    # cut = []
    # x = []
    # for i in range(100000):
    #     x.append(i)
    #     if i % 10000 == 0:
    #         cut.append(i)
    # rew = []
    # for i in range(len(r)):
    #     idx = i // 10000
    #     if idx != 0:
    #         rew.append(r[i] - r[cut[idx]])
    #     else:
    #         rew.append(r[i])

    # plt.plot(x,r)
    # plt.title("UCSG Total Reward") # title
    # plt.xlabel("Steps") # y label
    # plt.ylabel("Reward") # x label
    # plt.show()


