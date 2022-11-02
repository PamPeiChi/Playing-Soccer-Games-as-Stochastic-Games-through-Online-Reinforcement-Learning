import gfootball.env as football_env
import numpy as np
import random
import math
import nashpy as nash



nactions = 8
discount = 1
alpha = 0.02
epsilon =  0.01

def flat_states(obs):
    """
    ball: shape(1,3)
    ball_owned_team: (int)
    ball_owned_player: (int) 
    left_team: shape(nagents,2)
    left_team_role: shape(1,nagents) x
    score: shape(1,2) x
    steps_left: (int) x

    """
    minor_states = ['ball','ball_owned_team','ball_owned_player',
                        'left_team','left_team_roles','score','steps_left']

    
    states = {state: obs[0][state] if obs[0][state] is not None else None for state in minor_states}
 

    print(states)
    new_states = []
    for state in states.values():
        if type(state) == np.ndarray:
          state = list(state.flatten())[0]
          new_states.append(state)
        elif type(state) == list:
          for s in range(len(state)):
            new_states.append(state[s])
        else:
          new_states.append(state)
    print(new_states)


def bin_state(obs):
    observed_states = ["ball_owned_team","ball","left_team"]
    print("ball owned")
    print(obs[0]["ball_owned_team"])
    print("ball", obs[0]["ball"])
    print("left team",obs[0]["left_team"])
    states = {state: obs[0][state] if obs[0][state] is not None else None for state in observed_states}
    print("states", states)
    bins_width = np.round(np.arange(-1,1,0.2),3)
    bins_height = np.round(np.arange(-0.5,0.5,0.1),3)
    bin_states = {}
    for state, value in states.items():
        if state == 'left_team':
            value = value[0]
        if type(value) == int:
            bin_states[state] = [value,-1]
            print("int",state)
        else:
            print("not int",state)
            bin_states[state] = []
            for bin_w in range(1,len(bins_width)):
                if bins_width[0] > value[0]:
                    print("width out of bound",value[0])
                    bin_state_w = bins_width[0]
                    break
                if value[0] < bins_width[bin_w] and value[0] >= bins_width[bin_w-1]:
                    bin_state_w = bins_width[bin_w-1]
                    print("w: bin_state",bin_state_w)
                    break
                
            for bin_h in range(1,len(bins_height)):
                if bins_height[0] > value[1]:
                    print("height out of bound",value[1])
                    bin_state_h = bins_height[1]
                    bin_states[state] = [bin_state_w, bin_state_h]
                    break
                if value[1] < bins_height[bin_h] and value[1] >= bins_height[bin_h-1]:
                    bin_state_h = bins_height[bin_h-1]
                    print("h: bin_state",bin_state_h)
                    break
            bin_states[state] = [bin_state_w, bin_state_h]
                    

                
    bin_current_state = [bin_states[s] for s in bin_states]
    print("bin current state", bin_current_state)
    return bin_current_state

def create_q_tables():
    bin_width = 0.2
    bin_height = 0.1
    ball_own = 3 # -1,0,1
    width = 2 # 10 states [-1,1]
    height = 1 # 5 states [-.42, 0.42]
    W_bin = np.round(np.arange(-1.2,1.2,0.2), 3)
    print(W_bin)
    H_bin = np.round(np.arange(-0.5,0.6,0.1),3)
    states_table = {}
    states_table["ball_owned"] = np.array([[-1,-1],[0,-1],[1,-1]])
    states_table["ball"] = []
    states_table["left_team"] = []
    states_table["action1"] = []
    states_table["action2"] = []
    for w in W_bin:
        for h in H_bin:
            print("w,h",(w,h))
            states_table["ball"].append([w,h])
            states_table["left_team"].append([w,h])
    states_table["ball"] = np.array(states_table["ball"])
    states_table["left_team"] = np.array(states_table["left_team"])
    states_table["action1"] = np.arange(0,8,1)
    states_table["action2"] = np.arange(0,8,1)
    print(list(states_table.values()))
    print(states_table["ball_owned"].shape)
    print(states_table["ball"].shape)
    print(states_table["left_team"].shape)
    qtables1 = np.zeros((3,
    W_bin.shape[0]*H_bin.shape[0],
    W_bin.shape[0]*H_bin.shape[0],
    nactions, nactions))
    
    qtables2 = np.zeros((3,
    W_bin.shape[0]*H_bin.shape[0],
    W_bin.shape[0]*H_bin.shape[0],
    nactions, nactions))
    
    
    print(qtables1.shape)
    return states_table, qtables1, qtables2

def find_states(states_table, states):
    """
    find state:
    "ball_owned": [[-1,-1],[0, -1], [1,-1]]
    "ball": [[-1, -0.8, -0.6,...., 0.8,1] ,[-0.5, -0.4, ... , 0.4, 0.5]] # 11 x 11
    "left_team":    
    bins_width = np.round(np.arange(-1,1,0.2),3)
    bins_height = np.round(np.arange(-0.5,0.5,0.1),3)
    """

    find = []
    for i, key in enumerate(list(states_table.keys())[:3]):
        for j, s in enumerate(states_table[key]):
            #print("s",s)
            #print("states i ",states[i])
            if tuple(states[i]) == tuple(s):
                find.append([i,j])
                print("f:",find)
                continue
    print("find",find)
    return find

def GetPi(qtables1, qtables2, find):
    print("get pi find", find)
    Pi = []
    Pi_O = []
    for i in range(nactions):
        row_q = []
        row_opponent = []
        for j in range(nactions):
            row_q.append(qtables1[find[0], find[1], find[2], i,j])
            row_opponent.append(qtables2[find[0], find[1], find[2], i,j])

        Pi.append(row_q)
        Pi_O.append(row_opponent)
        
    nash_game = nash.Game(Pi, Pi_O)
    equilibria = nash_game.lemke_howson_enumeration()
    pi_nash = None
    try:
        pi_nash_list = list(equilibria)
    except:
        pi_nash_list = []
    for index, eq in enumerate(pi_nash_list):
        if eq[0].shape == (nactions, ) and eq[1].shape == (nactions, ):
            if any(np.isnan(eq[0])) == False and any(np.isnan(eq[1])) == False:
                if index != 0:
                    pi_nash = (eq[0], eq[1])
                    break
    if pi_nash is None:
        print("pi_nash is null, bug in nashpy")
        pi_nash = (np.ones(nactions)/nactions, np.ones(nactions)/nactions)
    return pi_nash[0], pi_nash[1]

def computeNashQ(qtable1, qtable2, agent, find):
    Pi, Pi_O = GetPi(qtable1, qtable2, find)
    nashq = 0
    print("compute q nash", find)
    for action1 in range(nactions):
        for action2 in range(nactions):
            if agent == 1:
                nashq += Pi[action1] * Pi_O[action2] * qtable1[find[0], find[1], find[2], action1, action2]
            elif agent == 2:
                nashq += Pi[action1] * Pi_O[action2] * qtable2[find[0], find[1], find[2], action1, action2]
            else:
                print("error agent")
    return nashq

def computeQ(agent, qtable1, qtable2, find, rewards, action1, action2):
    nashq = computeNashQ(qtable1, qtable2,agent,find)
    print("compute q find", find)
    print(qtable1.shape)
    if agent == 1:
         m_value = alpha * (rewards + discount * nashq)
         o_value = (1-alpha) * qtable1[find[0], find[1], find[2],action1, action2]
         qtable1[find[0], find[1], find[2], action1, action2] = o_value + m_value
         return qtable1
    else:
        m_value = alpha * ((-rewards) + discount * nashq)
        o_value = (1-alpha) * qtable2[find[0], find[1], find[2],action1, action2]
        qtable2[find[0], find[1], find[2], action1, action2] = o_value + m_value
        return qtable2

def choose_action(qtable1, qtable2, states_table, states, epsilon = 0.01, agent = 1):
    print(states)
    find = find_states(states_table, states)
    Pi, Pi_O = GetPi(qtable1, qtable2, find)
    print("pi shape ",Pi.shape)
    
    if random.random() <= epsilon:
        print("random_action")
        action_idx = random.randint(0,nactions-1)
        print(action_idx)
    else:
        print("best action")
        print("Pi",Pi)
        action_idx = random.choice(np.flatnonzero(Pi == Pi.max()))
        print(action_idx)
    return action_idx


#%%
print("start")
env = football_env.create_environment(env_name='1_vs_1_easy', representation='raw', render='True',channel_dimensions=(10,15), number_of_left_players_agent_controls=1 )
obs = env.reset() # states_of_agent_i = obs[agent_i] (type dic)
#states = flat_states(obs)
states_table, qtable1, qtable2 = create_q_tables()
bin_states = bin_state(obs)
steps = 0
while steps <= 1000:
    find = find_states(states_table,bin_states)
    action1 = choose_action(qtable1, qtable2, states_table, bin_states, epsilon, 1)
    action2 = choose_action(qtable2, qtable2, states_table, bin_states, epsilon, 2)
    print("action1: {}; action2: {}", action1, action2)
    obs, reward, info, done = env.step(action1)
    print("reward",reward)
    #states = flat_states(obs)
    bin_states = bin_state(obs)
    find = find_states(states_table, bin_states)
    # update
    qtable1 = computeQ(1, qtable1, qtable2, find, reward,action1,action2) 
    qtable2 = computeQ(2, qtable2, qtable2, find, reward,action1,action2)
    steps += 1



