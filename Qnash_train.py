import gfootball.env as football_env
import numpy as np
import random
import nashpy as nash

env = football_env.create_environment(env_name='1_vs_1_easy', representation='raw', render='True',channel_dimensions=(10,15), number_of_left_players_agent_controls=1 )
obs = env.reset() # states_of_agent_i = obs[agent_i] (type dic)
nactions = 8
discount = 1
alpha = 0.02
epsilon: 0.01

def flat_states(obs):
    """
    ball: shape(1,3)
    ball_owned_team: (int)
    ball_owned_player: (int)
    left_team: shape(nagents,2)
    left_team_role: shape(1,nagents)
    score: shape(1,2)
    steps_left: (int)

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

def create_q_tables(states):
    nstates = len(states)
    qtables1 = np.zeros((nstates, (nactions, nactions)))
    qtables2 = np.zeros((nstates, (nactions, nactions)))
    return qtables1, qtables2

def GetPi(qtables1, qtables2, state):
    Pi = []
    Pi_O = []
    # iterate over rows
    for i in range(nactions):
        row_q = []
        row_opponent = []
        for j in range(nactions):
            row_q.append(qtables1[state][(i,j)])
            row_opponent.append(qtables2[state[(i,j)]])
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
        if eq[0].shape == (nactions,) and eq[1].shape == (nactions, ):
            if any(np.isnan(eq[0])) == False and any(np.isnan(eq[1])) == False:
                if index != 0:
                    pi_nash = (eq[0], eq[1])
                    break
    if pi_nash is None:
        print("pi_nash is null, bug in nashpy")
        pi_nash = (np.ones(nactions)/nactions, np.ones(nactions)/nactions)
    return pi_nash[0], pi_nash[1]

def computeNashQ(qtables, agent, state):
    Pi, Pi_O = GetPi()
    nashq = 0
    for action1 in range(nactions):
        for action2 in range(nactions):
            if agent == 1:
                nashq += Pi[state][action1] * Pi_O[state][action2] * qtables[state][(action1, action2)]
            elif agent == 2:
                nashq += Pi[state][action1] * Pi_O[state][action2] * qtables[state][(action1, action2)]
            else:
                print("error agent")
    return nashq

def computeQ(qtables1, state, rewards, action1, action2):
    for agent in range(nagents):
        nashq = computeNashQ(qtables1,state)
        o_value = (1-alpha) * qtables1[state,(action1, action2)]
        m_value = alpha * (rewards[agent] + discount * nashq)
        qtables1[state, (action1, action2)] = o_value + m_value
    return qtables1

def choose_action(qtables1, state, epsilon = 0.01, agent = 1):
    Pi, Pi_O = GetPi(qtables1, agent, state)
    if random.random() <= epsilon:
        action_idx = random.choice(0,nactions-1)
    else:
        action_idx = random.choice(np.flatnonzero(Pi == Pi.max()))
    return action_idx

state = env.reset()

steps = 0
while True:
  obs, rew, done, info = env.step(env.action_space.sample())
  steps += 1
  if steps % 100 == 0:
    print(obs, rew, done, info)
  if done:
    break

print("Steps: %d Reward: %.2f" % (steps, rew))

