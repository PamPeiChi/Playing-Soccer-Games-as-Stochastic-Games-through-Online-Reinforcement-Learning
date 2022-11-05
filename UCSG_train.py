import gfootball.env as football_env
env = football_env.create_environment(env_name='5_vs_5', representation='raw', render='True',channel_dimensions=(10,15), number_of_left_players_agent_controls=5 )
state = env.reset()
steps = 0
while True:
  obs, rew, done, info = env.step(env.action_space.sample())
  steps += 1
  if steps % 100 == 0:
    # print(obs, rew, done, info)
    break
  if done:
    break

# print("Steps: %d Reward: %.2f" % (steps, rew))

# before 11/5
import numpy as np

R = {}
Pk = {}
S = []
st = 1
V = {}
A1 = []
A2 = []
nstates = 100
M = {}
Lambda = 0.01
alpha = 0.01
nsteps = 100

def value_iteration(st, Pk, V, t):
  for a1 in A1:
    for a2 in A2:
      V_t = []
      for st_1 in S:
        V_t = Pk[st,a1,a2] + V[t-1,st_1]
      v = max(V_t)
      V[t,st] = v + R[st,a1,a2]
  return V

def maxmin_EVI(M, Lambda, alpha):
  V[0] = np.zeros(nstates)
  t = 0
  while True:
    for st in S:
      val = maxmin_EVI(st,Pk,V,t)
      V[t,st] = (1-alpha) * val[t,st] + alpha * V[t-1, st]
    SP = max(V[t]) - min(V[t])
    if SP <= ((1-alpha) * Lambda):
      break
    t = t + 1




