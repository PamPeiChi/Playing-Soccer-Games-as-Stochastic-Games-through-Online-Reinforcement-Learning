import gfootball.env as football_env
import numpy as np
import gym
from gfootball.env import football_action_set
"""

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
"""

from gfootball.env import observation_preprocessing
import tensorflow as tf
import sonnet as snt


# observation[0]['frame'].shape = (720,1280,3)

def network_fn(frame):
  # Convert to floats.
  print(frame.shape) # pixles: shape(72,96,3), frame: shape(720,960,3,1)
  frame = tf.cast(frame, dtype= tf.float16)
  frame /= 255
  frame = tf.expand_dims(frame, axis = -1) # [720, 1280, 3, 1]
  print(frame.shape.as_list())
  with tf.variable_scope('convnet'):
    conv_out = frame
    conv_layers = [(16,2), (16, 2), (32, 2), (32, 2)]
    for i, (num_ch, num_blocks) in enumerate(conv_layers):
      # Downscale.
      print("num_channels", num_ch)
      print("num_blocks",num_blocks)
      conv_out = snt.Conv2D(num_ch, 4, stride=1, padding='SAME')(conv_out) # 4th dim = num_ch
      print(conv_out.shape.as_list()) # [720, 1280, 3, 16], [720, 640, 2, 16], [720, 320, 1, 32]
      conv_out = tf.nn.pool(
          conv_out,
          window_shape=[4, 4],
          pooling_type='MAX',
          padding='SAME',
          strides=[3, 3]) # 2nd_dim =
      print("for loop: ") 
      print(conv_out.shape.as_list()) # [720, 640, 2, 16], [720,320, 1, 16], [720, 160, 1, 32]
      # Residual block(s).
      for j in range(num_blocks):
        with tf.variable_scope('residual_%d_%d' % (i, j)):
          block_input = conv_out
          conv_out = tf.nn.relu(conv_out) 
          print("relu1:")
          print(conv_out.shape.as_list()) # 
          conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
          print("Conv2D1:")
          print(conv_out.shape.as_list()) # 
          conv_out = tf.nn.relu(conv_out)
          print("relu2:")
          print(conv_out.shape.as_list()) # 
          conv_out = snt.Conv2D(num_ch, 3, stride=1, padding='SAME')(conv_out)
          print("Conv2D1:")
          print(conv_out.shape.as_list()) # [720, 640, 2, 16], [720, 160, 1, 32], [720, 80, 1, 32], 
          conv_out += block_input

  conv_out = tf.nn.relu(conv_out)
  conv_out = snt.BatchFlatten()(conv_out)
  print("batchFlatten:")
  print(conv_out.shape.as_list()) # [720, 2560]
  conv_out = snt.Linear(256)(conv_out) # define 2nd dim size
  print("Linear:")
  print(conv_out.shape.as_list())
  conv_out = tf.nn.relu(conv_out) # [720, 256]

  return conv_out


class ObservationStacker(object):
    def __init__(self, stacking):
      self._stacking = stacking
      self._data = []

    def get(self, observation):
      if self._data:
        self._data.append(observation)
        self._data = self._data[-self._stacking:]
      else:
        self._data = [observation] * self._stacking
      return np.concatenate(self._data, axis=-1)

    def reset(self):
      self._data = []
class DummyEnv(object):
      # We need env object to pass to build_policy, however real environment
  # is not there yet.

  def __init__(self, action_set, stacking):
    self.action_space = gym.spaces.Discrete(
        len(action_set))
    print(self.action_space)
    # pixel size: 
    #self.observation_space = gym.spaces.Box(
    #    0, 255, shape=[72, 96, 4 * stacking], dtype=np.uint8)

    # observation["frame"]  size:
    self.observation_space = gym.spaces.Box(
      0, 255, shape=[720, 1280, 3], dtype=np.uint8
    )
    print(self.observation_space)

class policy_network(object):

    def __init__(self, build_policy):
        #Discrete(19)
        #Box(720, 1280, 3)
      
        self.policy = np.asarray([list(build_policy.observation_space.shape), list(build_policy.action_space.shape)])
        print("policy", self.policy)
    def step(self, stateid):
        actions = self.policy[stateid] # 這裡不知道怎麼寫

# 1. 本身維度就小的, 但具體如何downgrade部清楚的input: pixels: shape(72,96)
    
# 2. 仿造ppo寫法使用raw裡面的frame (1280,720,3)    


stacking = 1
env = football_env.create_environment(env_name="5_vs_5", 
representation='raw', render = True)
state = env.reset()
# convert frame to nn
nn_network = network_fn(state[0]['frame'])
action_set = football_action_set.get_action_set({'action_set': 'default'})
print("action_set",action_set)
# create a policy network, state = observation["frame"] ?? maybe should be nn_network
build_policy = DummyEnv(action_set, stacking)
#policy = np.array(build_policy.observation_space, build_policy.action_space)
#print("policy_shape",policy.shape)
policy = policy_network(build_policy)
# (start) 初始化policy的值...??

# (end) 初始化policy的值...??


steps = 0

# store observation into a stack
stacker = ObservationStacker(stacking)
while True:
  # take an action, observe env and receive reward
  next_state, rew, done, info = env.step(env.action_space.sample())
  print(next_state[0]['frame'])
  # convert raw observation into frame
  # frame = observation_preprocessing.generate_smm(next_state)
  new_nn_network = network_fn(next_state[0]['frame'])
  # store new frame
  frame_stack = stacker.get(new_nn_network)
  # print("frame_stack", frame_stack.shape) # frame_stack (1, 72, 96, 4) => pixel


  steps += 1
  if steps % 100 == 0:
    # print(obs, rew, done, info)
    break
  if done:
    break





# Initialization: t= 1


#建立dummy env 為pixel observation shape

# for phase k = 1,2,..., do

    # Initialize Phase

    # Update confidence set

    # Optimistic Planning

    # Execute Policies

