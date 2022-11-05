import gfootball.env as football_env
env = football_env.create_environment(env_name='5_vs_5', representation='raw', render='True', rewards='scoring,easy', channel_dimensions=(10,15), number_of_left_players_agent_controls=5)
state = env.reset()
steps = 0
acc_rew = 0
while True:
  obs, rew, done, info = env.step(env.action_space.sample())
  steps += 1
  acc_rew += rew
  if steps == 1000:
    print('rdi', rew, done, info)
    break
  if done:
    print('step', steps)
    break
# print(obs)
print('acc_rew', acc_rew)
