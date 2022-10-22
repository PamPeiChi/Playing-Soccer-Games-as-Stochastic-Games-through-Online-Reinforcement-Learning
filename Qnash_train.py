import gfootball.env as football_env
env = football_env.create_environment(env_name='1_vs_1', representation='raw', render='True',channel_dimensions=(10,15), number_of_left_players_agent_controls=5 )
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