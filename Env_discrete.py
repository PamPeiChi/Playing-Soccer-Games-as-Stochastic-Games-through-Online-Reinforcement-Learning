import gfootball.env as football_env
import pandas as pd
import numpy as np
# env = football_env.create_environment(env_name='5_vs_5', representation='raw', render='True',channel_dimensions=(10,15), number_of_left_players_agent_controls=5 )
# state = env.reset()
# obs, rew, done, info = env.step(env.action_space.sample())
def simplify(obs, chop):
    '''
    input: obs - from 'obs, rew, done, info = env.step(env.action_space.sample())' & chop - 切的多細 e.g. 0.01(粗)->0.0001(細)
    output: dictionary {
        ball: [x,y]
        left_team: [[x1,y1], [x2,y2], ...]
    }
    '''
    bin = np.arange(-1.5, 1.5, chop)
    state = {}
    ball = pd.cut(obs[0]["ball"][:-1],bin)
    state['ball'] = [ball[0].left, ball[1].left]
    print(state['ball'])
    x = []
    for player in obs[0]["left_team"]:
        # print("origin", player)
        player = pd.cut(player,bin)
        x.append([player[0].left, player[1].left])
        # print("after simplify", player[0].left, player[1].left)
    state['left_team'] = x
    return state

# ans = simplify(obs, 0.0005)
# print(ans)
