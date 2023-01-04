from tkinter import N
import numpy as np
import gfootball.env as football_env
import random
from gfootball.env import football_action_set
import matplotlib.pyplot as plt


def Random(actions, T):
    env = football_env.create_environment(env_name='1_vs_1_easy', representation='raw', render='True',channel_dimensions=(10,15), number_of_left_players_agent_controls=1 , rewards = "easy,scoring")
    st = env.reset()
    accu_reward = 0
    draw_reward = []
    draw_step = []
    draw_step_all = []
    draw_reward_all = []
    for t in range(T):
        action_i = random.randint(0, len(actions)-1)
        ac = actions[action_i]
        next_st, reward, done, info= env.step(ac)
        if done:
            print('step', t)
            break

        accu_reward += reward

        if(t % 100 == 0):
            print('step:', t)
            draw_step.append(t)
            draw_reward.append(accu_reward)
        draw_step_all.append(t)
        draw_reward_all.append(accu_reward)

    print('accumulate reward:', accu_reward)

    path = 'random_reward.txt'
    f = open(path, 'w')
    lines = []
    for i in draw_reward_all:
        lines.append(str(i) + "\n")
    f.writelines(lines)
    f.close()

    # print("steps", draw_step)
    # plt.plot(draw_step,draw_reward)
    # plt.title("Random Reward Convergence Rate") # title
    # plt.xlabel("Steps") # y label
    # plt.ylabel("Reward") # x label
    # plt.show()

    # plt.plot(draw_step_all,draw_reward_all)
    # plt.title("Random Reward Convergence Rate") # title
    # plt.xlabel("Steps") # y label
    # plt.ylabel("Reward") # x label
    # plt.show()


if __name__ == '__main__':

    actions = football_action_set.get_action_set({'action_set': 'default'})
    actions_del = [18, 16, 15, 14, 13, 11, 10, 9, 0]
    for idx in actions_del:
        actions.pop(idx)

    T = 100000
    Random(actions, T)

    # 讀reward
    # r = []
    # with open('random_reward.txt', 'r') as f:
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

    # plt.plot(x,rew)
    # plt.title("Random Reward Convergence") # title
    # plt.xlabel("Steps") # y label
    # plt.ylabel("Reward") # x label
    # plt.show()