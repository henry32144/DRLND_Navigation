import torch
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agent import Agent
from collections import deque
from unityagents import UnityEnvironment

seed = 0
saved_model_name = "best_model.pth"
env = UnityEnvironment(file_name="Banana.exe", seed=seed)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

def dqn(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Train the agent.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    mean_score = 0
    solved_episode = -1

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action.astype(int))[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0 and solved_episode == -1:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            solved_episode = i_episode
        if np.mean(scores_window) > mean_score and np.mean(scores_window)>=13.0:
            print('\nSave best model \tAverage Score: {:.2f}'.format(np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), saved_model_name)
        mean_score = max(mean_score, np.mean(scores_window))
        
    return scores, solved_episode

def main(args):
    buffer_size = args.buffer_size  # replay buffer size
    batch_size = args.batch_size         # minibatch size
    gamma = args.gamma            # discount factor
    alpha = args.alpha             # how much prioritization is used
    beta = args.beta              # initial value of beta for prioritized replay buffer
    beta_increment_per_step = args.beta_increment # number of iterations over which beta will be annealed from initial value
    epsilon = args.epsilon          # epsilon to add to the TD errors when updating priorities.
    tau = args.tau              # for soft update of target parameters
    lr = args.lr               # learning rate
    update_very = args.update_every        # how often to update the network


    agent = Agent(state_size, action_size, 0, batch_size, \
                    buffer_size, gamma, alpha, beta, beta_increment_per_step, \
                    epsilon, tau, lr, update_very)
    # start trainging
    scores, solved_episode = dqn(agent, env)
    print('\nTraining finish!')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axhline(y=13.0, color='r', linestyle='-')
    plt.plot(np.arange(len(scores)), pd.Series(scores).rolling(100).mean())
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores.png')
    print('\nRecord saved.')
    plt.close(fig)

    env.close()

if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser(description='Agent parameters')
    parser.add_argument('--buffer_size', '-bu', type=int, default=int(1e5),
                        help='replay buffer size')
    parser.add_argument('--batch_size', '-bs', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--gamma', '-g', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--alpha', '-a', type=float, default=0.6,
                        help='how much prioritization is used')
    parser.add_argument('--beta', '-b', type=float, default=0.4,
                        help='initial value of beta for prioritized replay buffer')
    parser.add_argument('--beta_increment', '-bi', type=float, default=0.001,
                        help='number of iterations over which beta will be annealed from initial value')
    parser.add_argument('--epsilon', '-e', type=float, default=0.01,
                        help='discount factor')
    parser.add_argument('--tau', '-t', type=float, default=1e-3,
                        help='how much prioritization is used')
    parser.add_argument('--lr', '-l', type=float, default=5e-4,
                        help='discount factor')
    parser.add_argument('--update_every', '-u', type=int, default=4,
                        help='how often to update the network')
    args = parser.parse_args()
    main(args)