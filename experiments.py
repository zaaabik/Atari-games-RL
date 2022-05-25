
# coding: utf-8

# In[10]:


import gym
from gym import spaces
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datetime import datetime
import cv2


from plotly import express as px
from plotly import graph_objects as go
from copy import deepcopy
import os

from torch import nn
import torch
import random 

from tqdm import tqdm
from torchvision import transforms as T
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device')
parser.add_argument('--mode')
parser.add_argument('--game_name')

args = parser.parse_args()
args = vars(args)
print(args)
DEVICE = str(args['device'])
MODE = str(args['mode'])
GAME_NAME = str(args['game_name'])

# DEVICE = 'cuda:1'
# GAME_NAME = 'BeamRiderNoFrameskip-v4'
# MODE = 'eps_greedy'
# # MODE = 'eps_greedy'
# # MODE = 'eps_greedy_decay'
# # MODE = 'only_explore'
# # MODE = 'only_strategy'



# ### Init enviroment

# In[15]:


env = gym.make(GAME_NAME)
env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4,
                                      screen_size=84, terminal_on_life_loss=False, 
                                      grayscale_obs=True, grayscale_newaxis=False, 
                                      scale_obs=False)
env = gym.wrappers.FrameStack(env, 4)


# In[7]:


N_ACTIONS = int(env.action_space.n)


# In[ ]:


if GAME_NAME == 'PongNoFrameskip-v4':
    N_ACTIONS = 4


# ### DQN

# In[8]:


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.MEAN = 0.416376
        self.STD = 0.1852287
        
        self.act = torch.nn.ReLU()
        
        self.head = torch.nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            self.act, 
            nn.Linear(512, 512),
            self.act, 
            nn.Linear(512, N_ACTIONS)
        )

    def forward(self, x):
        x = (x - self.MEAN) / self.STD
        bs = x.shape[0]        
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))        
        x = x.view(bs, -1)        
        return self.head(x)


# In[9]:


from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# In[2]:


# mode = 'eps_greedy'
# mode = 'eps_greedy_decay'
# mode = 'only_explore'
# mode = 'only_strategy'


# In[4]:


def get_eps_params_from_mode(mode):
    # eps_threshold, eps_decay, eps_min
    if 'eps_greedy' == mode:
        return 0.01, 1., 0.
    elif 'eps_greedy_decay' == mode:
        return 1., 0.999985, 0.02
    elif 'only_explore' == mode:
        return 1., 1., 1.
    elif 'only_strategy' == mode:
        return 0., 1., 0.
    else:
        return 


# In[10]:


class Agent:
    def __init__(self, device):
        self.device = device

        self.model = DQN().to(device)
        self.init_model = deepcopy(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.target_model = DQN().to(device)
        
        self.memory = ReplayMemory(50_000)                        
        self.n_actions = N_ACTIONS
        self.batch_size = 32
        self.step = 0
        self.episode = 0
        self.GAMMA = 0.99

        self.eps_threshold, self.eps_decay, self.eps_min = get_eps_params_from_mode(MODE)
        self.base_path = f'logs/when_should_agents_explore/{GAME_NAME}/{MODE}/' +str(datetime.now())
        self.log_path = os.path.join(self.base_path, 'tensorboard')
        os.makedirs(self.log_path, exist_ok=False)
        self.writer = SummaryWriter(self.log_path)
        
    def log_episode(self, reward, sum_loss, mean_loss, episode_len):
        self.writer.add_scalar('Reward', reward, self.episode)
        self.writer.add_scalar('Loss_mean', mean_loss, self.episode)
        self.writer.add_scalar('Loss_sum', sum_loss, self.episode)
        self.writer.add_scalar('Episode_len', episode_len, self.episode)
        self.episode += 1
    
    def log_reward(self, reward):
        self.writer.add_scalar('Reward', reward, self.step)

    def select_action(self, state): 
        self.eps_threshold *= self.eps_decay        
        self.eps_threshold = max(self.eps_min, self.eps_threshold)
        self.writer.add_scalar('THR', self.eps_threshold, self.step)
        self.step += 1
        
        state = torch.tensor([state]).to(self.device)        
        if np.random.rand() < self.eps_threshold:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                best_action = self.model(
                    state
                )
                best_action = best_action.max(1)[1]
            return best_action.item()
        
    

    def push_memory(self, state, action, next_state, reward, done):
        self.memory.push(
            torch.tensor([state]), 
            torch.tensor([[action]]), 
            torch.tensor([next_state]), 
            torch.tensor([reward]).float(),
            torch.tensor([done]).float(),
            
        )
    
    def update_target_model(self):
        self.target_model.load_state_dict(deepcopy(self.model.state_dict()))

    def optimize_model(self,):
        
        if len(self.memory) < self.batch_size:
            return 0.
        
        transitions = self.memory.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))

        is_done = torch.cat(batch.done).to(self.device)
                
        state_batch = torch.cat(batch.state).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        
        state_action_values = self.model(state_batch).gather(1, action_batch).squeeze(-1)
        
        next_state_values = self.target_model(next_state_batch).max(1)[0] * (1 - is_done.float())
        
        next_state_values = next_state_values.detach()        
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch


        criterion = nn.MSELoss()
        criterion_v2 = nn.MSELoss(reduce=None, reduction='none')
        loss = criterion(state_action_values, expected_state_action_values)
        self.writer.add_scalar('Loss', loss.item(), self.step)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# In[11]:


agent = Agent(device=DEVICE)


# In[ ]:


update_period = 1000
num_episodes = 9999
warmup = 750
episode_stats = {
    'reward' : [],
    'episode_len' : [],
    'loss' : [],
    'mean_loss' : []
}
episodes = []
for i_episode in range(0, num_episodes):        
    
    episode_stats['loss'].append(0)
    episode_stats['reward'].append(0)
    
    state = env.reset()    
    episodes.append({'states' : [], 'actions' : [], 'probs' : [], 'reward' : []})
    for step in tqdm(range(1, 5000)):
        action = agent.select_action(state)        
        episodes[-1]['states'].append(state)
        episodes[-1]['actions'].append(action)
        probs = agent.model(torch.tensor(state)[None].to(DEVICE))[0].detach().cpu().numpy()
        episodes[-1]['probs'].append(probs)
        
        next_state, reward, done, _ = env.step(action)           
        
        episodes[-1]['reward'].append(reward)
        
        episode_stats['reward'][-1] += reward
        
        agent.log_reward(episode_stats['reward'][-1])

        agent.push_memory(
            state, action, next_state, reward, done
        )
        state = next_state
        
        
        if done:
            episode_stats['episode_len'].append(step)
            episode_stats['mean_loss'].append(
                episode_stats['loss'][-1] / step
            )
            break  
            
        if agent.step > warmup:            
            loss = agent.optimize_model()        
            episode_stats['loss'][-1] += loss        

            if agent.step % update_period == 0:                
                agent.update_target_model()    
    
    agent.log_episode(
        episode_stats['reward'][-1],
        episode_stats['loss'][-1],
        episode_stats['mean_loss'][-1],
        episode_stats['episode_len'][-1]
    )
    
    actions = episodes[-1]['actions']
    print(pd.Series(actions).value_counts() / len(actions))
    print(episode_stats['reward'][-1])
    
    
    if i_episode % 10 == 0:
        save_path = os.path.join(agent.base_path, 'weights')
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            agent.model.state_dict(),
            os.path.join(save_path, f'model_{i_episode}.pkl')
        )
        torch.save(
            np.array(episode_stats['reward']),
            os.path.join(save_path, 'total_rewards.pkl')
        )
        
        

