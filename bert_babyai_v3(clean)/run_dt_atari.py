import csv
import logging
from xmlrpc.client import boolean
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.dt_model_babyai import DT_GPT, GPTConfig
from mingpt.bert_model_babyai import BERT_GPT, token2idx, tokens
from mingpt.trainer_atari import Trainer, TrainerConfig
from collections import deque
import random
import torch
import pickle
import gym
import blosc
import pdb
import copy
import argparse
#from create_dataset import create_dataset
# import babyai.utils as utils
from instruction_process import InstructionsPreprocessor

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--horizon', type=int, default=1)
# 'BabyAI-UnlockPickup-v0'
parser.add_argument('--env', type=str, default='BabyAI-PickupLoc-v0') #'BabyAI-Unlock-v0' #'BabyAI-GoToObj-v0'
#'/home/hchen657/Downloads/BabyAI-Unlock-v0.pkl' PickupLoc
#'/home/hchen657/Downloads/BabyAI-GoToObj-v0.pkl'
parser.add_argument('--bert_dataset_path', type=str, default="./data/BabyAI-PickupLoc-v0_agent1.pkl")#_all_new #_all_new_half #_all_new_fifth
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')

parser.add_argument('--debug_mode', type=int, default=-1, 
                    help="If -1, no debug. "
                    "If 0, will only use 5 traj w/ random init state sample."
                    "If 1, will only use 5 traj w/0 random init state sample."
                )
args = parser.parse_args()

set_seed(args.seed)

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


class BERTDataset(Dataset):

    def __init__(self, block_size, dataset_path, env, rate, plan_horizon, args):        
        self.block_size = block_size
        self.inst_preprocessor = InstructionsPreprocessor()
        with open(dataset_path, 'rb') as f:
            self.trajs = pickle.load(f)

        # store args
        self.args = args

        # debug mode
        if self.args.debug_mode >= 0:
            self.trajs = self.trajs[:5]

        self.insts = []
        self.all_delta_states = []          # Coordinate change. No length limit
        self.all_proc_states = []           # Coordinate sequence. NO End location.
        self.all_proc_images = []
        self.max_inst_len = 0
        self.vocab_size = len(tokens)
        lengths = []
        for traj in self.trajs:
            # change traj states to delta states and pick out the images
            proc_states, proc_images, delta_states= self.get_delta_states(traj[2], traj[1])
            if (len(delta_states) == 0):
                continue
            self.all_delta_states.append(delta_states)
            self.all_proc_images.append(proc_images)
            self.all_proc_states.append(proc_states)

            tmp_inst = self.inst_preprocessor(traj[0])
            self.insts.append(tmp_inst)
            self.max_inst_len = max(self.max_inst_len, len(tmp_inst))
            lengths.append(len(traj[3]))
        self.max_inst_len += 1
        self.env = env
        self.rate = rate
        self.plan_horizon = plan_horizon
        self.full_obs_shape = blosc.unpack_array(self.trajs[0][1])[0].shape[0]
        self.state_dim = 5 #len(self.trajs[0][2][0])


    def __len__(self):
        if self.args.debug:
            return 5
        else:
            return len(self.all_delta_states)
        # return 4

    def get_init_states(self, states):
        return np.copy(states[0])
    
    def get_full_obs(self, full_image):
        return np.copy(full_image[0])

    def get_delta_states(self, states, images):
        images = blosc.unpack_array(images)
        images = images.reshape(len(images), -1)
        delta_states = []
        proc_states = []
        proc_images = []
        for i, (s, next_s) in enumerate(zip(states[:-1], states[1:])):
            if (s[0]==next_s[0] and s[1]==next_s[1]):
                continue
            delta = [next_s[0]-s[0], next_s[1]-s[1]]
            if (delta == [1,0]):
                delta_states.append(token2idx('<-RIGHT->'))
            elif (delta == [-1,0]):
                delta_states.append(token2idx('<-LEFT->'))
            elif (delta == [0,-1]):
                delta_states.append(token2idx('<-UP->'))
            elif (delta == [0,1]):
                delta_states.append(token2idx('<-DOWN->'))
            proc_images.append(images[i])
            proc_states.append(states[i]) #+states[i][-2:] #states[i][:2]+states[i][-2:]
        #delta_states.append(token2idx('<-END->'))
        #proc_images.append(images[-1])
        #proc_states.append(states[-1][:2]+states[-1][-2:]) #+states[i][-2:]
        return proc_states, proc_images, delta_states

    def __getitem__(self, idx):
        block_size = self.block_size // self.rate

        instruction = self.insts[idx]
        instruction = np.concatenate([np.zeros(self.max_inst_len - len(instruction)), instruction])
        instruction = torch.from_numpy(instruction).to(dtype=torch.long)

        if self.args.debug_mode == 1:
            si = 0  # No initial state sampling
        else:
            si = random.randint(0, len(self.all_proc_states[idx])-1)
        states = np.array(self.all_proc_states[idx][si:si + block_size])
        states = states.reshape(len(states), -1)
        actions = np.array(self.all_delta_states[idx][si:si + block_size])
        actions = actions.reshape(len(actions), -1)

        rtgs = np.array([0.]*block_size)
        # rtgs[-1] = 1 - 0.9*(len(self.all_proc_states[idx])/float(self.env.max_steps))
        rtgs[-1] = 1 - 0.9*(len(self.all_proc_states[idx])/float(1024))
        rtgs = discount_cumsum(rtgs, gamma=1.0).reshape(-1,1)

        full_image = np.array(self.all_proc_images[idx][si:si + block_size])
        full_image = full_image.reshape(len(full_image), -1)

        init_state = self.get_init_states(states)
        init_image = self.get_full_obs(full_image)

        tlen = states.shape[0]
        states = np.concatenate([states, np.zeros((block_size - tlen, states.shape[1]))], axis=0)
        full_image = np.concatenate([full_image, np.zeros((block_size - tlen, full_image.shape[1]))], axis=0)
        actions = np.concatenate([actions, token2idx('<-PAD->') * np.ones((block_size - tlen, 1))], axis=0)

        msk = np.random.uniform(0, 1, (tlen-1,1)) > 0.5
        state_msk = np.concatenate([np.zeros((1, 1)), msk], axis=0).astype(boolean)
        action_msk = np.ones((tlen,1)).astype(boolean)
        state_msk = np.concatenate([state_msk, np.zeros((block_size - tlen, state_msk.shape[1]))], axis=0)
        action_msk = np.concatenate([action_msk, np.zeros((block_size - tlen, 1))], axis=0)

        states = torch.from_numpy(states).to(dtype=torch.long)
        actions = torch.from_numpy(actions).to(dtype=torch.long)
        rtgs = torch.from_numpy(rtgs).to(dtype=torch.float32)
        timesteps = torch.tensor([si], dtype=torch.int64).unsqueeze(1)
        state_msk = torch.tensor(state_msk, dtype=torch.bool)
        action_msk = torch.tensor(action_msk, dtype=torch.bool)

        init_state = torch.from_numpy(init_state).to(dtype=torch.long)
        init_image = torch.from_numpy(init_image).to(dtype=torch.float32)
        #pdb.set_trace()
        return states, actions, full_image, state_msk, action_msk, rtgs, timesteps, instruction, init_state, init_image


# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# env = gym.make(args.env)
env = None

rate = 3 if args.model_type == 'reward_conditioned' else 2
max_timesteps = 1024
sample_iteration = 300 # * args.horizon
plan_horizon = args.horizon

bert_train_dataset = BERTDataset(args.context_length*rate, args.bert_dataset_path, env, rate, plan_horizon, args=args)
# print(bert_train_dataset.trajs[0])
# print(len(bert_train_dataset[0]))
#for i in range(len(bert_train_dataset)):
#    print(bert_train_dataset[i])

mconf = GPTConfig(bert_train_dataset.vocab_size, bert_train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max_timesteps, env_size=bert_train_dataset.full_obs_shape, state_dim=bert_train_dataset.state_dim)
bert_model = BERT_GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(bert_train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max_timesteps)
trainer = Trainer(bert_model, bert_train_dataset, None, tconf, env, rate, plan_horizon, sample_iteration, bert_train_dataset.inst_preprocessor, bert_train_dataset.full_obs_shape)

trainer.train()
