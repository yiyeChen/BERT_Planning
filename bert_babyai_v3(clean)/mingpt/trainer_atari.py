"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import gym
import logging
from tqdm import tqdm
import numpy as np
import pdb
import torch
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
# import babyai.utils as utils

logger = logging.getLogger(__name__)

from mingpt.utils import bert_sample_multi_step, dt_sample, AGENT_ID, AGENT_COLOR
# import atari_py
#from collections import deque
#import random
#import cv2
#import torch
#from PIL import Image
import logging
# from babyai.utils.agent import BotAgent

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, bert_model, bert_train_dataset, test_dataset, config, env, rate, plan_horizon, sample_iteration, inst_preprocessor, env_size):
        self.bert_model = bert_model
        self.bert_train_dataset = bert_train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.env = env
        self.rate = rate
        self.plan_horizon = plan_horizon
        self.sample_iteration = sample_iteration
        self.inst_preprocessor = inst_preprocessor
        self.env_size = env_size
        # self.bot_advisor_agent = BotAgent(self.env)
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.bert_model = self.bert_model.to(self.device)
            #self.dt_model = torch.nn.DataParallel(self.dt_model).to(self.device)
            #self.bert_model = torch.nn.DataParallel(self.bert_model).to(self.device)
        console = logging.StreamHandler(sys.stdout)
        console_log_level = 100
        console.setLevel(console_log_level)
        self.logger = logging.getLogger(__name__)  
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f'log_file_{self.plan_horizon}.log')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.dt_model.module if hasattr(self.dt_model, "module") else self.dt_model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        bert_model, config = self.bert_model, self.config

        raw_bert_model = bert_model.module if hasattr(self.bert_model, "module") else bert_model
        bert_optimizer = raw_bert_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            bert_model.train(is_train)
            bert_data = self.bert_train_dataset if is_train else self.test_dataset
            bert_loader = DataLoader(bert_data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=0) #config.num_workers
            dt_losses = []
            bert_losses = []
            rates = []
            gt_traj_energys = []
            mcmc_better_than_first_rates = []
            mcmc__better_than_all_rates = []
            mcmc_energys = []
            free_rates = []
            action_correct_rates = []
            all_action_correct_rate_steps = []

            pbar = tqdm(enumerate(bert_loader), total=len(bert_loader)) if is_train else enumerate(bert_loader)
            for it, (x, y, full_imgs, msk_x, msk_y, r, t, inst, init_x, init_image) in pbar:

                x = x.to(self.device)
                y = y.to(self.device)
                full_imgs = full_imgs.to(self.device)
                msk_x = msk_x.to(self.device)
                msk_y = msk_y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)
                inst = inst.to(self.device)

                #init_x = init_x.to(self.device)
                #init_image = init_image.to(self.device)
                with torch.autograd.set_detect_anomaly(True):
                    with torch.set_grad_enabled(is_train):
                        is_debug = np.random.uniform(0, 1) > 0.95
                        bert_loss, rate, target_rate, gt_traj_energy, is_better_than_first, is_better_than_all, mcmc_energy, free_rate, action_correct_rate, action_correct_rate_steps \
                            = bert_model.train_step(x, y, full_imgs, state_masks=msk_x, action_masks=msk_y,\
                                timesteps=t, insts=inst, init_states=init_x, init_obss=init_image, is_debug=is_debug, logger = self.logger) 
                        if (bert_loss == 0):
                            continue
                        bert_loss = bert_loss.mean() 
                        bert_losses.append(bert_loss.item())
                        rates.append(rate)
                        gt_traj_energys.append(gt_traj_energy)
                        mcmc_better_than_first_rates.append(is_better_than_first)
                        mcmc__better_than_all_rates.append(is_better_than_all)
                        mcmc_energys.append(mcmc_energy)
                        free_rates.append(free_rate)
                        action_correct_rates.append(action_correct_rate)
                        all_action_correct_rate_steps.append(action_correct_rate_steps)

                    bert_model.zero_grad()
                    bert_loss.backward()
                    torch.nn.utils.clip_grad_norm_(bert_model.parameters(), config.grad_norm_clip)
                    bert_optimizer.step()
                
                if is_train:
                    # backprop and update the parameters in model
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in bert_optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: bert loss {bert_loss.item():.5f}. lr {lr:e}") # gt_traj_energy {gt_traj_energy}. rate {rate}. target_rate {target_rate}
                    #pbar.set_description(f"epoch {epoch+1} iter {it}: dt loss {dt_loss.item():.5f}. lr {lr:e}")
                interval = 300
                # very initial result
                if (epoch_num ==0 and it == 2):
                    msg = f'Test it {it}, epoch_num {epoch_num}, bert loss {np.mean(bert_losses):.5f}, gt_traj_energys {np.mean(gt_traj_energys):.5f}, mcmc_better_than_first_rates {np.mean(mcmc_better_than_first_rates):.5f}, mcmc__better_than_all_rates {np.mean(mcmc__better_than_all_rates):.5f}, mcmc_energys {np.mean(mcmc_energys):.5f}, free_rates {np.mean(free_rates):.5f}, action_correct_rate {np.mean(action_correct_rates):.5f}, all_action_correct_rate_steps {np.mean(all_action_correct_rate_steps[-interval:],axis=0)}' #rates {np.mean(rates)}, 
                    self.logger.info(msg)

                if (it % interval == 0 and epoch_num >=0 and it > 1): #    % 4000  and it > 100
                    msg = f'Test it {it}, epoch_num {epoch_num}, bert loss {np.mean(bert_losses[-interval:]):.5f}, gt_traj_energys {np.mean(gt_traj_energys[-interval:]):.5f}, mcmc_better_than_first_rates {np.mean(mcmc_better_than_first_rates[-interval:]):.5f}, mcmc__better_than_all_rates {np.mean(mcmc__better_than_all_rates[-interval:]):.5f}, mcmc_energys {np.mean(mcmc_energys[-interval:]):.5f}, free_rates {np.mean(free_rates[-interval:]):.5f}, action_correct_rate {np.mean(action_correct_rates[-interval:]):.5f}, all_action_correct_rate_steps {np.mean(all_action_correct_rate_steps[-interval:],axis=0)}' #rates {np.mean(rates)}, 
                    self.logger.info(msg)
                    #tmp = np.mean(all_action_correct_rate_steps[-interval:],axis=0)
                    #if (np.isnan(tmp[-1])):
                    #    pdb.set_trace()
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            run_epoch('train', epoch_num=epoch)   

