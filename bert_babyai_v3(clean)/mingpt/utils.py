"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb

AGENT_ID = 10
AGENT_COLOR = 6

RIGHT = [1,0]
LEFT = [-1,0]
UP = [0,-1]
DOWN = [0,1]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def bert_sample_multi_step(dt_model, bert_model, x, steps, rate, goal=None, temperature=1.0, sample=False, top_k=None, \
    actions=None, rtgs=None, timesteps=None, insts=None, full_obs=None, full_obs_wo_agent=None, logger=None, last_state=None, test_num=0):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = dt_model.get_block_size()

    dt_model.eval()
    bert_model.eval()
    sampled_actions = []

    cur_timestep = timesteps.cpu().numpy()[0,0,0]
    horizon = 5 #random.randint(3, 6)
    cur_timestep += horizon
    timesteps=((cur_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to('cuda'))
    batch_size = 1

    init_states = torch.clone(x[:, -1]).unsqueeze(1)
    init_states = torch.cat((init_states[:, :, :2],init_states[:, :, -2:]), dim=2) #init_states[:, :, :2] #
    last_states = torch.Tensor(last_state).unsqueeze(0).unsqueeze(0).to(dtype=torch.long).to('cuda')
    goals = torch.clone(x[:, -1, -2:]).cpu().to(dtype=torch.float32)
    goals = torch.repeat_interleave(torch.Tensor(goals).unsqueeze(1), horizon - 1, dim=1).to(dtype=torch.long).to('cuda')

    sample_states = [[0,0] for _ in range(horizon - 1)]
    sample_states = torch.repeat_interleave(torch.Tensor(sample_states).unsqueeze(0), batch_size, dim=0).to('cuda') #.to(dtype=torch.float32)
    #sample_states = torch.cat((sample_states, last_states), dim=1).to(dtype=torch.long)
    sample_states = torch.cat((sample_states, goals), dim=2)
    sample_states = torch.cat((init_states, sample_states), dim=1).to(dtype=torch.long) #, last_states

    init_obss = torch.clone(full_obs[0,-1]).cpu()
    init_obss = torch.repeat_interleave(torch.Tensor(init_obss).unsqueeze(1), horizon, dim=1).to(dtype=torch.float32).to('cuda')

    sample_actions = [[0] for i in range(horizon)]
    sample_actions = torch.repeat_interleave(torch.Tensor(sample_actions).unsqueeze(0), batch_size, dim=0).to(dtype=torch.long).to('cuda')
    
    init_temperature = 0.5
    sample_iteration = 50
    #pdb.set_trace()
    # MCMC construct sample trajectory
    for i in range(sample_iteration):
        temperature = init_temperature + i / (4*sample_iteration)
        action_masks = np.random.uniform(0, 1, (batch_size, horizon, 1)) > temperature
        action_masks = torch.Tensor(action_masks).to(torch.bool)

        action_logits = bert_model(sample_states, sample_actions, timesteps=timesteps, insts=insts, full_image=init_obss, mode='eval')
        action_val = torch.multinomial(F.softmax(action_logits.reshape(-1,action_logits.shape[2]), dim=-1), num_samples=1).reshape(batch_size,horizon)

        sample_actions[action_masks] = action_val.unsqueeze(2)[action_masks]
        iter_actions = torch.clone(sample_actions).cpu().numpy()
        sample_states = bert_model.update_sample_states(sample_states, iter_actions)
        #pdb.set_trace()
        if (i % 10 == 0 and test_num % 3 == 0):
            msg = f"iteration {i}, traj actions {sample_actions[:, -horizon:].cpu().numpy().squeeze(0)}, traj {sample_states[:, -horizon:].cpu().numpy().squeeze(0)}"
            logger.info(msg)
    
    
    #logger.info(str((f"after MCMC state: ",sample_states[:,-horizon:,:])))
    #logger.info(str((f"after MCMC actions: ",sample_actions[:,-horizon:,:])))
    mcmc_traj = sample_states[:, -horizon:].cpu().numpy().squeeze(0)
    last_action = sample_actions[0,-1,0].item()
    last_state = np.copy(mcmc_traj[-1,:])
    if (last_action == 0):
        last_state[:2] = last_state[:2] + RIGHT
    elif (last_action == 1):
        last_state[:2] = last_state[:2] + DOWN
    elif (last_action == 2):
        last_state[:2] = last_state[:2] + LEFT
    elif (last_action == 3):
        last_state[:2] = last_state[:2] + UP
    last_state[last_state<1] = 1
    last_state[last_state>bert_model.env_size-1] = bert_model.env_size-1

    mcmc_traj = np.concatenate((mcmc_traj, np.expand_dims(last_state,axis=0)), axis=0)
    sampled_actions = [action for action in sample_actions[0,-horizon:,0]]
    dt_model.train()
    bert_model.train()
    return sampled_actions, mcmc_traj, mcmc_traj


@torch.no_grad()
def dt_sample(model, x, steps, rate, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None, insts=None, full_obs=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        #pdb.set_trace()
        x_cond = x if x.size(1) <= (block_size//rate+1) else x[:, -block_size//rate-1:] # crop context if needed
        full_obs = full_obs if full_obs.size(1) <= (block_size//rate+1) else full_obs[:, -block_size//rate-1:]
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//rate else actions[:, -block_size//rate:] # crop context if needed
        #rtgs = rtgs if rtgs.size(1) <= block_size//rate else rtgs[:, -block_size//rate:] # crop context if needed
        logits, _ = model(x_cond, actions=actions, targets=None, timesteps=timesteps, insts=insts)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        x = ix

    return x


'''
@torch.no_grad()
def dt_sample_multi_step(dt_model, bert_model, x, steps, rate, goal=None, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None, insts=None, full_obs=None, full_obs_wo_agent=None, logger=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = dt_model.get_block_size()
    context_size = block_size//rate
    dt_model.eval()
    bert_model.eval()
    sampled_actions = []

    cur_timestep = timesteps.cpu().numpy()[0,0,0]
    horizon = 5
    for k in range(horizon):
        x_cond = x if x.size(1) <= (context_size+1) else x[:, -context_size-1:]
        full_obs_cond = full_obs if full_obs.size(1) <= (context_size+1) else full_obs[:, -context_size-1:]
        if actions is not None:
            actions_cond = actions if actions.size(1) <= context_size else actions[:, -context_size:]
        else:
            actions_cond = None
        rtgs = rtgs if rtgs.size(1) <= context_size else rtgs[:, -context_size:] 
        timesteps=((cur_timestep+k) * torch.ones((1, 1, 1), dtype=torch.int64).to('cuda'))
        act_logits, direction_logits, state_x_logits, state_y_logits, _ = dt_model(x_cond, actions=actions_cond, target_actions=None, rtgs=rtgs, timesteps=timesteps, insts=insts, full_image=full_obs_cond)
        # pluck the logits at the final step and scale by temperature
        act_logits = act_logits[:, -1, :] / temperature
        probs = F.softmax(act_logits, dim=-1)
        ia = torch.multinomial(probs, num_samples=1)
        if (actions is None):
            actions = ia.unsqueeze(0)
        else:
            actions = torch.cat((actions, ia.unsqueeze(0)), dim=1) 
        sampled_actions.append(ia)
        probs = F.softmax(direction_logits[:, -1, :], dim=-1)
        id = torch.multinomial(probs, num_samples=1)[0,-1]
        probs = F.softmax(top_k_logits(state_x_logits[:, -1, :], 5), dim=-1)
        ix = torch.multinomial(probs, num_samples=1)[0,-1]
        probs = F.softmax(top_k_logits(state_y_logits[:, -1, :], 5), dim=-1)
        iy = torch.multinomial(probs, num_samples=1)[0,-1]

        obs = torch.clone(full_obs_wo_agent)
        if (obs[ix,iy,0] == 1 or obs[ix,iy,0] == 4):
            obs[ix,iy,0] = AGENT_ID
            obs[ix,iy,1] = AGENT_COLOR
            obs[ix,iy,2] = id
        full_obs = torch.cat((full_obs, obs.flatten().unsqueeze(0).unsqueeze(0)), dim=1)
        next_state = [ix,iy,id]+goal
        next_state = torch.Tensor(next_state).type(torch.long).unsqueeze(0).unsqueeze(0).to('cuda')
        x = torch.cat((x, next_state), dim=1) 
    
    logger.info(str((f"before MCMC state: ",x[:, -horizon-1:-1])))
    ori_traj = x[:, -horizon-1:-1].cpu().numpy().squeeze(0)
    x_copy = torch.clone(x)
    full_obs_copy = torch.clone(full_obs)
    actions_copy = torch.clone(actions)

    # BERT MCMC
    x = x[:, -context_size-1:-1] # crop context if needed
    full_obs = full_obs[:, -context_size-1:-1]
    actions = actions[:, -context_size:] # crop context if needed
    for k in range(300):
        replan_node = random.randint(-horizon,-1)
        x_cond = torch.clone(x)#[:, :replan_node]
        full_obs_cond = torch.clone(full_obs)#[:, :replan_node]
        actions_cond = torch.clone(actions)#[:, :replan_node]
        if (replan_node != -1):
            x_cond[0,replan_node+1,0] = 0
            x_cond[0,replan_node+1,1] = 0
            x_cond[0,replan_node+1,2] = 0
            full_obs_cond[0, replan_node+1] = torch.clone(full_obs_wo_agent).flatten() #np.zeros((1, full_obs_cond.shape[2]))
        actions_cond[0,replan_node,0] = 0
        
        x_cond = x_cond if x_cond.size(1) <= (context_size) else x_cond[:, -context_size:]
        full_obs_cond = full_obs_cond if full_obs_cond.size(1) <= (context_size) else full_obs_cond[:, -context_size:]
        actions_cond = actions_cond if actions_cond.size(1) <= context_size else actions_cond[:, -context_size:]

        #timesteps=((cur_timestep+actions[:, :replan_node].size(1)) * torch.ones((1, 1, 1), dtype=torch.int64).to('cuda'))
        act_logits, direction_logits, state_x_logits, state_y_logits, _ = bert_model(x_cond, actions=actions_cond, target_actions=[], rtgs=rtgs, timesteps=timesteps, insts=insts, full_image=full_obs_cond, mode='eval')
        act_logits = act_logits[:, replan_node, :] / temperature
        probs = F.softmax(act_logits, dim=-1)
        ia = torch.multinomial(probs, num_samples=1)
        actions[0,replan_node,0] = ia[0,-1]

        if (replan_node != -1):
            probs = F.softmax(direction_logits[:, replan_node, :], dim=-1)
            id = torch.multinomial(probs, num_samples=1)[0,-1]
            x[0,replan_node+1,2] = id
            probs = F.softmax(top_k_logits(state_x_logits[:, replan_node, :], 5), dim=-1)
            ix = torch.multinomial(probs, num_samples=1)[0,-1]
            x[0,replan_node+1,0] = ix
            probs = F.softmax(top_k_logits(state_y_logits[:, replan_node, :], 5), dim=-1)
            iy = torch.multinomial(probs, num_samples=1)[0,-1]
            x[0,replan_node+1,1] = iy

            obs = torch.clone(full_obs_wo_agent)
            if (obs[ix,iy,0] == 1 or obs[ix,iy,0] == 4):
                obs[ix,iy,0] = AGENT_ID
                obs[ix,iy,1] = AGENT_COLOR
                obs[ix,iy,2] = id
            full_obs[0, replan_node+1] = obs.flatten()
    
    logger.info(str((f"after MCMC state: ",x[:,-horizon:,:])))
    mcmc_traj = x[:, -horizon:].cpu().numpy().squeeze(0)

    sampled_actions = [action for action in actions[0,-horizon:,0]]
    dt_model.train()
    bert_model.train()
    return sampled_actions, ori_traj, mcmc_traj

'''