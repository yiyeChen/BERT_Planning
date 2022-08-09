"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import pdb
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        #self.register_buffer("mask", torch.ones(config.block_size + 1, config.block_size + 1)
        #                             .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class ExpertControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out


class SimpleResidualBlock(nn.Module):
    def __init__(self, input_channel_size, out_channel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel_size, out_channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel_size, out_channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel_size)
        if stride == 1:
            if (input_channel_size == out_channel_size):
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Conv2d(input_channel_size, out_channel_size, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(input_channel_size, out_channel_size, kernel_size=1, stride=stride),
                                        nn.Conv2d(out_channel_size, out_channel_size, kernel_size=1, stride=stride))
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out = self.relu2(out + shortcut)        
        return out


class DT_GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


        self.state_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        #,nn.Flatten(), nn.Linear(512, config.n_embd)
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)
        self.word_embedding = nn.Embedding(100, config.n_embd)
        self.instr_rnn = nn.GRU(config.n_embd, config.n_embd, batch_first=True)

        self.controllers = []
        num_module = 2
        for ni in range(num_module):
            if ni < num_module-1:
                mod = ExpertControllerFiLM(
                    in_features=config.n_embd,
                    out_features=128, in_channels=128, imm_channels=128)
            else:
                mod = ExpertControllerFiLM(
                    in_features=config.n_embd, out_features=config.n_embd,
                    in_channels=128, imm_channels=128)
            self.controllers.append(mod)
            self.add_module('FiLM_Controler_' + str(ni), mod)
        self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d, torch.nn.MaxPool2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                #if (pn == 'FiLM_Controler_0.bias.weight'):
                #   pdb.set_trace()
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif (pn.find('weight') != -1 and isinstance(m, whitelist_weight_modules)): #and isinstance(m, whitelist_weight_modules)
                    decay.add(fpn)
                elif (pn.find('bias') != -1 and (fpn != 'FiLM_Controler_0.bias.weight' and fpn != 'FiLM_Controler_1.bias.weight')): # and isinstance(m, blacklist_weight_modules)
                    no_decay.add(fpn)


        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')
        no_decay.add('instr_rnn.bias_hh_l0')
        no_decay.add('instr_rnn.bias_ih_l0')
        #decay.add('FiLM_Controler_0.bias.weight')
        #decay.add('FiLM_Controler_1.bias.weight')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def _get_instr_embedding(self, instr):
        #pdb.set_trace()
        _, hidden = self.instr_rnn(self.word_embedding(instr))
        return hidden[-1]

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None, insts=None):
        # states: (batch, block_size, 3*7*7)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)
        #pdb.set_trace()
        state_embeddings = self.state_encoder(states.reshape(-1, 3, 7, 7).type(torch.float32).contiguous()) # (batch * block_size, n_embd)
        #pdb.set_trace()
        #state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch, block_size, n_embd)
        instr_embedding = self._get_instr_embedding(insts)
        instr_embedding = torch.repeat_interleave(instr_embedding.unsqueeze(1), states.shape[1], dim=1)
        instr_embedding = instr_embedding.reshape(-1, instr_embedding.size(2))

        #state_embeddings += instr_embedding
        # jointly learn the image and instr
        #pdb.set_trace()
        for controler in self.controllers:
            state_embeddings = controler(state_embeddings, instr_embedding)
            #pdb.set_trace()
        state_embeddings = F.relu(self.film_pool(state_embeddings)).squeeze(3).squeeze(2)
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd)
        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        #pdb.set_trace()
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        #pdb.set_trace()
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss


'''
class DT_GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.action_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.state_x_head = nn.Linear(config.n_embd, config.env_size, bias=False)
        self.state_y_head = nn.Linear(config.n_embd, config.env_size, bias=False)
        self.direction_head = nn.Linear(config.n_embd, 4, bias=False)

        self.block_size = config.block_size
        self.env_size = config.env_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        if (self.env_size == 22):
            self.state_encoder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2), # try remove maxpool
                SimpleResidualBlock(128, 128, 2)
            )
        elif (self.env_size == 8):
            self.state_encoder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        else:
            pdb.set_trace()

        self.state_embeddings = nn.Sequential(nn.Embedding(config.env_size, config.n_embd))
        self.state_embeddings_linear = nn.Sequential(nn.Linear(config.n_embd*config.state_dim, config.n_embd))
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)
        self.word_embedding = nn.Embedding(100, config.n_embd)
        self.instr_rnn = nn.GRU(config.n_embd, config.n_embd, batch_first=True)

        self.controllers = []
        num_module = 2
        for ni in range(num_module):
            if ni < num_module-1:
                mod = ExpertControllerFiLM(
                    in_features=config.n_embd,
                    out_features=128, in_channels=128, imm_channels=128)
            else:
                mod = ExpertControllerFiLM(
                    in_features=config.n_embd, out_features=config.n_embd,
                    in_channels=128, imm_channels=128)
            self.controllers.append(mod)
            self.add_module('FiLM_Controler_' + str(ni), mod)
        self.agent_pos_controllers = [] # fuse the fully observable image with agent goal position and current position
        
        for ni in range(num_module): #
            if ni < num_module-1:
                mod = ExpertControllerFiLM(
                    in_features=config.n_embd,
                    out_features=128, in_channels=128, imm_channels=128)
            else:
                mod = ExpertControllerFiLM(
                    in_features=config.n_embd, out_features=config.n_embd,
                    in_channels=128, imm_channels=128)
            self.agent_pos_controllers.append(mod)
            self.add_module('FiLM_Controler_' + str(ni+num_module), mod)

        if (self.env_size == 22):
            self.film_pool = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1152, config.n_embd), 
                nn.ReLU())
        elif (self.env_size == 8):
            self.film_pool = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, config.n_embd), 
                nn.ReLU())

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d, torch.nn.MaxPool2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                #if (pn == 'FiLM_Controler_0.bias.weight'):
                #   pdb.set_trace()
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif (pn.find('weight') != -1 and isinstance(m, whitelist_weight_modules)): #and isinstance(m, whitelist_weight_modules)
                    decay.add(fpn)
                elif (pn.find('bias') != -1 and fpn != 'FiLM_Controler_0.bias.weight' and fpn != 'FiLM_Controler_1.bias.weight' and 
                        fpn != 'FiLM_Controler_2.bias.weight' and fpn != 'FiLM_Controler_3.bias.weight'): # and isinstance(m, blacklist_weight_modules)
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')
        no_decay.add('instr_rnn.bias_hh_l0')
        no_decay.add('instr_rnn.bias_ih_l0')
        #decay.add('FiLM_Controler_0.bias.weight')
        #decay.add('FiLM_Controler_1.bias.weight')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def _get_instr_embedding(self, instr):
        #pdb.set_trace()
        _, hidden = self.instr_rnn(self.word_embedding(instr))
        return hidden[-1]

    # state, action, and return
    def forward(self, states, actions, target_states=None, target_actions=None, multi_state_masks=None, state_masks=None, action_masks=None, rtgs=None, timesteps=None, insts=None, full_image=None, mode='train'):
    #def forward(self, states, actions, targets=None, rtgs=None, timesteps=None, insts=None):
        # states: (batch, block_size, 3*7*7)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)

        image_embeddings = self.state_encoder(full_image.reshape(-1, 3, self.env_size, self.env_size).type(torch.float32).contiguous()) # (batch * block_size, n_embd)

        instr_embedding = self._get_instr_embedding(insts)
        instr_embedding = torch.repeat_interleave(instr_embedding.unsqueeze(1), states.shape[1], dim=1)
        instr_embedding = instr_embedding.reshape(-1, instr_embedding.size(2))

        state_embeddings = self.state_embeddings(states)
        state_embeddings = state_embeddings.reshape(states.shape[0]*states.shape[1], -1)
        state_embeddings = self.state_embeddings_linear(state_embeddings)

        # jointly learn the image and instr
        #print(full_image.shape, image_embeddings.shape, instr_embedding.shape, states.shape)
        for controler in self.controllers:
            image_embeddings = controler(image_embeddings, instr_embedding)

        #for controller in self.agent_pos_controllers:
        #    image_embeddings = controller(image_embeddings, state_embeddings)

        state_embeddings = self.film_pool(image_embeddings)        
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd)

        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(target_actions is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(target_actions is None):,:]
        elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(target_actions is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(target_actions is None):,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        #
        if (token_embeddings.shape[1] != position_embeddings.shape[1]):
            pdb.set_trace()
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        action_logits = self.action_head(x)
        direction_logits = self.direction_head(x)
        state_x_logits = self.state_x_head(x)
        state_y_logits = self.state_y_head(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = action_logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = action_logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            #pdb.set_trace()
            action_logits = action_logits[:, ::2, :] # only keep predictions from state_embeddings
            direction_logits = direction_logits[:,::2,:]
            x_logits = state_x_logits[:,::2,:]
            y_logits = state_y_logits[:,::2,:]
            #pdb.set_trace()
            #state_y_logits = state_y_logits[:,::2,:]
            #action_logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            #pdb.set_trace()
            action_logits = action_logits # for completeness
            direction_logits = direction_logits
            x_logits = state_x_logits
            y_logits = state_y_logits
        else:
            raise NotImplementedError()
            
        #if (mode == 'eval'):
        #    pdb.set_trace()
        # if we are given some desired targets also calculate the loss
        loss = None
        if target_actions is not None and mode == 'train':
            
            act_logits = action_logits.masked_select(action_masks).reshape(-1, self.config.vocab_size)
            act_targets = target_actions.masked_select(action_masks).reshape(-1)
            action_loss = F.cross_entropy(act_logits, act_targets,reduction='mean')
            #pdb.set_trace()
            direction_logtis = direction_logits.masked_select(state_masks).reshape(-1, 4)
            direction_targets = target_states.masked_select(multi_state_masks).reshape(-1)[2::3]
            direction_loss = F.cross_entropy(direction_logtis, direction_targets,reduction='mean')
            
            x_logtis = x_logits.masked_select(state_masks).reshape(-1, self.env_size)
            x_targets = target_states.masked_select(multi_state_masks).reshape(-1)[::3]
            x_loss = F.cross_entropy(x_logtis, x_targets,reduction='mean')
            #pdb.set_trace()
            y_logtis = y_logits.masked_select(state_masks).reshape(-1, self.env_size)
            y_targets = target_states.masked_select(multi_state_masks).reshape(-1)[1::3]
            y_loss = F.cross_entropy(y_logtis, y_targets,reduction='mean')

            loss = action_loss #x_loss + y_loss + direction_loss + action_loss
            #if (loss.item() < 1e-2):
            #    pdb.set_trace()
            #loss = action_loss
            #pdb.set_trace()
        return action_logits, direction_logits, x_logits, y_logits, loss
'''