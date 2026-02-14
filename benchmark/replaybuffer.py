"""
Replay buffer for RNN-based agents in QMIX.
Stores full trajectories per episode, and returns fixed-length sequences during training.

Stored per episode:
    - local_obs_seq:  (T, N, obs_dim)
    - state_seq:      (T, state_dim)
    - action_seq:     (T, N)
    - r_total_seq:    (T,)          # team reward per timestep
    - next_local_seq: (T, N, obs_dim)
    - next_state_seq: (T, state_dim)
    - done_seq:       (T, N)
    - r_indiv_seq:    (T, N)        # optional (for Option-A individual TD loss)
    - mask_seq:       (T, N, A)        # optional (for variable action spaces)
    - next_mask_seq:  (T, N, A)        # optional (for variable action spaces)
    
Where:
    T        = episode length
    N        = number of agents

During training, sample(B, L) returns:

    - local_obs:  (B, L, N, obs_dim)
    - state:      (B, L, state_dim)
    - action:     (B, L, N)
    - r_tot:      (B, L)           # team reward
    - next_local: (B, L, N, obs_dim)
    - next_state: (B, L, state_dim)
    - done:       (B, L, N)
    - r_ind:      (B, L, N)        # if stored
    - mask:       (B, L, N, A)     # if stored
    - next_mask:  (B, L, N, A)     # if stored

Where:
    B        = batch size
    L        = length of sampled subsequence
"""

import random
from collections import deque
import torch

class ReplayBufferRNN: 

    def __init__(self, capacity:int , device: str="cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, 
             local_obs_seq: torch.Tensor,       
             state_seq: torch.Tensor,           
             action_seq: torch.Tensor,         
             r_total_seq: torch.Tensor,        
             next_local_obs_seq: torch.Tensor,  
             next_state_seq: torch.Tensor,      
             done_seq: torch.Tensor,            
             r_indiv_seq: torch.Tensor = None,
             mask_seq: torch.Tensor = None,
             next_mask_seq: torch.Tensor = None
            ):
        assert local_obs_seq.dim() == 3, "local_obs_seq should be (T, N, obs_dim)"
        assert state_seq.dim() == 2, "state_seq should be (T, state_dim)"
        assert next_local_obs_seq.dim() == 3, "next_local_obs_seq should be (T, N, obs_dim)"
        assert next_state_seq.dim() == 2, "next_state_seq should be (T, state_dim)"
        assert action_seq.dim() == 2, "action_seq should be (T, N)"
        assert done_seq.dim() == 2, "done_seq should be (T, N)"
        
        if r_total_seq.dim() == 2 and r_total_seq.size(-1) == 1:
            r_total_seq = r_total_seq.squeeze(-1)  # (T, )
        assert r_total_seq.dim() == 1, "r_total_seq should be (T, )"
        if r_indiv_seq is not None:
            assert r_indiv_seq.dim() == 2, "r_indiv_seq should be (T, N)"

        data = (
            local_obs_seq.detach(),
            state_seq.detach(),
            action_seq.detach(),
            r_total_seq.detach(),
            next_local_obs_seq.detach(),
            next_state_seq.detach(),
            done_seq.detach(),
            None if r_indiv_seq is None else r_indiv_seq.detach(),
            None if mask_seq is None else mask_seq.detach(),
            None if next_mask_seq is None else next_mask_seq.detach()
        )
        
        self.buffer.append(data)

    def sample(self, batch_size: int, seq_len: int, use_indiv: bool = True):
        
        assert len(self.buffer) > 0, "Replay buffer is empty."
        assert batch_size > 0, "batch_size must be positive."
        assert seq_len > 0, "seq_len must be positive."
    
        local_batch, state_batch, action_batch, r_tot_batch, next_local_batch, next_state_batch, done_batch = [], [], [], [], [], [], []
        r_ind_batch, mask_batch, next_mask_batch = [], [], [] 

        while len(state_batch) < batch_size:
            ep = random.choice(self.buffer)
            (lo_seq, s_seq, a_seq, r_tot_seq, nlo_seq, ns_seq, d_seq, r_ind_seq, mask_seq, next_mask_seq) = ep

            T = lo_seq.size(0)
            if T < seq_len:
                continue
            
            start_idx = random.randint(0, T - seq_len)
            end_idx = start_idx + seq_len

            local_batch.append(lo_seq[start_idx:end_idx])
            state_batch.append(s_seq[start_idx:end_idx])
            action_batch.append(a_seq[start_idx:end_idx])
            r_tot_batch.append(r_tot_seq[start_idx:end_idx])
            next_local_batch.append(nlo_seq[start_idx:end_idx])
            next_state_batch.append(ns_seq[start_idx:end_idx])
            done_batch.append(d_seq[start_idx:end_idx])
            if use_indiv and r_ind_seq is not None:
                r_ind_batch.append(r_ind_seq[start_idx:end_idx])
            else:
                r_ind_batch.append(None)
            if mask_seq is not None:
                mask_batch.append(mask_seq[start_idx:end_idx])
            else:
                mask_batch.append(None)
            if next_mask_seq is not None:
                next_mask_batch.append(next_mask_seq[start_idx:end_idx])
            else:
                next_mask_batch.append(None)
            
        obs_tensor = torch.stack(local_batch).to(self.device)     
        s_tensor = torch.stack(state_batch).to(self.device)       
        a_tensor = torch.stack(action_batch).to(self.device)       
        r_tot_tensor = torch.stack(r_tot_batch).to(self.device)    
        nobs_tensor = torch.stack(next_local_batch).to(self.device)
        ns_tensor = torch.stack(next_state_batch).to(self.device)  
        d_tensor = torch.stack(done_batch).to(self.device)

        def _stack_or_none(lst):
            if len(lst) == 0:
                return None
            if any(x is None for x in lst):
                return None
            return torch.stack(lst).to(self.device)
        
        r_ind_tensor = _stack_or_none(r_ind_batch)
        mask_tensor = _stack_or_none(mask_batch)
        next_mask_tensor = _stack_or_none(next_mask_batch)
        
        return obs_tensor, s_tensor, a_tensor, r_tot_tensor, nobs_tensor, ns_tensor, d_tensor, r_ind_tensor, mask_tensor, next_mask_tensor

    def __len__(self):
        return len(self.buffer)