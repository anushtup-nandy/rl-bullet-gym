import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def computeLogProb(logits):
    prob_v = F.softmax(logits, dim=1) #softmax distribution
    dist = torch.distributions.Categorical(prob_v)
    action = dist.sample().detach()
    print(action)
    return action.numpy()[0]

# def rollout(pi, counter, params, model,
#             hx, cx, frame_queue, env,
#             current_state, ep_lenght, action_length, 
#             layers_, tot_r, scores, lock, avg_ep,  scores_avg):
    