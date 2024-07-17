import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

'''
prepares game frame for input into CNN.
1. grayscale it
2. resize it
3. Crop it
These 3 operations ensure that all frames are consistent in size and format.

Firstly, check for size of input frame:
- Atari games are 210x160x3 or 210x160x5 in size (pixels).
- reshape the frame according to them.

Secondly, convert the frames to grayscale using LUMINOSITY METHOD:
-  decrease the contribution of red color, and increase the contribution of the green color, 
    and put blue color contribution in between these two. So the new equation that form is: 
    New grayscale image = ( (0.3 * R) + (0.59 * G) + (0.11 * B) ).

Lastly, just crop and resize the img to remove some unnecessary parts for training/testin.
'''
def frame_proc(frame):
    if frame.size == 210*160*3:
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    elif frame.size == 250 * 160 * 3:
        img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
    else:
        assert "Unknown Resolution"
    img = img[:, :, 0]*0.3 + img[:, :, 1]*0.59 + img[:, :, 2]*0.11
    x_t = cv.resize(img, (84, 110), interpolation = cv.INTER_AREA)
    x_t = x_t[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    
    return x_t.astype(np.uint8)

'''
The Noop function initializes the state of an environment by performing a 
random number of "no-op" (no operation) actions. This is done to introduce 
some randomness in the starting state, which can help in training reinforcement 
learning agents by providing a diverse set of initial conditions.
'''
def No_op(env, actions_name, noop_max):
    _ = env.reset()
    assert actions_name[0] == "NOOP"
    noops = np.random.randint(1, noop_max + 1)
    action = 0
    init_state = None
    for i in range(noops):
        init_state, _, done, _,  _ = env.step(action)
        if done:
            init_state = env.reset()
    return init_state

def init_queue(queue, n_frame, init_frame, env, actions_name):
    queue.clear()
    init_frame = No_op(env, actions_name, noop_max = 30)
    for i in range(n_frame):
        queue.append(frame_proc(init_frame))
    
    return queue

'''
Skipping frames allows for decreased commpuation during training 
in addition max-pooling is performed by stacking every k frames 
together and taking the element-wise maximum.
Then we return the resulting frame as the current observation.
'''
def skip_frame(action, env, skip_frame = 4):
    skipped_frame = deque(maxlen = 2)
    skipped_frame.clear()
    total_rew = 0.0
    done = None
    for i in range(skip_frame):
        n_s, r, done, trunc, info = env.step(action)
        skipped_frame.append(n_s)
        total_rew += r
        if done:
            break
    max_frame = np.max(np.stack(skipped_frame), axis = 0)

    return max_frame, total_rew, done, info

'''
- The function concatenates the frames along the last axis 
(channel axis) using np.concatenate. This stacks the frames 
together to form a single multi-channel image.
- Normalize them to range the pixels between [0, 1]
- Convert to tensor
'''
def stack_frames(stacked_frames):
    #concatenate the frames 
    frames_stack = np.concatenate(stacked_frames, axis=-1)
    frames_stack = frames_stack.astype(np.float32) / 255.0 
    return torch.tensor(frames_stack, dtype=torch.float32)

'''
- c  = number of elements in x - N
- y is a zero array with c elements
- conv is an array of 1s with length N (used for convolution)
- For each iteration, the dot product of a slice of x (from i to i+N) 
and conv is computed, and the result is divided by N to get the mean. 
This value is stored in y[i].
'''
def running_mean(x, N = 100):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N]@conv)/N
    return y

def plot_avg_scores(array_avg, title):
    scores_avg = running_mean(np.array(array_avg), N=100)
    plt.title(title)
    plt.ylabel('Scores')
    plt.xlabel('Episode')
    plt.plot(scores_avg)
    plt.yticks(np.arange(np.min(scores_avg), np.max(scores_avg)+1, 3.0))
    plt.savefig('./plot_avg_scores.png')

def computeLogProb(logits):
    prob_v = F.softmax(logits, dim=1) #softmax distribution
    dist = torch.distributions.Categorical(prob_v)
    action = dist.sample().detach()
    print(action)
    return action.numpy()[0]

'''
- Function interacts with env to collect experiences over a series of steps
- Initialize lists
- set of fla
'''
def rollout(pi, counter, params, model,
            hx, cx, frame_queue, env,
            current_state, ep_length, actions_name, 
            layers_, tot_r, scores, lock, avg_ep,  scores_avg):
    states = []
    actions = []
    rewards = []
    masks = []
    hx_s = []
    cx_s = []
    steps_array = []

    fin_flag = False

    for i in range(params['rollout_size']):
        ep_length += 1
        current_state = current_state.unsqueeze(0).permute(0, 3, 1, 2)
        with torch.no_grad(): 
            #logits, values, hidden and cell states from current state
            logits, vals, (hx_, cx_) = model(current_state,(hx, cx))
            #get the action
            action = computeLogProb(logits)
        
        n_f, r, done, _ = skip_frame(action, env, skip_frame=4)
        states.append(current_state)
        actions.append(action)
        rewards.append(np.sign(r).astype(np.int8))
        masks.append(done)
        hx_s.append(hx)
        cx_s.append(cx)
        tot_r += r
        frame_queue.append(frame_proc(n_f))
        next_state = stack_frames(frame_queue)
        current_state = next_state
        hx, cx = hx_, cx_

        if ep_length > params['max_ep_length']:
            break

        #episode termination:
        if done: 
            #reset env:
            in_state_i = env.reset()
            frame_queue = init_queue(frame_queue, layers_["n_frames"], in_state_i, env, actions_name)
            #stack frames
            input_frames = stack_frames(frame_queue)
            current_state = input_frames
            ep_length = 0
            print(
                "Process: ", p_i,
                "Update:", counter.value,
                "| Ep_r: %.0f" % tot_rew,
            )
            print('------------------------------------------------------')
            flag_finish, scores_avg = print_avg(scores, p_i, tot_rew, lock, avg_ep, params, flag_finish, scores_avg)                        
            print('\n')
            if flag_finish == True:
                break
            
            tot_rew = 0
            hx = torch.zeros(1, layers_['lstm_dim'])
            cx = torch.zeros(1, layers_['lstm_dim'])

        #bootstrap
        with torch.no_grad():
            _, f_value , _  = model((current_state.unsqueeze(0).permute(0,3,1,2),(hx_, cx_)))
            steps_array.append((states, actions, rewards, masks, hx_s, cx_s, f_value))
       
        return hx, cx, steps_array, ep_length, frame_queue, current_state, tot_rew, counter, flag_finish, scores_avg
    
# def compute_returns(steps_array, gamma, model):