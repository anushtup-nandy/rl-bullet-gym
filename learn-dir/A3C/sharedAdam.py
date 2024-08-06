import torch 

class ShAdam(torch.optim.Adam):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0,99), eps = 1e-8, weight_decay = 0):
        super(ShAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for grp in self.param_groups:
            for p in grp['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)[0]
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_mem(self):
        for grp in self.param_groups:
            for p in grp['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ShRMSProp(torch.optim.RMSprop):
    def __init__(self, params, lr = 1e-2, alphas=0.99, eps = 1e-8, weight_decay = 0):
        super(ShRMSProp, self).__init__(params, lr, alphas, eps, weight_decay, momentum=0, centred=False)
        for grp in self.param_groups:
            for p in grp['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)[0]
                state['square_avg'] = p.data.new().resize_as_(p.data).zero_()
                
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['square_avg'].share_memory_()