import torch
import torch.nn as nn
from .util import init
from torch import autograd
from packaging import version

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

if version.parse(torch.__version__) >= version.parse("2.0.0"):

    class _SafeTanh(autograd.Function):
        generate_vmap_rule = True

        @staticmethod
        def forward(input, eps = 1e-5):
            output = input.tanh()
            lim = 1.0 - eps
            output = output.clamp(-lim, lim)
            # ctx.save_for_backward(output)
            return output

        @staticmethod
        def setup_context(ctx, inputs, output):
            # input, eps = inputs
            # ctx.mark_non_differentiable(ind, ind_inv)
            # # Tensors must be saved via ctx.save_for_backward. Please do not
            # # assign them directly onto the ctx object.
            ctx.save_for_backward(output)

        @staticmethod
        def backward(ctx, *grad):
            grad = grad[0]
            (output,) = ctx.saved_tensors
            return (grad * (1 - output.pow(2)), None)

    class _SafeaTanh(autograd.Function):
        generate_vmap_rule = True

        @staticmethod
        def setup_context(ctx, inputs, output):
            tanh_val, eps = inputs
            # ctx.mark_non_differentiable(ind, ind_inv)
            # # Tensors must be saved via ctx.save_for_backward. Please do not
            # # assign them directly onto the ctx object.
            ctx.save_for_backward(tanh_val)
            ctx.eps = eps

        @staticmethod
        def forward(tanh_val, eps):
            lim = 1.0 - eps
            output = tanh_val.clamp(-lim, lim)
            # ctx.save_for_backward(output)
            output = output.atanh()
            return output

        @staticmethod
        def backward(ctx, *grad):
            grad = grad[0]
            (tanh_val,) = ctx.saved_tensors
            eps = ctx.eps
            lim = 1.0 - eps
            output = tanh_val.clamp(-lim, lim)
            return (grad / (1 - output.pow(2)), None)

    SafeTanh = _SafeTanh.apply
    SafeaTanh = _SafeaTanh.apply
else:

    def SafeTanh(input, eps):
            output = torch.tanh(input)
            lim = 1.0 - eps
            output = torch.clamp(output, min=-lim, max=lim)
            # ctx.save_for_backward(output)
            return output

    def SafeaTanh(tanh_input, eps=1e-5):
        lim = 1.0 - eps
        output =  torch.clamp(tanh_input, min=-lim, max=lim)
        # ctx.save_for_backward(output)
        output = torch.atanh(output)
        return output


class TanhNormal(torch.distributions.Normal):

    def __init__(self, loc, scale, validate_args=None):
        clamp_mean = torch.clamp(loc, min=-5, max=5)
        delta_temp = torch.clamp(scale, min=0, max=1)
        super().__init__(clamp_mean, delta_temp, validate_args=validate_args)
        self.eps = 1e-4



    @property
    def mean(self):
        temp=SafeTanh(super().mean, self.eps)
        delta_temp = super().stddev
        mean_=temp-temp*(1-temp*temp)*delta_temp*delta_temp
        return mean_
    
    @property
    def stddev(self):
        temp=SafeTanh(super().mean, self.eps)
        return (1-temp*temp)*super().stddev
    
    @property
    def variance(self):
        return self.stddev.pow(2)
    
    def sample(self,sample_shape=torch.Size()):
        normal_tensor=super().sample(sample_shape)
        return SafeTanh(normal_tensor, self.eps)
    
    def rsample(self,sample_shape=torch.Size()):
        normal_tensor=super().rsample(sample_shape)
        return SafeTanh(normal_tensor, self.eps)
    
    def log_prob(self, value):
        clamp_value = torch.clamp(value, min=-1+self.eps, max=1-self.eps)
        value_f=SafeaTanh(clamp_value, self.eps)
        log_prob_normal=super().log_prob(value_f)
        log_df=torch.log(1-torch.mul(clamp_value,clamp_value))
        return log_prob_normal-log_df
    
    def cdf(self,value):
        clamp_value = torch.clamp(value, min=-1+self.eps, max=1-self.eps)
        value_normal=SafeaTanh(clamp_value, self.eps)
        return super().cdf(value_normal)
    
    def icdf(self,value):
        return SafeTanh(super().icdf(value), self.eps)
    
    def entropy(self):
        mean_s=SafeTanh(super().mean, self.eps)
        entropy_=super().entropy()+ torch.log(1-mean_s.pow(2))-super().variance*(1-mean_s.pow(2))
        return entropy_
    
    def mode(self):
        return SafeTanh(super().mean, self.eps)

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean
    
class FixedTanhNormal(TanhNormal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        temp = super().mode()
        return temp


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())
    
class DiagTanhGaussian(nn.Module):
    def __init__(self,num_inputs,num_outputs,use_orthogonal=True,gain=0.01):
        super(DiagTanhGaussian,self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedTanhNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
