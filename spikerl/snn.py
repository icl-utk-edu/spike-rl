import torch
import torch.nn as nn

# Parameters for SNN
ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
NEURON_CDECAY = 0.5
NEURON_VDECAY = 0.75
SPIKE_PSEUDO_GRAD_WINDOW = 0.5

class PseudoEncoderSpikeRegular(torch.autograd.Function):
    """Pseudo-gradient function for regular spike encoder."""
    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_REGULAR_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class PseudoEncoderSpikePoisson(torch.autograd.Function):
    """Pseudo-gradient function for Poisson spike encoder."""
    @staticmethod
    def forward(ctx, input):
        return torch.bernoulli(input).float()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class SpikingEncoderRegularSpike(nn.Module):
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.pop_dim = int(pop_dim)
        self.encoder_neuron_num = self.obs_dim * self.pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply

        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros((1, self.obs_dim, self.pop_dim), device=device)
        delta_mean = (mean_range[1] - mean_range[0]) / (self.pop_dim - 1)
        for num in range(self.pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.full((1, self.obs_dim, self.pop_dim), std, device=device)
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        obs = obs.view(-1, self.obs_dim, 1)
        pop_act = torch.exp(-0.5 * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        for step in range(self.spike_ts):
            pop_volt += pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt)
            pop_volt -= pop_spikes[:, :, step] * ENCODER_REGULAR_VTH
        return pop_spikes

class SpikingEncoderPoissonSpike(SpikingEncoderRegularSpike):
    """Population coding spike encoder with Poisson spike trains."""
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        super().__init__(obs_dim, pop_dim, spike_ts, mean_range, std, device)
        self.pseudo_spike = PseudoEncoderSpikePoisson.apply

    def forward(self, obs, batch_size):
        obs = obs.view(-1, self.obs_dim, 1)
        pop_act = torch.exp(-0.5 * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        for step in range(self.spike_ts):
            pop_spikes[:, :, step] = self.pseudo_spike(pop_act)
        return pop_spikes

class SpikingDecoder(nn.Module):
    """Population coding spike decoder."""
    def __init__(self, act_dim, pop_dim, output_activation=nn.Tanh):
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, pop_act):
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        raw_act = self.output_activation(self.decoder(pop_act).view(-1, self.act_dim))
        return raw_act

class PseudoSpikeRect(torch.autograd.Function):
    """Pseudo-gradient function for spike using rect function derivative."""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_output * spike_pseudo_grad.float()

class SpikeMLP(nn.Module):
    """Spike MLP with input and output population neurons."""
    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device):
        super().__init__()
        self.device = device
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.pseudo_spike = PseudoSpikeRect.apply
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0]).to(device)])
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.append(nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer]).to(device))
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim).to(device)

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, in_pop_spikes, batch_size):
        hidden_states = []
        for layer in range(self.hidden_num):
            hidden_states.append([
                torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device) for _ in range(3)
            ])
        out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device) for _ in range(3)]
        out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)
        for step in range(self.spike_ts):
            in_pop_spike_t = in_pop_spikes[:, :, step]
            hidden_states[0][0], hidden_states[0][1], hidden_states[0][2] = self.neuron_model(
                self.hidden_layers[0], in_pop_spike_t, hidden_states[0][0], hidden_states[0][1], hidden_states[0][2]
            )
            if self.hidden_num > 1:
                for layer in range(1, self.hidden_num):
                    hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2] = self.neuron_model(
                        self.hidden_layers[layer], hidden_states[layer-1][2], hidden_states[layer][0],
                        hidden_states[layer][1], hidden_states[layer][2]
                    )
            out_pop_states[0], out_pop_states[1], out_pop_states[2] = self.neuron_model(
                self.out_pop_layer, hidden_states[-1][2], out_pop_states[0], out_pop_states[1], out_pop_states[2]
            )
            out_pop_act += out_pop_states[2]
        out_pop_act = out_pop_act / self.spike_ts
        return out_pop_act
