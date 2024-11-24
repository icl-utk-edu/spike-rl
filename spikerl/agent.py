import torch
import torch.nn as nn
from spikerl.snn import (
    SpikingEncoderRegularSpike, SpikingEncoderPoissonSpike,
    SpikingDecoder, SpikeMLP
)
from spikerl.utils import MLPQFunction

class SpikingActor(nn.Module):
    """Population coding spike actor with fixed encoder."""
    def __init__(self, obs_dim, act_dim, en_pop_dim, de_pop_dim, hidden_sizes,
                 mean_range, std, spike_ts, act_limit, device, use_poisson):
        super().__init__()
        self.act_limit = act_limit
        if use_poisson:
            self.encoder = SpikingEncoderPoissonSpike(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        else:
            self.encoder = SpikingEncoderRegularSpike(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        self.snn = SpikeMLP(obs_dim * en_pop_dim, act_dim * de_pop_dim, hidden_sizes, spike_ts, device)
        self.decoder = SpikingDecoder(act_dim, de_pop_dim)

    def forward(self, obs, batch_size):
        in_pop_spikes = self.encoder(obs, batch_size)
        out_pop_activity = self.snn(in_pop_spikes, batch_size)
        action = self.act_limit * self.decoder(out_pop_activity)
        return action

class SpikingAC_Network(nn.Module):
    """Actor-Critic network using spiking neural networks."""
    def __init__(self, observation_space, action_space,
                 encoder_pop_dim, decoder_pop_dim, mean_range, std, spike_ts, device, use_poisson,
                 hidden_sizes=(4000, 4000), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.spikeRL = SpikingActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes,
                                    mean_range, std, spike_ts, act_limit, device, use_poisson)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, batch_size):
        with torch.no_grad():
            return self.spikeRL(obs, batch_size).cpu().numpy()
