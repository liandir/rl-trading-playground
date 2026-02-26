import torch, math


class StochasticVanillaNetwork(torch.nn.Module):

    def __init__(self, n_in, n_out, sizes=[], act=torch.tanh):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.act = act
        
        # create layers
        self.layers = torch.nn.ModuleList()
        prev = n_in
        for h in sizes:
            self.layers.append(torch.nn.Linear(prev, h))
            prev = h
        last = prev
        self.mean = torch.nn.Linear(last, n_out)
        self.log_std = torch.nn.Linear(last, n_out)

    def forward(self, x, explore=False):
        z = x.view(-1, x.size(-1))
        for layer in self.layers:
            z = self.act(layer(z))

        mean = self.mean(z)
        if not explore:
            return mean, None
        
        log_std = self.log_std(z)
        log_std = torch.clamp(log_std, min=-20, max=5)

        return mean, log_std
    
    def sample(self, x, explore=True):
        mean, log_std = self.forward(x, explore)
        if not explore:
            return mean, None
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        logits = normal.rsample()
        log_prob = normal.log_prob(logits).sum(dim=-1, keepdim=True)

        return logits, log_prob
    

class SquashedStochasticVanillaNetwork(torch.nn.Module):

    def __init__(self, n_in, n_out, sizes=[], act=torch.tanh):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.act = act
        
        # create layers
        self.layers = torch.nn.ModuleList()
        prev = n_in
        for h in sizes:
            self.layers.append(torch.nn.Linear(prev, h))
            prev = h
        last = prev
        self.mean = torch.nn.Linear(last, n_out)
        self.log_std = torch.nn.Linear(last, n_out)

        # constant
        const = 0.5 * math.log(2.0 * math.pi)
        self.register_buffer("n_log_sqrt2pi", torch.tensor(const) * n_out)

    def forward(self, x, explore=False):
        z = x.view(-1, x.size(-1))
        for layer in self.layers:
            z = self.act(layer(z))

        m = self.mean(z)
        if not explore:
            return m, None
        
        log_s = torch.clamp(self.log_std(z), min=-20, max=5)

        return m, log_s
    
    def sample(self, x, explore=False):
        m, log_s = self.forward(x, explore)
        if not explore:
            a = torch.tanh(m)
            return a, None
        
        s = torch.exp(log_s)
        u = m + s * torch.randn_like(m)
        a = torch.tanh(u)

        log_prob_u = -(0.5 * ((u - m) / s)**2 + log_s).sum(-1, keepdim=True) - self.n_log_sqrt2pi
        log_prob_a = log_prob_u - torch.log(1 - a**2 + 1e-6).sum(-1, keepdim=True)

        return a, log_prob_a
    