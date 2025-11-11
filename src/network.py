import torch, math


class VanillaNetwork(torch.nn.Module):

    def __init__(self, n_in, n_out, sizes=[], act=torch.tanh, out_act=lambda x: x):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.act = act
        self.out_act = out_act
        
        self.layers = torch.nn.ModuleList()
        if len(sizes) > 0:
            self.layers.append(torch.nn.Linear(n_in, sizes[0]))
            for i in range(1, len(sizes)):
                self.layers.append(torch.nn.Linear(sizes[i-1], sizes[i]))
        self.out = torch.nn.Linear(sizes[-1] if len(sizes) > 0 else n_in, n_out)

    def forward(self, x):
        z = x.view(-1, self.n_in)
        for layer in self.layers:
            if z.size(-1) == layer.out_features:
                z = z + layer(z)
            else:
                z = layer(z)
        return self.out_act(self.out(z)).squeeze()
    

class VanillaQNetwork(VanillaNetwork):

    def __init__(self, state_size, action_size, sizes=[], act=torch.tanh, out_act=lambda x: x):
        super().__init__(state_size + action_size, 1, sizes, act, out_act)
        self.state_size = state_size
        self.action_size = action_size

    def forward(self, x, a):
        xa = torch.cat([x, a], dim=-1)
        return super().forward(xa)


class StochasticVanillaNetwork(torch.nn.Module):

    def __init__(self, n_in, n_out, sizes=[], act=torch.tanh):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.act = act
        
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(n_in, sizes[0]))
        for i in range(len(sizes)):
            self.layers.append(torch.nn.Linear(sizes[i-1], sizes[i]))
        self.mean = torch.nn.Linear(sizes[-1], n_out)
        self.log_std = torch.nn.Linear(sizes[-1], n_out)

    def forward(self, x):
        z = x.view(-1, x.size(-1))
        for layer in self.layers:
            if z.size(-1) == layer.out_features:
                z = z + layer(z)
            else:
                z = layer(z)
        mean = self.mean(z)
        log_std = self.log_std(z)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob
    

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

    def forward(self, x):
        z = x.view(-1, x.size(-1))
        for layer in self.layers:
            if z.size(-1) == layer.out_features:
                z = z + self.act(layer(z))
            else:
                z = self.act(layer(z))
        mean = self.mean(z)
        log_std = torch.clamp(self.log_std(z), min=-20, max=3)
        return mean, log_std
    
    def action(self, x):
        m, _ = self.forward(x)
        return torch.tanh(m)
    
    def sample(self, x):
        m, log_s = self.forward(x)
        s = torch.exp(log_s)

        u = m + s * torch.randn_like(m)
        a = torch.tanh(u)

        log_prob_u = -(0.5 * ((u - m) / s)**2 + log_s).sum(-1, keepdim=True) - self.n_log_sqrt2pi
        log_prob_a = log_prob_u - torch.log(1 - a**2 + 1e-6).sum(-1, keepdim=True)

        return a, log_prob_a