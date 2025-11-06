import numpy as np

class HebbianTrader:
    def __init__(self, n_coins, window, hidden=64, s=0.5,
                 eta1=1e-3, eta2=1e-3, beta=0.05, kappa=0.0005,
                 tau=0.1, wclip=5.0, seed=0):
        rng = np.random.default_rng(seed)
        self.N, self.W = n_coins, window
        D = n_coins * window
        self.W1 = 0.1 * rng.standard_normal((hidden, D))
        self.b1 = np.zeros(hidden)
        self.W2 = 0.1 * rng.standard_normal((n_coins, hidden))
        self.b2 = np.zeros(n_coins)
        self.s, self.eta1, self.eta2 = s, eta1, eta2
        self.beta, self.kappa, self.tau, self.wclip = beta, kappa, tau, wclip
        self.prev_pos = np.zeros(n_coins)
        self.baseline = 0.0

    @staticmethod
    def _tanh(x): return np.tanh(x)

    def step(self, x_t, r_t):
        """
        x_t: shape (N*W,) windowed z-scored log-returns
        r_t: shape (N,) realized log-returns at this step
        """
        # Forward
        z = self._tanh(self.W1 @ x_t + self.b1) # (H,)
        y = self._tanh(self.W2 @ z + self.b2)   # (N,)
        pos = self.s * y

        # Reward from previous position
        pnl = float(self.prev_pos @ r_t)
        turnover = np.abs(pos - self.prev_pos).sum()
        R = pnl - self.kappa * turnover
        
        # Baseline and feedback
        self.baseline = (1 - self.beta) * self.baseline + self.beta * R
        delta = R - self.baseline

        # Hebbian Oja updates (local)
        # Output layer
        # outer(y, z) minus Oja stabilizer y^2 * W2 rowwise
        heb2 = np.outer(y, z)
        oja2 = (y**2)[:, None] * self.W2
        dW2 = self.eta2 * delta * (heb2 - oja2)
        db2 = self.eta2 * delta * y

        # Hidden layer
        heb1 = np.outer(z, x_t)
        oja1 = (z**2)[:, None] * self.W1
        dW1 = self.eta1 * delta * (heb1 - oja1)
        db1 = self.eta1 * delta * z

        self.W2 = np.clip(self.W2 + dW2, -self.wclip, self.wclip)
        self.b2 += db2
        self.W1 = np.clip(self.W1 + dW1, -self.wclip, self.wclip)
        self.b1 += db1

        # Risk gating
        action = np.where(np.abs(y) >= self.tau, y, 0.0)
        self.prev_pos = self.s * action
        return action, dict(R=R, delta=delta, pnl=pnl, turnover=turnover)


def stream(prices, window=32):
    logp = np.log(prices)
    r = np.diff(logp, axis=0)  # T-1 x N
    N = r.shape[1]
    agent = HebbianTrader(n_coins=N, window=window)
    from collections import deque
    buf = deque(maxlen=window)

    for t in range(r.shape[0]):
        buf.append(r[t])
        if len(buf) < window:
            continue
        X = np.array(buf)                      # W x N
        x_t = ((X - X.mean()) / (X.std() + 1e-6)).flatten()
        action, info = agent.step(x_t, r[t])
        # place orders using action here
