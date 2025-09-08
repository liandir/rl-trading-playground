import torch, math
from src.env import MultiCurrencyEnv, TradeResult


def geometric_brownian_step(
        p: torch.Tensor,
        mu: float,
        sigma: float,
        dt: float = 1.0,
        device=None
    ) -> torch.Tensor:
    device = device or p.device
    N = p.shape[0]
    z = torch.randn(N, device=device)
    drift = (mu - 0.5 * sigma**2) * dt
    shock = sigma * math.sqrt(dt) * z
    return p * torch.exp(drift + shock)


def demo_rollout(
        T: int = 200, 
        N: int = 4, 
        seed: int = 0,
        mu: float = 0.05, 
        sigma: float = 0.25
    ) -> tuple[list[float], list[list[float]], list[list[float]]]:
    torch.manual_seed(seed)

    # Initial state
    C0 = 10000.0
    p0 = torch.exp(torch.randn(N)) * 50.0  # random positive prices
    w0 = torch.zeros(N)
    s_fee = 0.001 * torch.ones(N)  # 10 bps sell fee
    b_fee = 0.001 * torch.ones(N)  # 10 bps buy fee

    env = MultiCurrencyEnv(C0=C0, w0=w0, p0=p0, sell_fee=s_fee, buy_fee=b_fee)

    V_hist = []
    C_hist = []
    w_hist = []
    p_hist = []

    for t in range(T):
        # Random policy: explores buys and sells weakly
        # a[0] controls investment budget fraction. Keep modest.
        a0 = 0.2 * torch.tanh(torch.randn(1))  # small positive budget on average
        a_cur = torch.tanh(0.7 * torch.randn(N))  # buy/sell signals
        a = torch.cat([a0, a_cur])

        # Evolve prices then step with new prices
        new_p = geometric_brownian_step(env.p, mu=mu, sigma=sigma)
        result = env.step(a, new_prices=new_p)

        V_hist.append(float(result.V))
        C_hist.append(float(result.C))
        w_hist.append(result.w.detach().cpu().tolist())
        p_hist.append(new_p.detach().cpu().tolist())

    return V_hist, w_hist, p_hist, C_hist