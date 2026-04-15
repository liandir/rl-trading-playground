import torch
from src.environment.discrete.discrete_buckets import MultiCurrencyEnv, State


class VecMultiCurrencyEnv:
    """
    Vectorised wrapper around N independent MultiCurrencyEnv instances.

    Environments run sequentially in-process (no subprocesses). This is
    appropriate because individual env steps are cheap Python operations and
    IPC overhead would dominate with subprocess-based parallelism.

    No auto-reset: when an environment reaches `done=True` the caller is
    responsible for resetting it via `reset_at(i, data)`. This keeps reset
    timing fully under trainer control (e.g. the trainer may want to choose a
    new random start point before resetting).

    Returns
    -------
    reset(datas)
        obs    : (N, state_dim) tensor
        states : list[State] of length N

    step(actions, datas)
        obs    : (N, state_dim) tensor
        states : list[State] of length N
        rewards: (N,) tensor
        dones  : (N,) bool tensor

    valid_action_mask()
        mask   : (N, action_dim) bool tensor

    reset_at(i, data)
        obs    : (state_dim,) tensor
        state  : State
    """

    def __init__(self, envs: list[MultiCurrencyEnv]):
        assert len(envs) > 0, "envs must not be empty"
        self.envs = envs
        self.n_envs = len(envs)
        self.state_dim = envs[0].state_dim
        self.action_dim = envs[0].action_dim
        self.dtype = envs[0].dtype

    def reset(self, datas: list[dict]) -> tuple[torch.Tensor, list[State]]:
        """
        Reset all environments.

        datas: list of length n_envs, one market data dict per env.

        Returns stacked obs (N, state_dim) and list of State objects.
        """
        states = [env.reset(d) for env, d in zip(self.envs, datas)]
        obs = torch.stack([s.to_tensor() for s in states])
        return obs, states

    def reset_at(self, i: int, data: dict) -> tuple[torch.Tensor, State]:
        """
        Reset a single environment by index.

        Returns the obs tensor (state_dim,) and State for that environment.
        """
        state = self.envs[i].reset(data)
        return state.to_tensor(), state

    def step(
        self,
        actions: torch.Tensor | list | None,
        datas: list[dict],
    ) -> tuple[torch.Tensor, list[State], torch.Tensor, torch.Tensor]:
        """
        Step all environments.

        actions: (N,) integer tensor, list of length N, or None (hold all).
        datas  : list of length n_envs, one market data dict per env.

        Returns:
            obs    : (N, state_dim)
            states : list[State] of length N
            rewards: (N,) float tensor
            dones  : (N,) bool tensor
        """
        next_states = []
        rewards = []
        dones = []

        for i, (env, d) in enumerate(zip(self.envs, datas)):
            a_i = None if actions is None else actions[i]
            state, r, done, _ = env.step(a_i, d)
            next_states.append(state)
            rewards.append(r)
            dones.append(done)

        obs = torch.stack([s.to_tensor() for s in next_states])
        return (
            obs,
            next_states,
            torch.tensor(rewards, dtype=self.dtype),
            torch.tensor(dones, dtype=torch.bool),
        )

    def step_with_info(
        self,
        actions: torch.Tensor | list | None,
        datas: list[dict],
    ) -> tuple[torch.Tensor, list[State], torch.Tensor, torch.Tensor, list[dict]]:
        """
        Same as step() but also returns the list of info dicts.
        Use for evaluation/logging; skip during training to avoid the dict overhead.
        """
        next_states = []
        rewards = []
        dones = []
        infos = []

        for i, (env, d) in enumerate(zip(self.envs, datas)):
            a_i = None if actions is None else actions[i]
            state, r, done, info = env.step(a_i, d)
            next_states.append(state)
            rewards.append(r)
            dones.append(done)
            infos.append(info)

        obs = torch.stack([s.to_tensor() for s in next_states])
        return (
            obs,
            next_states,
            torch.tensor(rewards, dtype=self.dtype),
            torch.tensor(dones, dtype=torch.bool),
            infos,
        )

    def valid_action_mask(self) -> torch.Tensor:
        """
        Returns a (N, action_dim) bool tensor.
        True = action is valid for that environment given its current state.
        """
        masks = [env.valid_action_mask() for env in self.envs]
        return torch.stack(masks)
