import torch
import multiprocessing as mp


class DummyVecMultiCurrencyEnv:
    def __init__(self, envs):
        self.envs = envs
        self.n_envs = len(self.envs)
        self.dtype = envs[0].dtype if hasattr(envs[0], "dtype") else torch.float32

    def reset(self, datas):
        # datas: list of length N, each a dict
        states = [e.reset(d) for e, d in zip(self.envs, datas)]
        obs = torch.stack([s.to_tensor() for s in states], dim=0)  # (N, state_dim)
        return obs, states

    def step(self, actions, datas):
        # actions: torch.Tensor (N, act_dim) or list length N (or None per env)
        # datas: list of dict length N
        next_states, rewards, dones, infos = [], [], [], []
        for i, (e, d) in enumerate(zip(self.envs, datas)):
            a_i = None if actions is None else actions[i]
            s2, r, done, info = e.step(a_i, d)
            if done:
                # auto reset (optional). You must supply a new reset snapshot.
                # Here we just reset with current data d (you may want next snapshot).
                s2 = e.reset(d)

            next_states.append(s2)
            rewards.append(r)
            dones.append(done)
            infos.append(info)

        obs = torch.stack([s.to_tensor() for s in next_states], dim=0)  # (N, state_dim)
        return (
            obs,
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            infos
        )


def _worker(remote, parent_remote, env_fn_wrapper):
    """Worker loop that listens for commands from the main process."""
    parent_remote.close()
    env = env_fn_wrapper.var()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action, step_data = data
                s2, r, done, info = env.step(action, step_data)
                if done:
                    # Auto-reset logic as per your Dummy version
                    s2 = env.reset(step_data)
                remote.send((s2, r, done, info))
            elif cmd == 'reset':
                s2 = env.reset(data)
                remote.send(s2)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        pass

class CloudpickleWrapper:
    """Uses cloudpickle to serialize arbitrary functions (like env creators)."""
    def __init__(self, var):
        self.var = var
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.var)
    def __setstate__(self, obs):
        import cloudpickle
        self.var = cloudpickle.loads(obs)

class SubprocVecMultiCurrencyEnv:
    def __init__(self, env_fns):
        """
        env_fns: list of callable functions that create an environment.
        """
        self.waiting = False
        self.closed = False
        self.n_envs = len(env_fns)
        
        # Create pipes
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        
        # Start processes
        self.ps = []
        for (wrk, rem, env_fn) in zip(self.work_remotes, self.remotes, env_fns):
            p = mp.Process(target=_worker, args=(wrk, rem, CloudpickleWrapper(env_fn)))
            p.daemon = True 
            p.start()
            self.ps.append(p)
            wrk.close()

    def reset(self, datas):
        # Send reset command to all workers
        for remote, data in zip(self.remotes, datas):
            remote.send(('reset', data))
        
        states = [remote.recv() for remote in self.remotes]
        obs = torch.stack([s.to_tensor() for s in states], dim=0)
        return obs, states

    def step(self, actions, datas):
        # Send step command to all workers
        for i, (remote, data) in enumerate(zip(self.remotes, datas)):
            a_i = None if actions is None else actions[i]
            remote.send(('step', (a_i, data)))
        
        # Collect results
        results = [remote.recv() for remote in self.remotes]
        next_states, rewards, dones, infos = zip(*results)

        obs = torch.stack([s.to_tensor() for s in next_states], dim=0)
        return (
            obs,
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            list(infos)
        )

    def close(self):
        if self.closed: return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True