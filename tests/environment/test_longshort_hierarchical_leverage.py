import unittest

import torch

from src.environment.discrete.longshort_hierarchical_leverage import (
    BatchedLongShortHierarchicalLeverageEnv,
    LongShortHierarchicalLeverageEnv,
)


def scalar_market(price: float, t: float, *, n_assets: int = 1, dtype: torch.dtype = torch.float64) -> dict:
    close = torch.full((n_assets,), float(price), dtype=dtype)
    return {
        "close": close,
        "high": close.clone(),
        "low": close.clone(),
        "volume": torch.ones(n_assets, dtype=dtype),
        "time": float(t),
    }


def batched_market(prices: list[float], t: float, *, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, ...]:
    close = torch.tensor([[float(price)] for price in prices], dtype=dtype)
    return (
        close,
        close.clone(),
        close.clone(),
        torch.ones_like(close),
        torch.full((len(prices),), float(t), dtype=torch.float64),
    )


def close_outcome(*, committed: float, units: float, entry: float, price: float, close_fee: float, tax_rate: float) -> tuple[float, float]:
    pnl = units * (price - entry)
    gross = committed + pnl
    proceeds = max(gross - close_fee, 0.0)
    pre_tax = proceeds - committed
    tax = max(pre_tax, 0.0) * tax_rate
    net = proceeds - tax
    realized_pnl = net - committed
    return net, realized_pnl


class LongShortHierarchicalLeverageEnvTests(unittest.TestCase):
    dtype = torch.float64

    def make_env(self, **kwargs) -> LongShortHierarchicalLeverageEnv:
        params = dict(
            N=1,
            C0=100.0,
            tau_p=torch.tensor([1.0], dtype=self.dtype),
            size_buckets=(1.0,),
            max_leverage=3.0,
            dtype=self.dtype,
        )
        params.update(kwargs)
        return LongShortHierarchicalLeverageEnv(**params)

    def make_batched_env(self, **kwargs) -> BatchedLongShortHierarchicalLeverageEnv:
        params = dict(
            B=2,
            N=1,
            C0=100.0,
            tau_p=torch.tensor([1.0], dtype=self.dtype),
            size_buckets=(0.5, 1.0),
            max_leverage=2.5,
            dtype=self.dtype,
        )
        params.update(kwargs)
        return BatchedLongShortHierarchicalLeverageEnv(**params)

    def test_open_positions_use_margin_times_leverage(self):
        long_env = self.make_env()
        long_env.reset(data=scalar_market(10.0, 0.0, dtype=self.dtype))
        _, _, _, long_info = long_env.step((1, 0), data=scalar_market(10.0, 60.0, dtype=self.dtype))

        expected_margin = 99.0
        expected_units = 29.7
        self.assertTrue(long_info["valid_trade"])
        self.assertAlmostEqual(long_info["C"], 0.0, places=8)
        self.assertAlmostEqual(long_info["committed"][0], expected_margin, places=8)
        self.assertAlmostEqual(long_info["pos_units"][0], expected_units, places=8)
        self.assertAlmostEqual(long_info["gross_exposure"], expected_margin * 3.0, places=8)
        self.assertAlmostEqual(long_info["V"], expected_margin, places=8)

        short_env = self.make_env()
        short_env.reset(data=scalar_market(10.0, 0.0, dtype=self.dtype))
        _, _, _, short_info = short_env.step((2, 0), data=scalar_market(10.0, 60.0, dtype=self.dtype))

        self.assertTrue(short_info["valid_trade"])
        self.assertAlmostEqual(short_info["pos_units"][0], -expected_units, places=8)
        self.assertAlmostEqual(short_info["committed"][0], expected_margin, places=8)
        self.assertAlmostEqual(short_info["gross_exposure"], expected_margin * 3.0, places=8)
        self.assertAlmostEqual(short_info["V"], expected_margin, places=8)

    def test_close_realizes_leveraged_pnl(self):
        env = self.make_env()
        env.reset(data=scalar_market(10.0, 0.0, dtype=self.dtype))
        env.step((1, 0), data=scalar_market(10.0, 60.0, dtype=self.dtype))
        env.step((0, 0), data=scalar_market(12.0, 120.0, dtype=self.dtype))

        _, reward, done, info = env.step((3, 0), data=scalar_market(12.0, 180.0, dtype=self.dtype))

        expected_net, expected_realized = close_outcome(
            committed=99.0,
            units=29.7,
            entry=10.0,
            price=12.0,
            close_fee=1.0,
            tax_rate=0.26,
        )
        self.assertFalse(done)
        self.assertAlmostEqual(info["C"], expected_net, places=8)
        self.assertAlmostEqual(info["V"], expected_net, places=8)
        self.assertAlmostEqual(info["realized_cost"][0], 99.0, places=8)
        self.assertAlmostEqual(info["realized_pnl"][0], expected_realized, places=8)
        self.assertAlmostEqual(reward, expected_realized / 99.0, places=8)

    def test_flip_reopens_with_leveraged_notional(self):
        env = self.make_env()
        env.reset(data=scalar_market(10.0, 0.0, dtype=self.dtype))
        env.step((1, 0), data=scalar_market(10.0, 60.0, dtype=self.dtype))
        env.step((0, 0), data=scalar_market(12.0, 120.0, dtype=self.dtype))

        _, reward, _, info = env.step((2, 0), data=scalar_market(12.0, 180.0, dtype=self.dtype))

        expected_net, expected_realized = close_outcome(
            committed=99.0,
            units=29.7,
            entry=10.0,
            price=12.0,
            close_fee=1.0,
            tax_rate=0.26,
        )
        expected_margin = expected_net - 1.0
        expected_units = -(expected_margin * 3.0 / 12.0)

        self.assertTrue(info["valid_trade"])
        self.assertAlmostEqual(info["C"], 0.0, places=8)
        self.assertAlmostEqual(info["committed"][0], expected_margin, places=8)
        self.assertAlmostEqual(info["pos_units"][0], expected_units, places=8)
        self.assertAlmostEqual(info["entry_price"][0], 12.0, places=8)
        self.assertAlmostEqual(info["realized_cost"][0], 99.0, places=8)
        self.assertAlmostEqual(info["realized_pnl"][0], expected_realized, places=8)
        self.assertAlmostEqual(reward, expected_realized / 99.0, places=8)

    def test_mask_rejects_same_side_and_immediate_margin_breach_bucket(self):
        env = self.make_env(
            size_buckets=(0.5, 1.0),
            max_leverage=4.0,
            maintenance_margin_ratio=0.3,
        )
        env.reset(data=scalar_market(10.0, 0.0, dtype=self.dtype))

        mask = env.valid_action_mask()
        self.assertTrue(mask["primary"][1].item())
        self.assertTrue(mask["buy"][0, 0].item())
        self.assertFalse(mask["buy"][0, 1].item())
        self.assertFalse(mask["primary"][3].item())

        env.step((1, 0), data=scalar_market(10.0, 60.0, dtype=self.dtype))
        mask_after_open = env.valid_action_mask()
        self.assertFalse(mask_after_open["primary"][1].item())
        self.assertFalse(mask_after_open["buy"][0].any().item())
        self.assertTrue(mask_after_open["primary"][3].item())

    def test_liquidation_flattens_positions_and_sets_info(self):
        env = self.make_env()
        env.reset(data=scalar_market(10.0, 0.0, dtype=self.dtype))
        env.step((1, 0), data=scalar_market(10.0, 60.0, dtype=self.dtype))

        _, reward, done, info = env.step((0, 0), data=scalar_market(7.9, 120.0, dtype=self.dtype))

        expected_net, expected_realized = close_outcome(
            committed=99.0,
            units=29.7,
            entry=10.0,
            price=7.9,
            close_fee=1.0,
            tax_rate=0.26,
        )
        self.assertTrue(info["liquidated"])
        self.assertTrue(info["margin_breach"])
        self.assertFalse(done)
        self.assertAlmostEqual(info["C"], expected_net, places=8)
        self.assertAlmostEqual(info["V"], expected_net, places=8)
        self.assertAlmostEqual(info["realized_cost"][0], 99.0, places=8)
        self.assertAlmostEqual(info["realized_pnl"][0], expected_realized, places=8)
        self.assertAlmostEqual(reward, expected_realized / 99.0, places=8)
        self.assertAlmostEqual(info["gross_exposure"], 0.0, places=8)
        self.assertAlmostEqual(info["gross_leverage"], 0.0, places=8)
        self.assertAlmostEqual(info["maintenance_requirement"], 0.0, places=8)
        self.assertAlmostEqual(info["margin_buffer"], 1.0, places=8)
        self.assertEqual(info["pos_units"], [0.0])
        self.assertEqual(info["committed"], [0.0])

    def test_state_includes_gross_leverage_and_margin_buffer(self):
        env = self.make_env()
        env.reset(data=scalar_market(10.0, 0.0, dtype=self.dtype))
        next_state, _, _, _ = env.step((1, 0), data=scalar_market(10.0, 60.0, dtype=self.dtype))

        tensor = next_state.to_tensor()
        self.assertEqual(tensor.numel(), env.state_dim)
        self.assertTrue(torch.isfinite(tensor).all().item())
        self.assertAlmostEqual(float(tensor[8].item()), 3.0, places=8)
        self.assertAlmostEqual(float(tensor[9].item()), 0.5, places=8)

    def test_batched_env_matches_scalar_env_sequence(self):
        scalar_envs = [
            LongShortHierarchicalLeverageEnv(
                N=1,
                C0=100.0,
                tau_p=torch.tensor([1.0], dtype=self.dtype),
                size_buckets=(0.5, 1.0),
                max_leverage=2.5,
                dtype=self.dtype,
            ),
            LongShortHierarchicalLeverageEnv(
                N=1,
                C0=100.0,
                tau_p=torch.tensor([1.0], dtype=self.dtype),
                size_buckets=(0.5, 1.0),
                max_leverage=2.5,
                dtype=self.dtype,
            ),
        ]
        batched_env = self.make_batched_env()

        init_close, init_high, init_low, init_volume, init_time = batched_market([10.0, 10.0], 0.0, dtype=self.dtype)
        batched_obs = batched_env.reset(init_close, init_high, init_low, init_volume, init_time)
        scalar_states = [
            scalar_envs[0].reset(data=scalar_market(10.0, 0.0, dtype=self.dtype)),
            scalar_envs[1].reset(data=scalar_market(10.0, 0.0, dtype=self.dtype)),
        ]

        for idx, state in enumerate(scalar_states):
            torch.testing.assert_close(batched_obs[idx], state.to_tensor())

        steps = [
            (
                torch.tensor([[1, 1], [2, 0]], dtype=torch.long),
                [10.0, 10.0],
                60.0,
            ),
            (
                torch.tensor([[0, 0], [0, 0]], dtype=torch.long),
                [12.0, 9.0],
                120.0,
            ),
            (
                torch.tensor([[3, 0], [3, 0]], dtype=torch.long),
                [12.0, 9.0],
                180.0,
            ),
        ]

        for actions, prices, t in steps:
            close, high, low, volume, time = batched_market(prices, t, dtype=self.dtype)
            batched_obs, batched_rewards, batched_dones = batched_env.step(actions, close, high, low, volume, time)

            for idx, env in enumerate(scalar_envs):
                state, reward, done, _ = env.step(tuple(actions[idx].tolist()), data=scalar_market(prices[idx], t, dtype=self.dtype))
                torch.testing.assert_close(batched_obs[idx], state.to_tensor())
                self.assertAlmostEqual(float(batched_rewards[idx].item()), reward, places=8)
                self.assertEqual(bool(batched_dones[idx].item()), done)
                self.assertAlmostEqual(float(batched_env.C[idx].item()), float(env.C.item()), places=8)
                self.assertAlmostEqual(float(batched_env.V[idx].item()), float(env.V.item()), places=8)
                torch.testing.assert_close(batched_env.pos_units[idx], env.pos_units)
                torch.testing.assert_close(batched_env.committed[idx], env.committed)


if __name__ == "__main__":
    unittest.main()
