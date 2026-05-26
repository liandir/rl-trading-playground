"""EMA encoder mixin for world-model networks.

Adds a frozen, slow-moving shadow of the encoder modules so the model-prediction
loss can be computed against a stationary target (BYOL/DreamerV2-style). The
mixin is opt-in via ``enable_ema_encoder=True`` in the host network constructor.

A host network must:
  * call ``self._init_ema_encoder(enable_ema_encoder)`` at the end of ``__init__``
    after the encoder submodules have been created.
  * implement ``_encoder_modules() -> dict[str, nn.Module]`` returning the
    online encoder submodules keyed by name.
  * implement ``_encode_tokens_with(x_flat, **modules) -> Tensor`` containing
    the encoder body, taking the modules as keyword arguments so the mixin can
    reuse the body with the EMA shadow.
"""
import copy
import torch


class EMAEncoderMixin:
    """Mixin providing a frozen EMA shadow of the encoder modules."""

    def _init_ema_encoder(self, enable: bool) -> None:
        """Create the EMA shadow (or mark it disabled).

        Args:
            enable (bool): If True, deepcopy the encoder submodules listed by
                ``_encoder_modules`` and freeze them. If False, no shadow is
                created and ``encode_target`` falls back to ``encode().detach()``.

        Returns:
            None: This function does not return a value.
        """
        self._ema_enabled = bool(enable)
        if not self._ema_enabled:
            self._ema_encoder = None
            return
        shadow = {}
        for name, module in self._encoder_modules().items():
            dup = copy.deepcopy(module)
            for p in dup.parameters():
                p.requires_grad_(False)
            shadow[name] = dup
        self._ema_encoder = torch.nn.ModuleDict(shadow)

    @property
    def has_ema_encoder(self) -> bool:
        """Return True when an EMA shadow of the encoder is active.

        Returns:
            bool: Whether the EMA shadow exists.
        """
        return bool(getattr(self, "_ema_enabled", False))

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` with the EMA shadow when enabled, else with the online encoder.

        Output always has no gradient attached. Shape matches ``encode(x)``.

        Args:
            x (torch.Tensor): The state tensor to encode.

        Returns:
            torch.Tensor: The (detached) target latent.
        """
        if not self.has_ema_encoder:
            return self.encode(x).detach()
        shape = x.shape
        x_flat = x.reshape(-1, self.state_dim)
        with torch.no_grad():
            tokens = self._encode_tokens_with(x_flat, **dict(self._ema_encoder.items()))
            z = tokens.reshape(x_flat.shape[0], self.latent_dim)
        return z.reshape(shape[:-1] + (self.latent_dim,))

    @torch.no_grad()
    def ema_update(self, tau: float) -> None:
        """Polyak-update the EMA shadow toward the online encoder.

        ``tau`` is the decay factor on the existing EMA weights, so larger
        values track more slowly. Buffers (e.g. registered constants, running
        statistics) are hard-copied. No-op when the shadow is disabled.

        Args:
            tau (float): Decay factor in [0, 1). Typical values 0.99-0.999.

        Returns:
            None: This function does not return a value.
        """
        if not self.has_ema_encoder:
            return
        if not 0.0 <= tau < 1.0:
            raise ValueError(f"ema_update expects tau in [0, 1), got {tau}.")
        online = self._encoder_modules()
        for name, src in online.items():
            dst = self._ema_encoder[name]
            for p_src, p_dst in zip(src.parameters(), dst.parameters()):
                p_dst.data.mul_(tau).add_(p_src.data, alpha=1.0 - tau)
            for b_src, b_dst in zip(src.buffers(), dst.buffers()):
                b_dst.data.copy_(b_src.data)

    @torch.no_grad()
    def reset_ema(self) -> None:
        """Reset the EMA shadow to the current online encoder weights.

        Useful after loading a checkpoint that does not contain EMA parameters.
        No-op when the shadow is disabled.

        Returns:
            None: This function does not return a value.
        """
        if not self.has_ema_encoder:
            return
        for name, src in self._encoder_modules().items():
            self._ema_encoder[name].load_state_dict(src.state_dict())
