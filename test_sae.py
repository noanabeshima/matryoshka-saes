"""Run with `python test_sae.py` (or pytest)."""
import copy
import warnings

import numpy as np
import torch

from sae import MatryoshkaSAE, RunningAvgNormalizer

warnings.filterwarnings("ignore")  # GradScaler warns when CUDA isn't available


def test_normalizer_starts_from_first_batch():
    torch.manual_seed(0)
    x = 5.0 * torch.randn(64, 32)
    normalizer = RunningAvgNormalizer()
    normalized = normalizer.normalize(x, update=True)
    assert torch.allclose(normalizer.running_avg, x.norm(dim=-1).mean().reshape(1))
    # Inputs come out with mean norm sqrt(d_model), not ~100x bigger
    assert np.isclose(normalized.norm(dim=-1).mean().item(), np.sqrt(32), rtol=1e-5)


def test_permutation_moves_optimizer_state_with_latents():
    torch.manual_seed(0)
    n_latents = 16
    sae = MatryoshkaSAE(
        d_model=8, n_latents=n_latents, n_prefixes=3, target_l0=4, n_steps=100
    )
    x = torch.randn(32, 8)
    # Take a first step so Adam has moments, then force a known re-sort (reverse order)
    np.random.seed(0)
    sae.step(x)
    sae.sq_act_running_avg.data = 1e6 * torch.arange(n_latents, dtype=torch.float)
    latent_perm = torch.arange(n_latents - 1, -1, -1)

    # Same step on a copy that doesn't re-sort, so the two should differ only by latent_perm
    unsorted = copy.deepcopy(sae)
    unsorted.permute_latents = False
    np.random.seed(1)
    sae.step(x)
    np.random.seed(1)
    unsorted.step(x)

    assert torch.allclose(sae.W_enc, unsorted.W_enc[:, latent_perm])
    # The running average used for sorting moved with the latents, so it is now in descending order
    assert torch.all(sae.sq_act_running_avg[:-1] >= sae.sq_act_running_avg[1:])
    for name, dim in (("W_dec", 0), ("W_enc", 1), ("b_enc", 0)):
        state = sae.optimizer.state[getattr(sae, name)]
        unsorted_state = unsorted.optimizer.state[getattr(unsorted, name)]
        for key in ("exp_avg", "exp_avg_sq"):
            expected = unsorted_state[key].index_select(dim, latent_perm)
            assert torch.allclose(state[key], expected), (name, key)


if __name__ == "__main__":
    test_normalizer_starts_from_first_batch()
    test_permutation_moves_optimizer_state_with_latents()
    print("all tests pass")
