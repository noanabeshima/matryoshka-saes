import json
from pathlib import Path
import torch
import pytest
from toy_model import Tree, _sample_is_active


def test_sample_is_active_samples_according_to_prob():
    acts = _sample_is_active(
        0.25,
        batch_size=10_000,
        parent_feats_mask=None,
        force_active_mask=None,
        force_inactive_mask=None,
    )
    assert acts.shape == (10_000,)
    assert (acts.sum().item() / acts.shape[0]) == pytest.approx(0.25, abs=0.05)


def test_sample_is_active_sets_parent_acts_to_zero():
    parent_acts = _sample_is_active(
        0.25,
        batch_size=10_000,
        parent_feats_mask=None,
        force_active_mask=None,
        force_inactive_mask=None,
    )
    acts = _sample_is_active(
        0.25,
        batch_size=10_000,
        parent_feats_mask=parent_acts,
        force_active_mask=None,
        force_inactive_mask=None,
    )
    assert (parent_acts == 0).sum() > 10  # make sure we actually have some entries here
    assert (
        acts[parent_acts == 0].sum() == 0
    )  # acts should be 0 everywhere parent acts is 0
    assert (acts.sum().item() / (parent_acts > 0).sum()) == pytest.approx(
        0.25, abs=0.05
    )


def test_sample_is_active_respects_force_active_mask():
    batch_size = 10_000
    force_active_mask = torch.zeros(batch_size, dtype=torch.bool)
    force_active_mask[::2] = True  # Set every other sample to be forced active

    acts = _sample_is_active(
        0.25,
        batch_size=batch_size,
        parent_feats_mask=None,
        force_active_mask=force_active_mask,
        force_inactive_mask=None,
    )

    # Check forced active samples are all 1
    assert torch.all(acts[force_active_mask] == 1)

    # Check non-forced samples follow probability distribution
    non_forced = acts[~force_active_mask]
    assert non_forced.sum().item() / non_forced.shape[0] == pytest.approx(
        0.25, abs=0.05
    )


def test_sample_is_active_respects_force_inactive_mask():
    batch_size = 10_000
    force_inactive_mask = torch.zeros(batch_size, dtype=torch.bool)
    force_inactive_mask[::2] = True  # Set every other sample to be forced inactive

    acts = _sample_is_active(
        0.75,  # High probability to better test forced inactivity
        batch_size=batch_size,
        parent_feats_mask=None,
        force_active_mask=None,
        force_inactive_mask=force_inactive_mask,
    )

    # Check forced inactive samples are all 0
    assert torch.all(acts[force_inactive_mask] == 0)

    # Check non-forced samples follow probability distribution
    non_forced = acts[~force_inactive_mask]
    assert non_forced.sum().item() / non_forced.shape[0] == pytest.approx(
        0.75, abs=0.05
    )


def test_sample_is_active_force_masks_override_parent_mask():
    batch_size = 10_000
    parent_feats_mask = torch.ones(batch_size, dtype=torch.bool)
    force_active_mask = torch.zeros(batch_size, dtype=torch.bool)
    force_inactive_mask = torch.zeros(batch_size, dtype=torch.bool)

    # Set different regions for forced active/inactive
    force_active_mask[: batch_size // 3] = True
    force_inactive_mask[batch_size // 3 : batch_size // 2] = True

    acts = _sample_is_active(
        0.5,
        batch_size=batch_size,
        parent_feats_mask=parent_feats_mask,
        force_active_mask=force_active_mask,
        force_inactive_mask=force_inactive_mask,
    )

    # Check forced regions
    assert torch.all(acts[: batch_size // 3] == 1)  # force_active region
    assert torch.all(
        acts[batch_size // 3 : batch_size // 2] == 0
    )  # force_inactive region

    # Check remaining region follows probability with parent mask
    remaining = acts[batch_size // 2 :]
    assert remaining.sum().item() / remaining.shape[0] == pytest.approx(0.5, abs=0.05)


def test_Tree_sample_respects_rules_defined_in_the_tree_json():
    batch_size = 50_000
    with open(Path(__file__).parent.parent / "tree.json", "r") as f:
        tree_dict = json.load(f)
    tree = Tree(tree_dict)
    assert tree.n_features == 20
    batch = tree.sample(batch_size)
    assert batch.shape == (batch_size, 20)

    # every feature should fire at least once
    assert (batch.sum(dim=0) == 0).sum() == 0

    # the last 8 features fire independently with prob 0.05
    for i in range(8):
        assert batch[:, -i - 1].sum().item() / batch_size == pytest.approx(
            0.05, abs=0.01
        )

    # feats 0, 4, and 8 all fire independently with prob 0.15
    for i in [0, 4, 8]:
        parent_batch = batch[:, i]
        assert parent_batch.sum().item() / batch_size == pytest.approx(0.15, abs=0.01)

        # they each have 3 child features
        for j in range(3):
            child_idx = i + j + 1
            child_batch = batch[:, child_idx]
            # child should never fire unless parent fires
            assert torch.all(child_batch[parent_batch == 0] == 0)
            # each child should fire 20% of the time that the parent fires
            assert child_batch.sum() / parent_batch.sum() == pytest.approx(
                0.2, abs=0.05
            )

        # each child should never fire if another child fires
        # this means all these children should sum to 1 across a row
        assert torch.all((batch[:, i + 1 : i + 4].sum(dim=-1) > 1) == 0)
