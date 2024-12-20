import torch
from torch.utils.data import Dataset
import jsonschema
import json
import numpy as np


def validate_tree(instance):
    with open("tree.schema.json", "r") as f:
        schema = json.load(f)

    jsonschema.validate(instance, schema)


with open("tree.json") as f:
    tree_dict = json.load(f)

validate_tree(tree_dict)


class Tree:
    def __init__(self, tree_dict, start_idx=0):
        assert "active_prob" in tree_dict
        self.active_prob = tree_dict["active_prob"]
        self.is_read_out = tree_dict.get("is_read_out", True)
        self.mutually_exclusive_children = tree_dict.get(
            "mutually_exclusive_children", False
        )
        self.id = tree_dict.get("id", None)

        self.is_binary = tree_dict.get("is_binary", True)
        if self.is_read_out:
            self.index = start_idx
            start_idx += 1
        else:
            self.index = False

        self.children = []
        for child_dict in tree_dict.get("children", []):
            child = Tree(child_dict, start_idx)
            start_idx = child.next_index
            self.children.append(child)

        self.next_index = start_idx

        if self.mutually_exclusive_children:
            assert len(self.children) >= 2

    def __repr__(self, indent=0):
        s = " " * (indent * 2)
        s += (
            str(self.index) + " "
            if self.index is not False
            else " " * len(str(self.next_index)) + " "
        )
        s += "B" if self.is_binary else " "
        s += "x" if self.mutually_exclusive_children else " "
        s += f" {self.active_prob}"

        for child in self.children:
            s += "\n" + child.__repr__(indent + 2)
        return s

    @property
    def n_features(self):
        count = 1 if self.is_read_out else 0
        for child in self.children:
            count += child.n_features
        return count

    @property
    def child_probs(self):
        return torch.tensor([child.active_prob for child in self.children])

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        batch = torch.zeros((batch_size, self.n_features))
        self._fill_batch(batch)
        return batch

    def _fill_batch(
        self,
        batch: torch.Tensor,
        parent_feats_mask: torch.Tensor | None = None,
        force_active_mask: torch.Tensor | None = None,
        force_inactive_mask: torch.Tensor | None = None,
    ):
        batch_size = batch.shape[0]
        is_active = _sample_is_active(
            self.active_prob,
            batch_size=batch_size,
            parent_feats_mask=parent_feats_mask,
            force_active_mask=force_active_mask,
            force_inactive_mask=force_inactive_mask,
        )

        # append something if this is a readout
        if self.is_read_out:
            values = (
                is_active if self.is_binary else torch.rand_like(is_active) * is_active
            )
            batch[:, self.index] = values

        active_child = None
        if self.mutually_exclusive_children:
            active_child = torch.multinomial(
                self.child_probs.expand(batch_size, -1), 1
            ).squeeze(-1)

        for child_idx, child in enumerate(self.children):
            child_force_inactive = (
                None if active_child is None else active_child != child_idx
            )
            child_force_active = (
                None if active_child is None else active_child == child_idx
            )
            child._fill_batch(
                batch,
                parent_feats_mask=is_active,
                force_active_mask=child_force_active,
                force_inactive_mask=child_force_inactive,
            )


def _sample_is_active(
    active_prob,
    batch_size: int,
    parent_feats_mask: torch.Tensor | None,
    force_active_mask: torch.Tensor | None,
    force_inactive_mask: torch.Tensor | None,
):
    is_active = torch.bernoulli(torch.tensor(active_prob).expand(batch_size))
    if force_active_mask is not None:
        is_active[force_active_mask] = 1
    if force_inactive_mask is not None:
        is_active[force_inactive_mask] = 0
    if parent_feats_mask is not None:
        is_active[parent_feats_mask == 0] = 0
    return is_active


class TreeDataset(Dataset):
    def __init__(self, tree, true_feats, batch_size, num_batches):
        self.tree = tree
        self.true_feats = true_feats.cpu()
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        true_acts = self.tree.sample(self.batch_size)
        # random_scale = 1+torch.randn_like(true_acts, device=self.true_feats.device) * 0.05
        # true_acts = true_acts * random_scale
        x = true_acts @ self.true_feats
        return x
