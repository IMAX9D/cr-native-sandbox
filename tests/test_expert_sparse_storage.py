from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from expert_v1.compile_native_bc_dataset import _grid, _sparse_grid_row
from expert_v1.tick_store_v1.schema import (
    ActorEntity,
    ActorEpisode,
    ActorTick,
    PlayerPrivate,
    TowerState,
)
from expert_v1.training_v1.dataset import (
    NativeExpertSequenceDataset,
    collate_sequences,
)
from expert_v1.training_v1.losses import behaviour_cloning_loss
from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from expert_v1.training_v1.schema import (
    DatasetContractError,
    pack_sparse_grid,
    read_manifest,
    unpack_sparse_grid,
    unpack_sparse_position_masks,
    validate_shard,
)
from expert_v1.training_v1.smoke_data import create_smoke_dataset


class SparseNativeStorageTests(unittest.TestCase):
    def test_direct_sparse_grid_builder_matches_legacy_dense_quantization(self) -> None:
        actor = ActorTick(
            tick=777,
            actor_side=0,
            own_player=PlayerPrivate(0, 50_000, (0, 1, 2, 3), 4),
            towers=(
                TowerState(0, 0, 0, -1, 9000, 3000, 7728, 7728),
                TowerState(1, 0, 1, 0, 3500, 6500, 2429, 4858),
                TowerState(3, 1, 0, -1, 9000, 29000, 1, 7728),
            ),
            entities=(
                ActorEntity(1, 0, 8500, 10500, 26_000_000, 16, 1, 3, 0, 0),
                ActorEntity(2, 0, 8500, 10500, 26_000_001, 16, 2, 7, 0, 0),
                ActorEntity(3, 1, 14500, 24500, 26_000_002, 16, 7, 11, 0, 0),
            ),
            episode=ActorEpisode(0, 0, 0),
        )
        expected = _grid(actor)
        indices, values = _sparse_grid_row(actor)
        actual = np.zeros(expected.size, dtype=np.uint8)
        actual[indices] = values
        self.assertEqual(actual.reshape(expected.shape).tobytes(), expected.tobytes())

    def test_csr_roundtrip_is_byte_exact(self) -> None:
        rng = np.random.default_rng(20260827)
        dense = np.zeros((37, 8, 32, 18), dtype=np.uint8)
        for row in range(len(dense)):
            flat = dense[row].reshape(-1)
            cells = rng.choice(flat.size, size=53 + row % 11, replace=False)
            flat[cells] = rng.integers(1, 256, size=len(cells), dtype=np.uint8)
        offsets, indices, values = pack_sparse_grid(dense)
        rebuilt = unpack_sparse_grid(
            offsets,
            indices,
            values,
            start=0,
            stop=len(dense),
            channels=dense.shape[1],
        )
        self.assertEqual(rebuilt.tobytes(), dense.tobytes())
        partial = unpack_sparse_grid(
            offsets,
            indices,
            values,
            start=7,
            stop=29,
            channels=dense.shape[1],
        )
        self.assertEqual(partial.tobytes(), dense[7:29].tobytes())
        self.assertLess(indices.nbytes + values.nbytes + offsets.nbytes, dense.nbytes)

    def test_sparse_dataset_batch_and_model_loss_match_dense_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = create_smoke_dataset(Path(directory) / "dataset")
            manifest = read_manifest(root)
            shard = root / manifest["splits"]["train"][0]
            arrays = {
                path.stem: np.load(path, mmap_mode="r")
                for path in shard.glob("*.npy")
            }
            dense = unpack_sparse_grid(
                arrays["grid_offsets"],
                arrays["grid_indices"],
                arrays["grid_values"],
                start=0,
                stop=16,
                channels=int(manifest["dimensions"]["grid_channels"]),
            )
            dense_position = unpack_sparse_position_masks(
                arrays["selected_position_mask_rows"],
                arrays["selected_position_mask_packed"],
                start=0,
                stop=16,
            )
            dense_ability_position = unpack_sparse_position_masks(
                arrays["ability_position_mask_rows"],
                arrays["ability_position_mask_packed"],
                start=0,
                stop=16,
            )
            dataset = NativeExpertSequenceDataset(
                root, split="train", sequence_length=16, burn_in=4
            )
            sparse_item = dataset[0]
            self.assertEqual(
                torch.round(sparse_item["grid"] * 255).byte().numpy().tobytes(),
                dense.tobytes(),
            )
            self.assertEqual(
                sparse_item["position_mask"].numpy().tobytes(),
                dense_position.tobytes(),
            )
            self.assertEqual(
                sparse_item["ability_position_mask"].numpy().tobytes(),
                dense_ability_position.tobytes(),
            )

            # Emulate the former dense-row loader independently, then prove
            # the model logits and the complete conditional BC objective are
            # unchanged by the physical storage migration.
            dense_item = deepcopy(sparse_item)
            dense_item["grid"] = torch.from_numpy(dense.copy()).float().div_(255.0)
            dense_item["position_mask"] = torch.from_numpy(dense_position.copy())
            dense_item["ability_position_mask"] = torch.from_numpy(
                dense_ability_position.copy()
            )
            sparse_batch = collate_sequences([sparse_item])
            dense_batch = collate_sequences([dense_item])
            dimensions = manifest["dimensions"]
            config = ExpertPolicyConfig(
                grid_channels=dimensions["grid_channels"],
                public_scalar_size=dimensions["public_scalar_size"],
                card_vocab_size=dimensions["card_vocab_size"],
                ability_vocab_size=dimensions["ability_vocab_size"],
                max_ability_slots=dimensions["max_ability_slots"],
                entity_numeric_size=dimensions["entity_numeric_size"],
                hidden_size=32,
                card_embedding_size=16,
                spatial_size=16,
            )
            torch.manual_seed(27)
            model = RecurrentExpertPolicy(config).eval()
            with torch.no_grad():
                sparse_output = model.forward_batch(sparse_batch)
                dense_output = model.forward_batch(dense_batch)
                for left, right in zip(sparse_output[:-1], dense_output[:-1]):
                    self.assertTrue(torch.equal(left, right))
                sparse_loss, sparse_metrics = behaviour_cloning_loss(
                    sparse_output, sparse_batch, config
                )
                dense_loss, dense_metrics = behaviour_cloning_loss(
                    dense_output, dense_batch, config
                )
            self.assertTrue(torch.equal(sparse_loss, dense_loss))
            self.assertEqual(sparse_metrics, dense_metrics)
            self.assertLess(
                arrays["selected_position_mask_packed"].shape[0],
                arrays["position_label_mask"].shape[0],
            )
            dataset.close()
            del arrays

    def test_corrupt_csr_and_sparse_mask_indices_fail_closed(self) -> None:
        mutations = ("offset", "index", "mask_row", "legacy_dense")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = create_smoke_dataset(Path(directory) / "dataset")
                manifest = read_manifest(root)
                shard = root / manifest["splits"]["train"][0]
                if mutation == "offset":
                    offsets = np.load(shard / "grid_offsets.npy")
                    offsets[-1] += 1
                    np.save(shard / "grid_offsets.npy", offsets, allow_pickle=False)
                elif mutation == "index":
                    indices = np.load(shard / "grid_indices.npy")
                    offsets = np.load(shard / "grid_offsets.npy")
                    row = next(
                        index
                        for index, count in enumerate(np.diff(offsets))
                        if int(count) >= 2
                    )
                    start = int(offsets[row])
                    indices[start + 1] = indices[start]
                    np.save(shard / "grid_indices.npy", indices, allow_pickle=False)
                elif mutation == "mask_row":
                    rows = np.load(shard / "selected_position_mask_rows.npy")
                    rows[0] += 1
                    np.save(
                        shard / "selected_position_mask_rows.npy",
                        rows,
                        allow_pickle=False,
                    )
                else:
                    channels = int(manifest["dimensions"]["grid_channels"])
                    np.save(
                        shard / "grid.npy",
                        np.zeros((80, channels, 32, 18), dtype=np.uint8),
                        allow_pickle=False,
                    )
                with self.assertRaises(DatasetContractError):
                    validate_shard(shard, manifest)


if __name__ == "__main__":
    unittest.main()
