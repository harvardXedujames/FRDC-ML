from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import Sampler


class RandomStratifiedSampler(Sampler[int]):
    def __init__(
        self,
        targets: torch.Tensor,
        num_samples: int | None = None,
    ) -> None:
        """Stratified sampling from a dataset, such that each class is
        sampled with equal probability.

        Examples:
            Use this with DataLoader to sample from a dataset in a stratified
            fashion. For example::

                ds = TensorDataset(...)
                dl = DataLoader(
                    ds,
                    batch_size=...,
                    sampler=RandomStratifiedSampler(),
                )

            This will use the targets' frequency as the inverse probability
            for sampling. For example, if the targets are [0, 0, 1, 2],
            then the probability of sampling the

        Args:
            targets: The targets to stratify by. Must be integers.
            num_samples: The number of samples to draw. If None, the
                number of samples is equal to the length of the dataset.
        """
        super().__init__()

        # Given targets [0, 0, 1]
        # bincount = [2, 1]
        # 1 / bincount = [0.5, 1]
        # 1 / bincount / len(bincount) = [0.25, 0.5]
        # The indexing then just projects it to the original targets.
        self.target_probs: torch.Tensor = (
            1 / (bincount := torch.bincount(targets)) / len(bincount)
        )[targets]

        self.num_samples = num_samples if num_samples else len(targets)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        """This should be a generator that yields indices from the dataset."""
        yield from torch.multinomial(
            self.target_probs,
            num_samples=self.num_samples,
            replacement=True,
        )
