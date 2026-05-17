from collections import defaultdict
import random

import torch
from torch.utils.data import DataLoader, Sampler
import torchvision.datasets as datasets

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY=True

class BalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, targets, batch_size, seed=42):
        self.targets = list(targets)
        self.batch_size = batch_size
        self.seed = seed

        self.class_to_indices = defaultdict(list)
        for idx, y in enumerate(self.targets):
            self.class_to_indices[int(y)].append(idx)

        self.classes = sorted(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

        self.rng = random.Random(seed)

        for c in self.classes:
            self.rng.shuffle(self.class_to_indices[c])

        self.ptr = {c: 0 for c in self.classes}

        self.num_batches = len(self.targets) // batch_size

    def __len__(self):
        return self.num_batches

    def _next_index(self, c):
        indices = self.class_to_indices[c]
        p = self.ptr[c]

        if p >= len(indices):
            self.rng.shuffle(indices)
            p = 0

        idx = indices[p]
        self.ptr[c] = p + 1
        return idx

    def __iter__(self):
        for _ in range(self.num_batches):
            if self.batch_size >= self.num_classes:
                chosen = self.classes.copy()
                rest = self.batch_size - self.num_classes
                if rest > 0:
                    chosen += [self.rng.choice(self.classes) for _ in range(rest)]
            else:
                chosen = self.rng.sample(self.classes, self.batch_size)

            yield [self._next_index(c) for c in chosen]

def get_train_dataset_loader(data_dir, batch_size, generator_train):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"

    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )

    sampler = BalancedBatchSampler(
        targets=train_dataset.targets,
        batch_size=batch_size,
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    return train_dataset, train_loader
