from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


class ActiveLearningDataset:
    """
    Wraps CIFAR-10 and manages the labeled / unlabeled index pools
    for the active learning loop.

    Attributes
    ----------
    full_dataset : torchvision Dataset
        The complete training set (images + labels, untouched).
    labeled_indices : list[int]
        Indices of examples that have been queried and labeled.
    unlabeled_indices : list[int]
        Indices of examples not yet queried.
    """

    # Normalisation statistics for CIFAR-10
    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2023, 0.1994, 0.2010)

    def __init__(self, root: str = "./data", download: bool = True):
        """
        Parameters
        ----------
        root : str
            Directory in which CIFAR-10 will be stored / read from.
        download : bool
            Download the dataset if it is not already present.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

        self.full_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=download,
            transform=transform,
        )

        self.test_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=False,
            download=download,
            transform=transform,
        )

        n = len(self.full_dataset)
        self.labeled_indices: list[int]   = []
        self.unlabeled_indices: list[int] = list(range(n))

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def query_samples(self, indices: list[int]) -> None:
        """Move `indices` from the unlabeled pool to the labeled pool.

        Parameters
        ----------
        indices : list[int]
            Dataset indices to label.  Must all be in unlabeled_indices.

        Raises
        ------
        ValueError
            If any index is not currently in the unlabeled pool.
        """
        unlabeled_set = set(self.unlabeled_indices)
        bad = [i for i in indices if i not in unlabeled_set]
        if bad:
            raise ValueError(f"Indices not in unlabeled pool: {bad}")

        self.labeled_indices   += indices
        self.unlabeled_indices  = [i for i in self.unlabeled_indices
                                   if i not in set(indices)]

    def reset(self) -> None:
        """Return all examples to the unlabeled pool."""
        n = len(self.full_dataset)
        self.labeled_indices   = []
        self.unlabeled_indices = list(range(n))

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------

    def get_labeled_loader(self, batch_size: int = 64,
                           shuffle: bool = True) -> DataLoader:
        """DataLoader over the currently labeled examples."""
        subset = Subset(self.full_dataset, self.labeled_indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=True)

    def get_unlabeled_loader(self, batch_size: int = 256,
                             shuffle: bool = False) -> DataLoader:
        """DataLoader over the currently unlabeled examples."""
        subset = Subset(self.full_dataset, self.unlabeled_indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=True)

    def get_test_loader(self, batch_size: int = 256) -> DataLoader:
        """DataLoader over the CIFAR-10 test set."""
        return DataLoader(self.test_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_labeled(self) -> int:
        return len(self.labeled_indices)

    @property
    def num_unlabeled(self) -> int:
        return len(self.unlabeled_indices)

    @property
    def classes(self) -> list[str]:
        return self.full_dataset.classes

    def __repr__(self) -> str:
        return (f"ActiveLearningDataset(CIFAR-10 | "
                f"labeled={self.num_labeled}, "
                f"unlabeled={self.num_unlabeled})")
