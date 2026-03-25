import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader


class ResNet18FeatureExtractor(nn.Module):
    """
    Pre-trained ResNet-18 with the classification head removed.

    Returns the 512-dimensional penultimate-layer features,
    L2-normalised so that all embedding vectors lie on the unit sphere.
    This simulates the self-supervised SimCLR embedding required by
    the TPC_RP algorithm (Hacohen et al., 2022).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained
                    else None
        )

        # Drop the final fully-connected classification layer.
        # Everything up to (and including) the global average-pool
        # produces a 512-dim feature vector per image.
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (N, 3, H, W)

        Returns
        -------
        torch.Tensor  shape (N, 512)  — L2-normalised embeddings
        """
        # (N, 512, 1, 1)  →  (N, 512)
        features = self.encoder(x).flatten(start_dim=1)
        return F.normalize(features, p=2, dim=1)


# ---------------------------------------------------------------------------
# Helper: extract embeddings for an entire DataLoader
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: ResNet18FeatureExtractor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run `model` over every batch in `loader` and collect embeddings.

    Parameters
    ----------
    model  : ResNet18FeatureExtractor  (already moved to `device`)
    loader : DataLoader  (images, labels) or just images
    device : torch.device

    Returns
    -------
    embeddings : np.ndarray  shape (N, 512)
    labels     : np.ndarray  shape (N,)   — ground-truth class indices
    """
    model.eval()
    all_embeddings = []
    all_labels     = []

    for batch in loader:
        images, targets = batch
        images = images.to(device, non_blocking=True)

        emb = model(images).cpu().numpy()
        all_embeddings.append(emb)
        all_labels.append(targets.numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def build_extractor(pretrained: bool = True,
                    device: torch.device | None = None
                    ) -> tuple["ResNet18FeatureExtractor", torch.device]:
    """
    Convenience factory: build, move to device, and set to eval mode.

    Parameters
    ----------
    pretrained : bool
        Use ImageNet-pretrained weights (recommended).
    device : torch.device or None
        Target device; auto-detects CUDA/MPS if None.

    Returns
    -------
    (model, device)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = ResNet18FeatureExtractor(pretrained=pretrained).to(device)
    model.eval()
    return model, device