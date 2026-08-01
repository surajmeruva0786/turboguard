"""SHAP GradientExplainer for the deep-model view (README 13.2).

TurboGuard-Hybrid consumes raw waveforms, which don't have named features
to attribute importance to. To produce human-readable explanations
consistent with the classical SHAP-tree path, a small feed-forward network
is trained on the same ~176-dim engineered feature vector "computed in
parallel with the raw signal" (README's phrasing) and explained with
``shap.GradientExplainer`` instead. This auxiliary network is for
explainability only — it is not the fault-classification model deployed
elsewhere in the pipeline.
"""
from __future__ import annotations

import numpy as np
import shap
import torch
import torch.nn as nn

from src.utils.seed import set_seed


class FeatureMLP(nn.Module):
    """Small differentiable classifier over the engineered feature vector, for SHAP only."""

    def __init__(self, input_dim: int = 176, n_classes: int = 5, hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim, *hidden_dims]
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_feature_mlp(
    X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 1e-3, seed: int = 42, **model_kwargs
) -> FeatureMLP:
    set_seed(seed)
    model = FeatureMLP(input_dim=X.shape[1], n_classes=int(y.max()) + 1, **model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.from_numpy(X.astype(np.float32))
    labels = torch.from_numpy(y.astype(np.int64))

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def explain_deep_model(model: FeatureMLP, X_background: np.ndarray, X_explain: np.ndarray) -> np.ndarray:
    """SHAP values via GradientExplainer. Returns ``(n_samples, n_features, n_classes)``."""
    background = torch.from_numpy(X_background.astype(np.float32))
    explain = torch.from_numpy(X_explain.astype(np.float32))

    explainer = shap.GradientExplainer(model, background)
    raw = explainer.shap_values(explain)

    if isinstance(raw, list):
        return np.stack(raw, axis=-1)
    return raw
