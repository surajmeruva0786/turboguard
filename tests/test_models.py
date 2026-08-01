import torch

from src.models.autoencoder import FeatureAutoencoder
from src.models.turboguard_cnn import CNNEncoder, TurboGuardCNN
from src.models.turboguard_hybrid import TurboGuardHybrid


def test_cnn_encoder_output_shape():
    encoder = CNNEncoder(n_channels=3)
    x = torch.randn(4, 3, 12000)
    feats = encoder(x)
    assert feats.shape == (4, 256)


def test_turboguard_cnn_forward_shape():
    model = TurboGuardCNN(n_channels=3, n_classes=5)
    x = torch.randn(2, 3, 12000)
    logits = model(x)
    assert logits.shape == (2, 5)


def test_turboguard_cnn_param_count_same_order_of_magnitude_as_readme():
    model = TurboGuardCNN()
    n_params = sum(p.numel() for p in model.parameters())
    # README quotes ~360k; our exact kernel/channel choices land a bit higher
    # (~500k) but should stay within the same order of magnitude.
    assert 200_000 < n_params < 1_000_000


def test_turboguard_cnn_backward_pass_runs():
    model = TurboGuardCNN()
    x = torch.randn(3, 3, 12000)
    y = torch.tensor([0, 1, 2])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_turboguard_hybrid_forward_shapes():
    model = TurboGuardHybrid(n_channels=3, n_classes=5)
    x = torch.randn(2, 10, 3, 12000)
    fault_logits, rul_pred = model(x)
    assert fault_logits.shape == (2, 5)
    assert rul_pred.shape == (2,)


def test_turboguard_hybrid_param_count_same_order_of_magnitude_as_readme():
    model = TurboGuardHybrid()
    n_params = sum(p.numel() for p in model.parameters())
    # README quotes ~880k; allow generous headroom for exact kernel/hidden-size choices.
    assert 500_000 < n_params < 3_000_000


def test_turboguard_hybrid_backward_pass_runs():
    model = TurboGuardHybrid()
    x = torch.randn(2, 10, 3, 12000)
    fault_logits, rul_pred = model(x)
    fault_labels = torch.tensor([0, 1])
    rul_targets = torch.tensor([50.0, 10.0])
    loss = torch.nn.functional.cross_entropy(fault_logits, fault_labels)
    loss = loss + torch.nn.functional.huber_loss(rul_pred, rul_targets)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_feature_autoencoder_roundtrip_shape():
    model = FeatureAutoencoder(input_dim=176, latent_dim=16)
    x = torch.randn(8, 176)
    recon = model(x)
    assert recon.shape == x.shape


def test_feature_autoencoder_reconstruction_error_shape_and_nonnegative():
    model = FeatureAutoencoder(input_dim=176, latent_dim=16)
    x = torch.randn(8, 176)
    err = model.reconstruction_error(x)
    assert err.shape == (8,)
    assert torch.all(err >= 0)


def test_feature_autoencoder_learns_to_reduce_reconstruction_error():
    torch.manual_seed(0)
    model = FeatureAutoencoder(input_dim=20, latent_dim=4, hidden_dims=(12,))
    x = torch.randn(16, 20)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial_err = model.reconstruction_error(x).mean().item()
    for _ in range(200):
        optim.zero_grad()
        loss = model.reconstruction_error(x).mean()
        loss.backward()
        optim.step()
    final_err = model.reconstruction_error(x).mean().item()
    assert final_err < initial_err
