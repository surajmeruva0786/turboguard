import torch

from src.models.turboguard_cnn import CNNEncoder, TurboGuardCNN


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
