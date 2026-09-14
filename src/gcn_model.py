"""
Graph Convolutional Network (GCN) classifier.

Implements the layer-wise propagation rule of Eq. (2):

    h_i^(l+1) = sigma( sum_{j in N(i)} (1/sqrt(d_i * d_j)) * W^(l) h_j^(l) + b^(l) )

using PyTorch Geometric's GCNConv, which implements exactly this
symmetric-normalized aggregation. sigma = ReLU between hidden layers;
the final layer outputs logits for binary crack / non-crack
classification (softmax applied via CrossEntropyLoss during training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CrackGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert num_layers >= 1

        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, num_classes))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, num_classes))

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # logits; apply softmax outside for probabilities


def train_gcn(
    data,
    train_mask,
    val_mask,
    hidden_channels=32,
    num_layers=2,
    lr=1e-3,
    weight_decay=5e-4,
    dropout=0.3,
    epochs=100,
    device="cpu",
):
    """Minimal full-batch training loop for the GCN (node classification)."""
    model = CrackGCN(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            val_pred = out[val_mask].argmax(dim=1).cpu().numpy()
            val_true = data.y[val_mask].cpu().numpy()

        from sklearn.metrics import f1_score

        val_f1 = f1_score(val_true, val_pred, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_f1
