"""
Spatial-Temporal Graph Convolutional Network for skeleton-based sign
recognition — the core model family proposed for SignSight (Yan et al. 2018;
adapted here to MediaPipe's hand+arm skeleton).

Where the GRU/LSTM baseline flattens every frame into one long vector and lets
the recurrence discover structure, ST-GCN is told the structure: keypoints are
nodes in a graph whose edges are the actual bones, so a convolution mixes a
joint only with the joints it is physically attached to. Layers alternate

  spatial graph convolution   X' = A_norm @ X @ W     (mix along bones)
  temporal convolution        1D conv over time       (mix along motion)

which is the inductive bias the proposal argued for: translation/scale
invariance and far fewer parameters than a dense sequence model.

Implemented directly in PyTorch rather than via PyTorch Geometric: the graph is
fixed and tiny (48 nodes), so a normalized adjacency matmul IS the graph
convolution, and it avoids a heavyweight dependency for one layer type.

Usage: imported by ml/evaluate.py as the "stgcn" model arm.
"""
import numpy as np
import torch
import torch.nn as nn

# MediaPipe indices, matching ml/features.py. Nodes kept: arms + both hands.
ARM = [11, 12, 13, 14, 15, 16]          # shoulders, elbows, wrists
LEFT_HAND = list(range(33, 54))
RIGHT_HAND = list(range(54, 75))
STGCN_NODES = ARM + LEFT_HAND + RIGHT_HAND   # 48 nodes

# Bones within one hand: wrist->knuckle chains, MediaPipe's standard topology.
_HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
               (0, 5), (5, 6), (6, 7), (7, 8),
               (0, 9), (9, 10), (10, 11), (11, 12),
               (0, 13), (13, 14), (14, 15), (15, 16),
               (0, 17), (17, 18), (18, 19), (19, 20),
               (5, 9), (9, 13), (13, 17)]   # knuckle ridge


def build_adjacency() -> np.ndarray:
    """Symmetric, self-looped, degree-normalized adjacency for STGCN_NODES.

    Normalization is D^-1/2 (A + I) D^-1/2, the standard GCN renormalization —
    without it, high-degree joints (wrists) dominate purely by connection count.
    """
    n = len(STGCN_NODES)
    index = {node: i for i, node in enumerate(STGCN_NODES)}
    A = np.eye(n, dtype=np.float32)

    def link(a, b):
        A[index[a], index[b]] = A[index[b], index[a]] = 1.0

    # Arm chains: shoulder -> elbow -> wrist, and across the shoulders.
    link(11, 13); link(13, 15)      # left arm
    link(12, 14); link(14, 16)      # right arm
    link(11, 12)                    # shoulder girdle
    # Wrist joins its hand's root, so hand and arm form one connected skeleton.
    link(15, LEFT_HAND[0]); link(16, RIGHT_HAND[0])
    for block in (LEFT_HAND, RIGHT_HAND):
        for a, b in _HAND_BONES:
            link(block[a], block[b])

    deg = A.sum(axis=1)
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    return (A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]).astype(np.float32)


class STGCNBlock(nn.Module):
    """One spatial graph conv followed by one temporal conv, with residual."""

    def __init__(self, in_ch, out_ch, adjacency, temporal_kernel=9, stride=1, dropout=0.3):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(adjacency))
        self.spatial = nn.Conv2d(in_ch, out_ch, kernel_size=1)   # the W in A@X@W
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(temporal_kernel, 1),
                      padding=((temporal_kernel - 1) // 2, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )
        # Residual keeps gradients alive through the stack; identity when shapes match.
        if in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):                     # x: (B, C, T, V)
        res = self.residual(x)
        x = self.spatial(x)
        # Graph convolution: mix each node with its neighbours along the bones.
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        x = self.temporal(x)
        return self.relu(x + res)


class STGCN(nn.Module):
    """Stacked ST-GCN blocks -> global average pool -> linear classifier.

    Deliberately small (3 blocks, 32->64 channels): with ~100 training clips a
    full 10-block ST-GCN would have orders of magnitude more parameters than
    samples. This is the same reasoning that caps the word model's forest depth.
    """

    def __init__(self, num_classes, in_channels=3, adjacency=None, dropout=0.3):
        super().__init__()
        A = build_adjacency() if adjacency is None else adjacency
        self.data_bn = nn.BatchNorm1d(in_channels * len(STGCN_NODES))
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 32, A, dropout=dropout),
            STGCNBlock(32, 64, A, stride=2, dropout=dropout),
            STGCNBlock(64, 64, A, dropout=dropout),
        ])
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):                     # x: (B, T, V, C)
        b, t, v, c = x.shape
        # Normalize per joint-channel across the batch before graph convs.
        x = x.permute(0, 3, 2, 1).contiguous()          # (B, C, V, T)
        x = self.data_bn(x.view(b, c * v, t)).view(b, c, v, t)
        x = x.permute(0, 1, 3, 2).contiguous()          # (B, C, T, V)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[2, 3])                          # global pool over time+joints
        return self.fc(x)


def to_graph_input(sequences: np.ndarray) -> np.ndarray:
    """(N, T, 75, 4) landmark windows -> (N, T, 48, 3) graph tensors.

    Keeps x, y, z for the arm+hand nodes only; visibility is dropped because
    absent hands are already all-zero rows, which the network reads as
    "no signal here" without needing a separate channel.
    """
    return sequences[:, :, STGCN_NODES, :3].astype(np.float32)
