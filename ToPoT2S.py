"""
TopoT2S model. Demo version.

This file shows the TopoT2S architecture from the paper. The core is an
adaptive gate. At each time step the gate reads the node features and
decides how much to trust a static graph and how much to trust a dynamic
graph. The model learns this gate from data. No manual choice is needed.

Run it with:
    python topot2s_demo.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse


class TopoT2S(nn.Module):
    """TopoT2S. A static graph fused with a dynamic graph via an adaptive gate."""

    def __init__(self, num_nodes, in_channels, hidden_dim, num_classes,
                 static_adj, static_dim=0, heads=6):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.register_buffer('static_adj', static_adj)  # (N, N)

        # h^t = x_t W_0
        self.W0 = nn.Linear(in_channels, hidden_dim)
        self.norm_h = nn.LayerNorm(hidden_dim)

        # A^t_dy = softmax((h^t W_Q)(h^t W_K)^T / sqrt(d))
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # [alpha_static, alpha_dy] = softmax(W_o z^t)
        # z^t = mean_i (A^t_dy h^t W_V)_i
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)

        # g(t) = g0 exp(-lambda t)
        self.g0 = nn.Parameter(torch.tensor(1.0))
        self.reg_lambda = nn.Parameter(torch.tensor([0.05]))

        # spatial update
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads,
                              dropout=0.2, edge_dim=1)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False,
                              dropout=0.2, edge_dim=1)
        self.norm = nn.LayerNorm(hidden_dim)

        # temporal aggregation
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # second gate for the static context (age, gender, ...)
        self.static_dim = static_dim
        if static_dim > 0:
            self.static_proj = nn.Linear(static_dim, hidden_dim)
            self.static_gate = nn.Linear(hidden_dim + static_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_seq, static_ctx=None):
        B, T, N, Fdim = x_seq.shape
        device = x_seq.device

        h_all = self.norm_h(self.W0(x_seq))  # (B, T, N, d)

        step_vecs = []
        alpha_trace = []  # (alpha_static, alpha_dy) per step

        for t in range(T):
            h_t = h_all[:, t]  # (B, N, d)

            # dynamic adjacency from spatial self attention
            Q = self.W_Q(h_t)
            K = self.W_K(h_t)
            attn = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_dim)
            A_dy = torch.softmax(attn, dim=-1)  # (B, N, N)

            # adaptive gate
            V = self.W_V(h_t)
            z_t = torch.bmm(A_dy, V).mean(dim=1)  # (B, d)
            alpha = torch.softmax(self.W_o(z_t), dim=-1)  # (B, 2)
            alpha_static = alpha[:, 0].view(B, 1, 1)
            alpha_dy = alpha[:, 1].view(B, 1, 1)
            alpha_trace.append((alpha_static.mean().item(),
                                alpha_dy.mean().item()))

            # temporal decay of the static graph
            gt = torch.clamp(self.g0, min=0.0) * torch.exp(
                -torch.clamp(self.reg_lambda, min=0.01) * t)

            # fused adjacency
            A_fused = alpha_static * gt * self.static_adj.unsqueeze(0) \
                + alpha_dy * A_dy

            # spatial update with GAT
            h_flat = h_t.reshape(B * N, self.hidden_dim)
            ei_list, ew_list = [], []
            for b in range(B):
                ei, ew = dense_to_sparse(A_fused[b])
                ei_list.append(ei + b * N)
                ew_list.append(ew)
            edge_index = torch.cat(ei_list, dim=1)
            edge_weight = torch.cat(ew_list)

            h_spatial = F.elu(self.gat1(h_flat, edge_index,
                                        edge_attr=edge_weight.unsqueeze(-1)))
            h_spatial = F.elu(self.gat2(h_spatial, edge_index,
                                        edge_attr=edge_weight.unsqueeze(-1)))
            h_spatial = h_spatial.reshape(B, N, self.hidden_dim)

            h_out = self.norm(h_t + h_spatial)
            step_vecs.append(h_out.mean(dim=1))  # (B, d)

        h_seq = torch.stack(step_vecs, dim=1)  # (B, T, d)
        gru_out, _ = self.gru(h_seq)
        final = gru_out[:, -1]  # (B, d)

        if static_ctx is not None and self.static_dim > 0:
            s = self.static_proj(static_ctx)  # (B, d)
            gate = torch.sigmoid(self.static_gate(
                torch.cat([final, static_ctx], dim=-1)))
            final = gate * s + (1.0 - gate) * final

        return self.classifier(final), alpha_trace


def demo():
    torch.manual_seed(0)
    N, T, Fin, d, C = 5, 30, 2, 96, 4

    # a static graph, e.g. a Pearson correlation matrix
    static_adj = torch.rand(N, N)
    static_adj = (static_adj + static_adj.T) / 2
    static_adj.fill_diagonal_(0.0)

    model = TopoT2S(N, Fin, d, C, static_adj, static_dim=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TopoT2S with {n_params} parameters")

    x = torch.randn(2, T, N, Fin)
    static_ctx = torch.randn(2, 16)

    def mean_alpha(alpha_trace):
        a_s = sum(a[0] for a in alpha_trace) / len(alpha_trace)
        a_d = sum(a[1] for a in alpha_trace) / len(alpha_trace)
        return a_s, a_d

    # random init: the gate is balanced
    with torch.no_grad():
        logits, alpha_trace = model(x, static_ctx)
    a_s, a_d = mean_alpha(alpha_trace)
    print(f"logits: {tuple(logits.shape)}")
    print(f"random gate:      alpha_static={a_s:.3f}  alpha_dy={a_d:.3f}")

    # a trained gate that favors the dynamic graph
    with torch.no_grad():
        model.W_o.weight.zero_()
        model.W_o.bias.copy_(torch.tensor([0.0, 3.0]))
        _, alpha_trace = model(x, static_ctx)
    a_s, a_d = mean_alpha(alpha_trace)
    print(f"gate favors dyn:  alpha_static={a_s:.3f}  alpha_dy={a_d:.3f}")

    # a trained gate that favors the static graph
    with torch.no_grad():
        model.W_o.bias.copy_(torch.tensor([3.0, 0.0]))
        _, alpha_trace = model(x, static_ctx)
    a_s, a_d = mean_alpha(alpha_trace)
    print(f"gate favors stat: alpha_static={a_s:.3f}  alpha_dy={a_d:.3f}")

    print()
    print("The gate learns these weights from data. Strong features get a")
    print("high weight. Weak features get a weight close to zero. The model")
    print("picks its own direction. No manual setting is needed.")


if __name__ == '__main__':
    demo()
