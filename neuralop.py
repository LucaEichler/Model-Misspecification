import os

import torch

from torch import nn

import in_context_models
import metrics


#import debug
#from normalization import normalize_observations, renormalize

def normalize_min_max(series, min, max):
    return (series - min) / (max - min)

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters in {model.__class__.__name__}: {total_params:,}")


def normalize_observations(t_eval, t_observed, y_observed, mask):

    # Calculate mean and std of y_observed (masked)
    y_observed_masked = y_observed.masked_fill(~mask, float('nan'))
    y_observed_mean = torch.nanmean(y_observed_masked, dim=1, keepdim=True)
    y_observed_var = torch.nanmean((y_observed_masked - y_observed_mean) ** 2, dim=1, keepdim=True)
    y_observed_std = torch.sqrt(y_observed_var)

    # Normalize y_observed (via mean and std)
    y_normalized = (y_observed - y_observed_mean) / y_observed_std

    # Calculate min and max of t_observed (masked)
    t_observed_min = torch.min(t_observed.masked_fill(~mask, float('inf')), 1).values.unsqueeze(1)
    t_observed_max = torch.max(t_observed.masked_fill(~mask, float('-inf')), 1).values.unsqueeze(1)

    # Normalize t_observed (via min and max)
    t_observed_normalized = (t_observed - t_observed_min) / (t_observed_max - t_observed_min)

    # Allow t_eval to be None in input
    if t_eval is not None:
        # Normalize t_eval (via min and max of t_observed (-> can be <0 or >1))
        t_eval_normalized = normalize_min_max(t_eval, t_observed_min, t_observed_max)
    else: t_eval_normalized = None

    return t_eval_normalized, t_observed_normalized, y_normalized, t_observed_min, y_observed_mean, t_observed_max, y_observed_std


def normalize_multiple_windows(t_observed, y_observed, mask):
    # Ensure the length of t_observed, y_observed, and mask is a multiple of 128
    assert t_observed.shape[1] % 128 == 0, "Length of t_observed must be a multiple of 128"
    assert y_observed.shape[1] == t_observed.shape[1], "Length of y_observed must be equal to length of t_observed"
    assert mask.shape[1] == t_observed.shape[1], "Length of mask must be equal to length of t_observed"

    num_windows = t_observed.shape[1] // 128

    # Split into windows
    t_windows = t_observed.split(128, dim=1)
    y_windows = y_observed.split(128, dim=1)
    mask_windows = mask.split(128, dim=1)

    # Normalize y_observed for each window separately
    y_normalized_windows = []
    y_means = []
    y_stds = []
    y_mins = []
    y_maxs = []
    y_ranges = []
    y_firsts = []
    y_lasts = []
    y_diffs = []
    t_firsts = []
    t_lasts = []
    t_diffs = []

    for y_window, t_window, mask_window in zip(y_windows, t_windows, mask_windows):
        y_window_masked = y_window.masked_fill(~mask_window, float('nan'))
        y_mean = torch.nanmean(y_window_masked, dim=1, keepdim=True)
        y_var = torch.nanmean((y_window_masked - y_mean) ** 2, dim=1, keepdim=True)
        y_std = torch.sqrt(y_var)
        y_normalized = (y_window - y_mean) / y_std
        y_normalized_windows.append(y_normalized)
        y_means.append(y_mean)
        y_stds.append(y_std)

        y_min = torch.min(y_window_masked.masked_fill(~mask_window, float('inf')), dim=1, keepdim=True).values
        y_max = torch.max(y_window_masked.masked_fill(~mask_window, float('-inf')), dim=1, keepdim=True).values

        y_range = y_max - y_min
        # Find the indices of the first True in the mask for each row

        # Create a range tensor for each row
        row_indices = torch.arange(y_window.shape[1], device=y_window.device).expand(y_window.shape[0], -1)

        # Mask the indices
        masked_indices = row_indices * mask_window
        masked_indices_min = masked_indices.masked_fill(~mask_window, 200)

        # Find the argmin and argmax of the masked indices
        first_indices = torch.min(masked_indices_min, dim=1).indices
        last_indices = torch.max(masked_indices, dim=1).indices

        # Use advanced indexing to extract values
        y_first = y_window.gather(1, first_indices.unsqueeze(1))
        y_last = y_window.gather(1, last_indices.unsqueeze(1))

        y_diff = y_last - y_first

        y_mins.append(y_min)
        y_maxs.append(y_max)
        y_ranges.append(y_range)
        y_firsts.append(y_first)
        y_lasts.append(y_last)
        y_diffs.append(y_diff)


        t_first = torch.min(t_window.masked_fill(~mask_window, float('inf')), dim=1, keepdim=True).values
        t_last = torch.max(t_window.masked_fill(~mask_window, float('-inf')), dim=1, keepdim=True).values

        t_diff = t_last - t_first

        t_firsts.append(t_first)
        t_lasts.append(t_last)
        t_diffs.append(t_diff)

    y_observed_normalized = torch.cat(y_normalized_windows, dim=1)
    y_mean = torch.cat(y_means, dim=1).unsqueeze(-1)
    y_std = torch.cat(y_stds, dim=1).unsqueeze(-1)
    y_min = torch.cat(y_mins, dim=1).unsqueeze(-1)
    y_max = torch.cat(y_maxs, dim=1).unsqueeze(-1)
    y_range = torch.cat(y_ranges, dim=1).unsqueeze(-1)
    y_first = torch.cat(y_firsts, dim=1).unsqueeze(-1)
    y_last = torch.cat(y_lasts, dim=1).unsqueeze(-1)
    y_diff = torch.cat(y_diffs, dim=1).unsqueeze(-1)
    t_first = torch.cat(t_firsts, dim=1).unsqueeze(-1)
    t_last = torch.cat(t_lasts, dim=1).unsqueeze(-1)
    t_diff = torch.cat(t_diffs, dim=1).unsqueeze(-1)

    # Normalize t_observed using min-max normalization
    t_min = torch.min(t_windows[0].masked_fill(~mask_windows[0], float('inf')), 1).values.unsqueeze(1)
    t_max = torch.max(t_windows[-1].masked_fill(~mask_windows[-1], float('-inf')), 1).values.unsqueeze(1)

    t_normalized_windows = []
    for i, (t_window, mask_window) in enumerate(zip(t_windows, mask_windows)):
        if i == 0:
            scaling_min = t_min
        else:
            prev_max = torch.max(t_windows[i-1].masked_fill(~mask_windows[i-1], float('-inf')), 1).values.unsqueeze(1)
            curr_min = torch.min(t_window.masked_fill(~mask_window, float('inf')), 1).values.unsqueeze(1)
            scaling_min = (prev_max + curr_min) / 2

        if i == num_windows - 1: #last window doesnt have a next window
            scaling_max = t_max
        elif i == num_windows - 2: # penultimate window must not use zero-padded values from last window (and shouldnt use the last window in general)
            curr_max = torch.max(t_window.masked_fill(~mask_window, float('-inf')), 1).values.unsqueeze(1)
            scaling_max = curr_max
        else:
            next_min = torch.min(t_windows[i+1].masked_fill(~mask_windows[i+1], float('inf')), 1).values.unsqueeze(1)
            curr_max = torch.max(t_window.masked_fill(~mask_window, float('-inf')), 1).values.unsqueeze(1)
            scaling_max = (next_min + curr_max) / 2

        t_normalized = (t_window - scaling_min) / (scaling_max - scaling_min)
        t_normalized_windows.append(t_normalized)

    t_observed_normalized = torch.cat(t_normalized_windows, dim=1)

    statistics = (
        y_mean, y_std, y_min, y_max, y_range, y_first, y_last, y_diff,
        t_first, t_last, t_diff
    )

    return t_observed_normalized, y_observed_normalized, statistics


def renormalize(output_normalized, y_mean, y_std):
    return output_normalized * y_std + y_mean


# Hyperparameters
temporal_embedding_dim = 21
transformer_dim = temporal_embedding_dim*3 + 1
transformer_nhead = 8
transformer_encoder_layers = 3
transformer_ff_dim = 256
ffn_hidden_dim = 512
ffn_num_hidden_layers = 2
branch_net_num_hidden_layers = 1
branch_trunk_output_dim = 128
mlp_hidden_dim1 = 512
mlp_hidden_dim2 = 256


def ffn(input_dim=512, output_dim=512, num_hidden_layers=2, ffn_hidden_dim=1024): #TODO dropout testen
    layers = [nn.Linear(input_dim, ffn_hidden_dim), nn.LeakyReLU(negative_slope=0.1)]

    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(ffn_hidden_dim, ffn_hidden_dim))
        layers.append(nn.LeakyReLU(negative_slope=0.1))

    layers.append(nn.Linear(ffn_hidden_dim, output_dim))

    return nn.Sequential(*layers)


class TransformerAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.learnable_query = nn.Parameter(torch.randn(1, 1, transformer_dim))
        self.multihead_attention = nn.MultiheadAttention(embed_dim=transformer_dim, num_heads=1, batch_first=True) #TODO mehr heads testen

    def forward(self, sequence, mask, need_weights=False, attn_mask=None, average_attn_weights=True, is_causal=False):
        batch_size = sequence.size(0)
        q = self.learnable_query.expand(batch_size, -1, -1)
        if mask is not None:
            # Invert the mask for the transformer (True for padding, False for non-padding)
            mask = ~mask

        return self.multihead_attention(
            query=q, key=sequence, value=sequence, key_padding_mask=mask,
            need_weights=need_weights, attn_mask=attn_mask, average_attn_weights=average_attn_weights, is_causal=is_causal)


class TemporalEmbedding(nn.Module):
    def __init__(self, embedding_dim=temporal_embedding_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(3, embedding_dim)) # shape: (embedding_dim,)
        self.biases = nn.Parameter(torch.randn(3, embedding_dim)) # shape: (embedding_dim,)

    def forward(self, t):
        # t: (B, N, 3)
        # expand weights and biases to broadcast
        weights = self.weights.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, embedding_dim)
        biases = self.biases.unsqueeze(0).unsqueeze(0)    # (1, 1, 3, embedding_dim)

        # transform: (B, N, 3, embedding_dim)
        transformed = t.unsqueeze(-1) * weights + biases

        # split linear and sinusoidal parts
        first_dim = transformed[..., 0:1]   # (B, N, 3, 1) — linear part
        other_dims = torch.sin(transformed[..., 1:])  # (B, N, 3, embedding_dim-1)

        # concat along the last axis
        output = torch.cat([first_dim, other_dims], dim=-1)

        # flatten the 3 input dims
        return output.reshape(t.size(0), t.size(1), -1)  # (B, N, 3 * embedding_dim)


class TransformerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_ff_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_encoder_layers)

    def forward(self, x, mask):
        if mask is not None:
            mask = ~mask
        return self.transformer_encoder(x, src_key_padding_mask=mask)


class TrunkNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_embedding = TemporalEmbedding()
        self.ffn = ffn(input_dim=temporal_embedding_dim*3, output_dim=branch_trunk_output_dim, ffn_hidden_dim=ffn_hidden_dim)

    def forward(self, t_eval):
        temp_embedding = self.temporal_embedding(t_eval) # (batch_size, num_grid_points, embedding_dim)
        return self.ffn(temp_embedding) # (batch_size, num_grid_points, branch_trunk_output_dim)


class BranchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_embedding = TemporalEmbedding()
        self.transformer = TransformerNetwork()
        self.aggregator = TransformerAggregator()
        self.ffn = ffn(input_dim=transformer_dim, output_dim=branch_trunk_output_dim, num_hidden_layers=branch_net_num_hidden_layers, ffn_hidden_dim=ffn_hidden_dim)

    def forward(self, t_observed, y_observed, mask):
        sensor_embeddings = self.temporal_embedding(t_observed) # (batch_size, num_grid_points, embedding_dim)
        transformer_input = torch.cat((y_observed, sensor_embeddings), dim=-1) # (batch_size, num_grid_points, 1+ embedding_dim)
        transformer_output = self.transformer(transformer_input, mask=mask)  # (batch_size, num_grid_points, 1+ embedding_dim)
        transformer_output = self.aggregator(transformer_output,mask=mask)[0] #(batch_size, 1, 1+ embedding_dim)
        return self.ffn(transformer_output)


class CombinationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(branch_trunk_output_dim * 2, mlp_hidden_dim1),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim1, mlp_hidden_dim2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim2, 1)
        )

    def forward(self, branch_output, trunk_output):
        combined = torch.cat((branch_output.expand(-1, trunk_output.size(1), -1), trunk_output), dim=2)
        return self.mlp(combined).squeeze(-1)


class InterpolationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trunk_net = TrunkNet()
        self.branch_net = BranchNet()
        self.combination_net = CombinationNet()
        print_model_parameters(self)

    def forward(self, t_eval, t_observed, y_observed, mask):
        #Move to GPU
        t_eval = t_eval.to(self.device)  # (batch_size, num_grid_points, 3)
        y_observed = y_observed.to(self.device)  # (batch_size, num_grid_points)
        t_observed = t_observed.to(self.device)  # (batch_size, num_grid_points, 3)
        #mask = mask.to(self.device)  # (batch_size, num_grid_points) # is True for observed values and 0 for 0 padded values

        #Normalize
        #t_eval, t_observed, y_observed, tau_min, y_mean, tau_max, y_std = normalize_observations(t_eval, t_observed, y_observed, mask)
        norm_input, scales = in_context_models.normalize_input(torch.cat((t_observed, y_observed), dim=-1).transpose(0,1))
        norm_input = norm_input.transpose(0,1)
        t_eval_norm = in_context_models.normalize_to_scales(t_eval, scales)

        #Trunk net
        trunk_output = self.trunk_net(t_eval_norm)

        #Branch net
        branch_output = self.branch_net(norm_input[:,:,0:3], norm_input[:,:,3:4], mask)

        #Combine the outputs
        estimate = self.combination_net(branch_output, trunk_output)

        #Renormalize
        return in_context_models.renormalize(estimate.unsqueeze(-1), scales)

    def compute_loss(self, batch):
        datasets_in, gt_params, y_gt = batch
        mask = None
        out = self(datasets_in[:, :, 0:3], datasets_in[:, :, 0:3], datasets_in[:, :, 3:4], mask)

        return metrics.mse(out, y_gt)


