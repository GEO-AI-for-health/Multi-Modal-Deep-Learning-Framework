# -*- coding: utf-8 -*-
"""
"""

# ============ Import all libraries and configurations ============
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import datetime
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
import gc
from sklearn.metrics import precision_recall_fscore_support
import torch.optim.lr_scheduler as lr_scheduler
import math
from pytorch_tabnet.tab_network import TabNet
# ============ Import tabular processor ============
from tabular_processors import create_tabular_processor
from sklearn.metrics import roc_auc_score

# ============ Logger class ============
class Logger:
    """Logger that outputs to both console and file simultaneously"""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ============ All original configurations remain unchanged ============
SEQ_LEN = 30  # 20
DATA_DIR = "./processed_data_stress_new"
FEATURE_TYPE = "clip"

if FEATURE_TYPE == "clip":
    SAT_PT_DIR = "./clip/satellite_images_clip_pt"
elif FEATURE_TYPE == "sam3":
    SAT_PT_DIR = "./sam3/satellite_images_pt"
else:
    raise ValueError(f"Unknown FEATURE_TYPE: {FEATURE_TYPE}")

PHYSICAL_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
initial_lr = 5e-4
#min_lr = 1e-4

lr_decay_factor = 0.5
lr_plateau_epochs = 2  # 2
NUM_WORKERS = 0
EPOCHS = 50
LOG_FILE = "training_log_ablation_l2.txt"
HIDEEN_DIM = 128

# Adapter type selection: ["mobilenet", "cnn", "attention", "no_adapter"]
ADAPTER_TYPE = "mobilenet"
ADAPTER_PATCH_DIM = 64
ADAPTER_CLS_DIM = 16
ADAPTER_OUTPUT_DIM = ADAPTER_PATCH_DIM + ADAPTER_CLS_DIM
USE_CLS_IN_ADAPTER = True


CENTER_AWARE = True
CENTER_WEIGHT_INIT = "gaussian"


TABULAR_OUTPUT_DIM = 84
# Tabular processor type selection
TABULAR_PROCESSING = "grouped"  #linear", "mlp",   "raw","grouped"

# Temporal model selection: ["gru", "lstm", "mlp"]
TEMPORAL_MODEL = "gru"

# Cross-modal fusion method
CROSS_MODAL_FUSION = "concat"

LABEL_SMOOTHING = 0
# L2 regularization configuration
L2_REGULARIZATION = False  # Whether to enable L2 regularization
WEIGHT_DECAY = 0  # Optimizer weight_decay parameter (standard L2 regularization) #0.4
L2_LAMBDA = 0.00  # Weight of explicit L2 loss (0 means weight_decay only, >0 adds explicit L2 loss)
GAMMA = 5  # 5
# SVI as part of tabular features, individually controllable like other feature types
ABLATION_CONFIG = {
    "use_svi": True,
    "use_satellite": True,
    "use_gps": True,
    "use_env_exposure": True,
    "use_urban_function": True,
    "use_activity": True,
    "use_personal_ses": True
}

# Only 2 classes
NUM_CLASSES = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


RANDOM_SEED = 42
set_seed(RANDOM_SEED)

# def lr_lambda(epoch):
#     factor = lr_decay_factor ** (epoch // lr_plateau_epochs)
#     return max(factor, min_lr / initial_lr)

def get_feature_config(feature_type):
    """Satellite image feature configuration"""
    if feature_type == "clip":
        return {
            "sat_shape": (50, 768),
            "sat_has_cls": True,
            "feature_dim": 768,
            "sat_grid_size": 7
        }
    elif feature_type == "sam3":
        return {
            "sat_shape": (5184, 1024),
            "sat_has_cls": False,
            "feature_dim": 1024,
            "sat_grid_size": 72
        }
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


FEATURE_CONFIG = get_feature_config(FEATURE_TYPE)


# Compute L2 regularization loss
def compute_l2_loss(model):
    """
    Compute L2 norm of model parameters (explicit L2 loss)

    Args:
        model: PyTorch model

    Returns:
        l2_loss: L2 regularization loss value
    """
    l2_loss = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:  # Only apply L2 to weight parameters
            l2_loss += torch.sum(param ** 2)
    return l2_loss


# Label conversion function - keep only class 0 and class 2
def convert_labels(original_labels):
    """
    Convert original labels to binary classification
    """
    labels = np.zeros_like(original_labels)
    valid_mask = np.ones_like(original_labels, dtype=bool)

    for i, label in enumerate(original_labels):
        if label == 1:
            labels[i] = 0  # Low stress
        elif label == 2:
            valid_mask[i] = False  # Skip middle class
        elif label in [3, 4, 5, 6]:
            labels[i] = 1  # High stress
        else:
            valid_mask[i] = False  # Unknown label, also skip

    return labels, valid_mask


def get_class_description():
    return "2-class (Binary): [Low(1), High(3-6)]"


def init_weights(module):
    if isinstance(module, nn.Linear):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
        if fan_out <= 10:
            nn.init.xavier_normal_(module.weight, gain=0.5)
        else:
            nn.init.xavier_normal_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.02)
    elif isinstance(module, (nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                if 'LSTM' in module.__class__.__name__:
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)


# ============ Neural Decision Tree Head (true strong nonlinear version) ============
# class NeuralDecisionTreeHead(nn.Module):
#     """
#     Differentiable neural decision tree
#     Improvements:
#     1. Feature selection via Gumbel-Softmax (differentiable)
#     2. Threshold comparison via Sigmoid (soft decision)
#     3. Leaf nodes use simple nonlinear layers
#     """
#
#     def __init__(self, input_dim, num_classes, depth=3, temperature=1.0):
#         super().__init__()
#         self.input_dim = input_dim
#         self.num_classes = num_classes
#         self.depth = depth
#         self.temperature = temperature
#
#         self.num_leaves = 2 ** depth
#         self.num_internal = self.num_leaves - 1
#
#         # Approach A: Each node computes weighted combination of all features (simple and reliable)
#         self.split_weights = nn.ParameterList([
#             nn.Parameter(torch.randn(input_dim) / np.sqrt(input_dim))
#             for _ in range(self.num_internal)
#         ])
#
#         self.split_thresholds = nn.Parameter(
#             torch.randn(self.num_internal)
#         )
#
#         # Leaf nodes: simple linear + activation
#         self.leaf_networks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(input_dim, 32),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(32, num_classes)
#             ) for _ in range(self.num_leaves)
#         ])
#
#     def forward(self, x):
#         """
#         x: [B, input_dim]
#         """
#         batch_size = x.shape[0]
#
#         # Compute probability of reaching each leaf
#         leaf_probs = self._compute_leaf_probabilities(x)  # [B, num_leaves]
#
#         # Output of each leaf
#         leaf_outputs = torch.stack([
#             leaf_net(x) for leaf_net in self.leaf_networks
#         ], dim=1)  # [B, num_leaves, num_classes]
#
#         # Weighted sum
#         logits = (leaf_outputs * leaf_probs.unsqueeze(-1)).sum(dim=1)
#
#         return logits
#
#     def _compute_leaf_probabilities(self, x):
#         """
#         Hierarchically compute routing probabilities for each node
#         """
#         batch_size = x.shape[0]
#         # Initial: all samples at root node (probability=1)
#         node_probs = [torch.ones(batch_size, device=x.device)]
#
#         for level in range(self.depth):
#             new_probs = []
#             for node_idx in range(2 ** level):
#                 internal_idx = (2 ** level - 1) + node_idx
#
#                 current_prob = node_probs[node_idx]
#
#                 # Feature weighted combination
#                 weighted_sum = (x * self.split_weights[internal_idx]).sum(dim=1)  # [B]
#
#                 # Compare with threshold (Sigmoid soft decision)
#                 split_value = weighted_sum - self.split_thresholds[internal_idx]
#                 split_prob = torch.sigmoid(split_value / self.temperature)
#
#                 # Left and right child node probabilities
#                 left_prob = current_prob * (1 - split_prob)
#                 right_prob = current_prob * split_prob
#
#                 new_probs.extend([left_prob, right_prob])
#
#             node_probs = new_probs
#
#         # Last level is the leaf nodes
#         return torch.stack(node_probs, dim=1)  # [B, num_leaves]


class SparseNeuralDecisionTreeHead(nn.Module):
    """
    Sparse feature selection neural decision tree classification head

    Core features:
    1. Each node uses Gumbel-Softmax to select top-k features (differentiable)
    2. Soft threshold decision (Sigmoid)
    3. Leaf nodes use nonlinear MLP
    4. Supports feature importance visualization
    """

    def __init__(self, input_dim, num_classes, depth=3,
                 num_features_per_node=5, temperature=1.0,
                 leaf_hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.depth = depth
        self.num_features_per_node = min(num_features_per_node, input_dim)
        self.temperature = temperature

        self.num_leaves = 2 ** depth
        self.num_internal = self.num_leaves - 1

        self.feature_logits = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim) * 0.1)
            for _ in range(self.num_internal)
        ])

        self.split_thresholds = nn.Parameter(
            torch.zeros(self.num_internal)
        )

        self.feature_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim) / np.sqrt(input_dim))
            for _ in range(self.num_internal)
        ])

        self.leaf_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, leaf_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(leaf_hidden_dim, num_classes)
            ) for _ in range(self.num_leaves)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.leaf_networks:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x, return_routing=False):
        batch_size = x.shape[0]

        leaf_probs, routing_info = self._compute_leaf_probabilities(x)

        leaf_outputs = torch.stack([
            leaf_net(x) for leaf_net in self.leaf_networks
        ], dim=1)

        logits = (leaf_outputs * leaf_probs.unsqueeze(-1)).sum(dim=1)

        if return_routing:
            routing_info['leaf_probs'] = leaf_probs
            routing_info['leaf_outputs'] = leaf_outputs
            return logits, routing_info

        return logits

    def _compute_leaf_probabilities(self, x):
        batch_size = x.shape[0]
        device = x.device

        routing_info = {
            'feature_selections': [],
            'split_probs': [],
            'feature_importance': []
        }

        node_probs = [torch.ones(batch_size, device=device)]

        for level in range(self.depth):
            new_probs = []

            for node_idx in range(2 ** level):
                internal_idx = (2 ** level - 1) + node_idx

                if internal_idx >= self.num_internal:
                    break

                current_prob = node_probs[node_idx]

                split_prob, node_routing = self._compute_node_split(
                    x, internal_idx
                )

                routing_info['feature_selections'].append(node_routing['selected_features'])
                routing_info['split_probs'].append(split_prob)
                routing_info['feature_importance'].append(node_routing['importance'])

                left_prob = current_prob * (1 - split_prob)
                right_prob = current_prob * split_prob

                new_probs.extend([left_prob, right_prob])

            node_probs = new_probs

        leaf_probs = torch.stack(node_probs, dim=1)

        return leaf_probs, routing_info

    def _compute_node_split(self, x, node_idx):
        batch_size = x.shape[0]

        feature_importance = F.gumbel_softmax(
            self.feature_logits[node_idx].unsqueeze(0).expand(batch_size, -1),
            tau=self.temperature,
            hard=False,
            dim=-1
        )

        selected_features = x * feature_importance

        weighted_features = selected_features * self.feature_weights[node_idx]

        aggregated_value = weighted_features.sum(dim=1)

        split_value = aggregated_value - self.split_thresholds[node_idx]
        split_prob = torch.sigmoid(split_value / self.temperature)

        routing_info = {
            'selected_features': feature_importance.mean(dim=0),
            'importance': feature_importance,
            'aggregated_value': aggregated_value
        }

        return split_prob, routing_info

    def get_feature_importance(self):
        importance = []
        for node_idx in range(self.num_internal):
            node_importance = F.softmax(self.feature_logits[node_idx], dim=0)
            importance.append(node_importance.detach().cpu())

        return torch.stack(importance)

    def visualize_tree_structure(self, feature_names=None):
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(self.input_dim)]

        importance_matrix = self.get_feature_importance()

        print("=" * 60)
        print("Neural Decision Tree Structure")
        print("=" * 60)

        for level in range(self.depth):
            print(f"\n{'Level'} {level}:")
            print("-" * 60)

            for node_idx in range(2 ** level):
                internal_idx = (2 ** level - 1) + node_idx

                if internal_idx >= self.num_internal:
                    break

                node_imp = importance_matrix[internal_idx]
                top_k_idx = torch.topk(node_imp, k=min(5, self.num_features_per_node)).indices
                top_k_values = node_imp[top_k_idx]

                threshold = self.split_thresholds[internal_idx].item()

                print(f"  Node {internal_idx} (Threshold: {threshold:.3f}):")
                for i, (idx, val) in enumerate(zip(top_k_idx, top_k_values)):
                    print(f"    {i + 1}. {feature_names[idx]:20s}: {val:.4f}")

        print("=" * 60)

    def get_routing_stats(self, dataloader):
        self.eval()
        leaf_counts = torch.zeros(self.num_leaves)
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(next(self.parameters()).device)
                leaf_probs, _ = self._compute_leaf_probabilities(x)

                hard_routing = leaf_probs.argmax(dim=1)
                for leaf_idx in hard_routing:
                    leaf_counts[leaf_idx] += 1

                total_samples += x.shape[0]

        stats = {
            'leaf_counts': leaf_counts.numpy(),
            'leaf_ratios': (leaf_counts / total_samples).numpy(),
            'total_samples': total_samples
        }

        return stats


# ============ Focal Loss (unchanged) ============
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.shape[-1]
        if self.label_smoothing > 0:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_smooth = targets_one_hot * (1 - self.label_smoothing) + \
                             self.label_smoothing / num_classes
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(targets_smooth * log_probs).sum(dim=-1)
            if self.alpha is not None:
                alpha_t = (targets_smooth * self.alpha.unsqueeze(0)).sum(dim=-1)
                ce_loss = alpha_t * ce_loss
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============ Dataset - Filter middle class ============
class NPZDataset(Dataset):
    def __init__(self, npz_path, tabular_min=None, tabular_max=None,
                 class_weights=None, feature_config=None):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing: {npz_path}")
        self.feature_config = feature_config or FEATURE_CONFIG
        print(f"Loading {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)

        # First convert labels and get valid sample mask
        original_labels = data['label'].astype(np.int64)
        converted_labels, valid_mask = convert_labels(original_labels)

        # Keep only valid samples
        self.tabular = data['tabular'][valid_mask]
        self.gps_x = data['gps_x'][valid_mask]
        self.gps_y = data['gps_y'][valid_mask]
        self.mask = data['mask'][valid_mask]
        self.label = converted_labels[valid_mask]
        self.img_names = data['img_names'][valid_mask]

        self.num_classes = 2
        self.tabular_dim = self.tabular.shape[2]

        if tabular_min is None or tabular_max is None:
            self.tabular_min = np.min(self.tabular, axis=(0, 1))
            self.tabular_max = np.max(self.tabular, axis=(0, 1))
        else:
            self.tabular_min = tabular_min
            self.tabular_max = tabular_max

        self.class_counts = np.bincount(self.label, minlength=2)
        if class_weights is None:
            beta = 0.999
            effective_num = 1.0 - np.power(beta, self.class_counts)
            self.class_weights = (1.0 - beta) / (effective_num + 1e-8)
            self.class_weights = self.class_weights / self.class_weights.sum() * 2
        else:
            self.class_weights = class_weights

        print(
            f"  Loaded {len(self.label)} samples (filtered from {len(original_labels)}). Tabular Dim: {self.tabular_dim}")
        print(f"   Feature Type: {FEATURE_TYPE.upper()}")
        print(f"   Classification: {get_class_description()}")
        print(f"   Tabular Features Layout:")
        print(f"     - Columns 0-7:   SVI (8 features)")
        print(f"     - Columns 8-11:  Environmental Exposure (4 features)")
        print(f"     - Columns 12-20: Urban Function/POI (9 features)")
        print(f"     - Columns 21-25: Activity Context (5 features)")
        print(f"     - Columns 26-29: Personal SES (4 features)")
        print(f"   Class Counts: {self.class_counts} (Class 0: Low, Class 1: High)")
        print(f"   Class Distribution: {(self.class_counts / len(self.label) * 100).astype(int)}%")
        print(f"   Class Weights: {np.round(self.class_weights, 4)}")
        print(f"   Filtered out {(~valid_mask).sum()} middle-class samples")

    def __len__(self):
        return len(self.label)

    def _get_pt_path(self, folder, fname):
        if not fname or str(fname) in ["nan", "", "None"]:
            return None
        fname = str(fname)
        name_core = os.path.splitext(fname)[0]
        return os.path.join(folder, name_core + ".pt")

    def __getitem__(self, idx):
        tab = self.tabular[idx]
        tab_normalized = tab
        tab = torch.tensor(tab_normalized).float()

        gx = torch.tensor(self.gps_x[idx] / 1.0).float()
        gy = torch.tensor(self.gps_y[idx] / 1.0).float()
        msk = torch.tensor(self.mask[idx]).long()
        lbl = torch.tensor(self.label[idx]).long()
        names = self.img_names[idx]

        sat_feats = []
        sat_shape = self.feature_config["sat_shape"]

        for i in range(SEQ_LEN):
            fname = names[i]
            s_path = self._get_pt_path(SAT_PT_DIR, fname)
            try:
                if s_path and os.path.exists(s_path):
                    loaded = torch.load(s_path, map_location="cpu")
                    if loaded.shape == sat_shape:
                        sat_feats.append(loaded.to(dtype))
                    else:
                        sat_feats.append(torch.zeros(*sat_shape, dtype=dtype))
                else:
                    sat_feats.append(torch.zeros(*sat_shape, dtype=dtype))
            except:
                sat_feats.append(torch.zeros(*sat_shape, dtype=dtype))

        return {
            'sat_feats': torch.stack(sat_feats),
            'tabular': tab,
            'gps_x': gx,
            'gps_y': gy,
            'mask': msk,
            'label': lbl
        }


def custom_collate(batch):
    return torch.utils.data.default_collate(batch)


# ============ CNN Adapter ============
class CNNAdapter(nn.Module):
    """Simple CNN adapter"""

    def __init__(self, input_dim=768, output_dim=80, patch_dim=64, cls_dim=16,
                 grid_size=7, has_cls=True, use_cls_in_adapter=True,
                 center_aware=False, center_weight_init="gaussian"):
        super().__init__()
        self.grid_size = grid_size
        self.has_cls = has_cls
        self.use_cls_in_adapter = use_cls_in_adapter and has_cls
        self.patch_dim = patch_dim
        self.cls_dim = cls_dim
        self.center_aware = center_aware

        if self.center_aware:
            self._init_spatial_weights(center_weight_init)
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(input_dim, 1, kernel_size=1),
                nn.Sigmoid()
            )

        self.conv1 = nn.Conv2d(input_dim, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if self.use_cls_in_adapter:
            self.cls_proj = nn.Sequential(nn.Linear(input_dim, cls_dim))
            self.patch_proj = nn.Sequential(
                nn.Linear(64, patch_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(patch_dim * 2, patch_dim)
            )
        else:
            self.output_proj = nn.Linear(64, output_dim)

    def _init_spatial_weights(self, init_type):
        y, x = torch.meshgrid(
            torch.arange(self.grid_size, dtype=torch.float32),
            torch.arange(self.grid_size, dtype=torch.float32),
            indexing='ij'
        )
        center = (self.grid_size - 1) / 2.0
        dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
        if init_type == "gaussian":
            sigma = self.grid_size / 3.0
            weights = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
        elif init_type == "linear":
            max_dist = torch.sqrt(torch.tensor(2.0)) * center
            weights = 1.0 - (dist / max_dist) * 0.7
        else:
            weights = torch.ones_like(dist)
        self.register_parameter(
            'spatial_weights',
            nn.Parameter(weights.unsqueeze(0).unsqueeze(0))
        )

    def forward(self, features):
        B = features.shape[0]
        if self.use_cls_in_adapter and self.has_cls:
            cls_token = features[:, 0]
            patches = features[:, 1:]
        elif self.has_cls:
            cls_token = None
            patches = features[:, 1:]
        else:
            cls_token = None
            patches = features

        spatial_feat = patches.reshape(B, self.grid_size, self.grid_size, -1)
        spatial_feat = spatial_feat.permute(0, 3, 1, 2)

        if self.center_aware:
            spatial_weights = self.spatial_weights.to(spatial_feat.device)
            weighted_feat = spatial_feat * spatial_weights
            gate = self.spatial_gate(spatial_feat)
            weighted_feat = weighted_feat * gate
        else:
            weighted_feat = spatial_feat

        x = F.relu(self.bn1(self.conv1(weighted_feat)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        spatial_output = self.gap(x).squeeze(-1).squeeze(-1)

        if self.use_cls_in_adapter and cls_token is not None:
            cls_feat = self.cls_proj(cls_token)
            patch_feat = self.patch_proj(spatial_output)
            output = torch.cat([cls_feat, patch_feat], dim=-1)
        else:
            output = self.output_proj(spatial_output)
        return output


# ============ Multi-Head Attention Adapter ============
class AttentionAdapter(nn.Module):
    """Multi-head self-attention adapter"""

    def __init__(self, input_dim=768, output_dim=80, patch_dim=64, cls_dim=16,
                 grid_size=7, has_cls=True, use_cls_in_adapter=True,
                 center_aware=False, center_weight_init="gaussian",
                 num_heads=8):
        super().__init__()
        self.grid_size = grid_size
        self.has_cls = has_cls
        self.use_cls_in_adapter = use_cls_in_adapter and has_cls
        self.patch_dim = patch_dim
        self.cls_dim = cls_dim
        self.center_aware = center_aware
        self.num_heads = num_heads

        if self.center_aware:
            self._init_spatial_weights(center_weight_init)

        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim)
        )

        if self.use_cls_in_adapter:
            self.cls_proj = nn.Sequential(nn.Linear(input_dim, cls_dim))
            self.patch_proj = nn.Sequential(
                nn.Linear(input_dim, patch_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(patch_dim * 2, patch_dim)
            )
        else:
            self.output_proj = nn.Linear(input_dim, output_dim)

    def _init_spatial_weights(self, init_type):
        num_patches = self.grid_size ** 2
        y, x = torch.meshgrid(
            torch.arange(self.grid_size, dtype=torch.float32),
            torch.arange(self.grid_size, dtype=torch.float32),
            indexing='ij'
        )
        center = (self.grid_size - 1) / 2.0
        dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2).flatten()

        if init_type == "gaussian":
            sigma = self.grid_size / 3.0
            weights = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
        elif init_type == "linear":
            max_dist = torch.sqrt(torch.tensor(2.0)) * center
            weights = 1.0 - (dist / max_dist) * 0.7
        else:
            weights = torch.ones_like(dist)

        self.register_buffer('spatial_weights', weights)

    def forward(self, features):
        B = features.shape[0]

        if self.has_cls:
            cls_token = features[:, 0:1]
            patches = features[:, 1:]
        else:
            cls_token = None
            patches = features

        if self.center_aware and self.has_cls:
            spatial_weights = self.spatial_weights.to(patches.device)
            patches = patches * spatial_weights.unsqueeze(0).unsqueeze(-1)

        if cls_token is not None:
            x = torch.cat([cls_token, patches], dim=1)
        else:
            x = patches

        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        if self.use_cls_in_adapter and cls_token is not None:
            cls_feat = self.cls_proj(x[:, 0])
            patch_feat = x[:, 1:].mean(dim=1)
            patch_feat = self.patch_proj(patch_feat)
            output = torch.cat([cls_feat, patch_feat], dim=-1)
        else:
            output = x.mean(dim=1)
            output = self.output_proj(output)

        return output


# ============ No Adapter (direct pooling) ============
class NoAdapter(nn.Module):
    """No adapter used; directly applies global average pooling to features"""

    def __init__(self, input_dim=768, output_dim=80, patch_dim=64, cls_dim=16,
                 grid_size=7, has_cls=True, use_cls_in_adapter=True,
                 center_aware=False, center_weight_init="gaussian"):
        super().__init__()
        self.has_cls = has_cls
        self.use_cls_in_adapter = use_cls_in_adapter and has_cls
        self.cls_dim = cls_dim
        self.patch_dim = patch_dim

        if self.use_cls_in_adapter:
            self.cls_proj = nn.Linear(input_dim, cls_dim)
            self.patch_proj = nn.Linear(input_dim, patch_dim)
        else:
            self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        B = features.shape[0]

        if self.use_cls_in_adapter and self.has_cls:
            cls_token = features[:, 0]
            patches = features[:, 1:]

            cls_feat = self.cls_proj(cls_token)
            patch_feat = patches.mean(dim=1)
            patch_feat = self.patch_proj(patch_feat)
            output = torch.cat([cls_feat, patch_feat], dim=-1)
        else:
            pooled = features.mean(dim=1)
            output = self.output_proj(pooled)

        return output


# ============ MobileNetAdapter (unchanged) ============
class MobileNetAdapter(nn.Module):
    """Center-aware MobileNet adapter"""

    def __init__(self, input_dim=768, output_dim=80, patch_dim=64, cls_dim=16,
                 grid_size=7, has_cls=True, use_cls_in_adapter=True,
                 center_aware=False, center_weight_init="gaussian"):
        super().__init__()
        self.grid_size = grid_size
        self.has_cls = has_cls
        self.use_cls_in_adapter = use_cls_in_adapter and has_cls
        self.patch_dim = patch_dim
        self.cls_dim = cls_dim
        self.center_aware = center_aware

        if self.center_aware:
            self._init_spatial_weights(center_weight_init)
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(input_dim, 1, kernel_size=1),
                nn.Sigmoid()
            )

        self.depthwise1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim)
        self.pointwise1 = nn.Conv2d(input_dim, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.depthwise2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256)
        self.pointwise2 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.depthwise3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128)
        self.pointwise3 = nn.Conv2d(128, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if self.use_cls_in_adapter:
            self.cls_proj = nn.Sequential(nn.Linear(input_dim, cls_dim))
            self.patch_proj = nn.Sequential(
                nn.Linear(64, patch_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(patch_dim * 2, patch_dim)
            )
        else:
            self.output_proj = nn.Linear(64, output_dim)

    def _init_spatial_weights(self, init_type):
        y, x = torch.meshgrid(
            torch.arange(self.grid_size, dtype=torch.float32),
            torch.arange(self.grid_size, dtype=torch.float32),
            indexing='ij'
        )
        center = (self.grid_size - 1) / 2.0
        dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
        if init_type == "gaussian":
            sigma = self.grid_size / 3.0
            weights = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
        elif init_type == "linear":
            max_dist = torch.sqrt(torch.tensor(2.0)) * center
            weights = 1.0 - (dist / max_dist) * 0.7
        else:
            weights = torch.ones_like(dist)
        self.register_parameter(
            'spatial_weights',
            nn.Parameter(weights.unsqueeze(0).unsqueeze(0))
        )

    def forward(self, features):
        B = features.shape[0]
        if self.use_cls_in_adapter and self.has_cls:
            cls_token = features[:, 0]
            patches = features[:, 1:]
        elif self.has_cls:
            cls_token = None
            patches = features[:, 1:]
        else:
            cls_token = None
            patches = features

        spatial_feat = patches.reshape(B, self.grid_size, self.grid_size, -1)
        spatial_feat = spatial_feat.permute(0, 3, 1, 2)

        if self.center_aware:
            spatial_weights = self.spatial_weights.to(spatial_feat.device)
            weighted_feat = spatial_feat * spatial_weights
            gate = self.spatial_gate(spatial_feat)
            weighted_feat = weighted_feat * gate
        else:
            weighted_feat = spatial_feat

        x = F.relu(self.bn1(self.pointwise1(self.depthwise1(weighted_feat))))
        x = F.relu(self.bn2(self.pointwise2(self.depthwise2(x))))
        x = F.relu(self.bn3(self.pointwise3(self.depthwise3(x))))
        spatial_output = self.gap(x).squeeze(-1).squeeze(-1)

        if self.use_cls_in_adapter and cls_token is not None:
            cls_feat = self.cls_proj(cls_token)
            patch_feat = self.patch_proj(spatial_output)
            output = torch.cat([cls_feat, patch_feat], dim=-1)
        else:
            output = self.output_proj(spatial_output)
        return output


# ============ HierarchicalGating ============
class HierarchicalGating(nn.Module):
    def __init__(self, modality_dims, output_dim, dropout=0.3):
        super().__init__()
        self.intra_modal_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, output_dim), nn.Tanh())
            for dim in modality_dims
        ])
        self.intra_modal_gating = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, output_dim), nn.Sigmoid())
            for dim in modality_dims
        ])
        total_dim = output_dim * len(modality_dims)
        self.inter_modal_gate = nn.Sequential(
            nn.Linear(total_dim, len(modality_dims)),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, modality_list, return_details=False):
        intra_gated = []
        for feat, h_layer, g_layer in zip(
                modality_list, self.intra_modal_gates, self.intra_modal_gating
        ):
            h = h_layer(feat)
            g = g_layer(feat)
            z = h * g
            intra_gated.append(z)

        stacked = torch.stack(intra_gated, dim=2)
        concat_for_weight = torch.cat(intra_gated, dim=-1)
        modal_weights = self.inter_modal_gate(concat_for_weight)
        modal_weights_expanded = modal_weights.unsqueeze(-1)
        fused = (stacked * modal_weights_expanded).sum(dim=2)
        fused = self.dropout(fused)

        if return_details:
            return fused, {"modal_weights": modal_weights, "intra_gated": stacked}
        return fused


class ContextAwareGating(nn.Module):
    def __init__(self, modality_dims, output_dim, context_dim=64, dropout=0.3):
        super().__init__()
        self.context_dim = context_dim
        total_dim = sum(modality_dims)
        self.context_encoder = nn.Sequential(
            nn.Linear(total_dim, context_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, context_dim),
            nn.ReLU()
        )
        self.h_transforms = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in modality_dims
        ])
        self.context_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim + context_dim, output_dim),
                nn.Sigmoid()
            ) for dim in modality_dims
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, modality_list, return_details=False):
        concat_all = torch.cat(modality_list, dim=-1)
        context = self.context_encoder(concat_all)
        gated_features = []
        for feat, h_transform, gate in zip(
                modality_list, self.h_transforms, self.context_gates
        ):
            h = h_transform(feat)
            feat_with_context = torch.cat([feat, context], dim=-1)
            g = gate(feat_with_context)
            z = h * g
            gated_features.append(z)
        fused = sum(gated_features)
        fused = self.dropout(fused)
        if return_details:
            return fused, {"gated_features": gated_features}
        return fused


# ============ Other basic modules ============
class CoordsRFFEncoder(nn.Module):
    def __init__(self, rff_components=32, output_dim=16, scales=[2.0, 4.0, 8.0, 16.0]):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.rff_weights_list, self.rff_offsets_list = [], []
        for scale in scales:
            rbf = RBFSampler(gamma=1.0 / (2 * scale ** 2), n_components=rff_components, random_state=42)
            rbf.fit(np.zeros((1, 2)))
            self.rff_weights_list.append(torch.from_numpy(rbf.random_weights_).float())
            self.rff_offsets_list.append(torch.from_numpy(rbf.random_offset_).float())
        self.register_buffer('rff_weights', torch.stack(self.rff_weights_list))
        self.register_buffer('rff_offsets', torch.stack(self.rff_offsets_list))
        input_dim = rff_components * self.num_scales
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU()
        )

    def forward(self, x, y):
        x_flat, y_flat = x.reshape(-1), y.reshape(-1)
        coords = torch.stack([x_flat, y_flat], dim=-1)
        rff_weights = self.rff_weights.to(coords.device)
        rff_offsets = self.rff_offsets.to(coords.device)
        feats = []
        for s in range(self.num_scales):
            proj = torch.matmul(coords, rff_weights[s]) + rff_offsets[s]
            feats.append(torch.cos(proj) * np.sqrt(2.0))
        encoded = self.mlp(torch.cat(feats, dim=-1))
        return encoded.view(x.shape[0], x.shape[1], -1) if len(x.shape) > 1 else encoded


# ============ LSTM Temporal Encoder ============
class LSTMTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.output_dim = hidden_dim

    def forward(self, x, mask):
        lstm_out, _ = self.lstm(x)
        mask_expanded = mask.unsqueeze(-1).float()
        masked_out = lstm_out * mask_expanded
        pooled = masked_out.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        return pooled


# ============ MLP Temporal Encoder ============
class MLPTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
                current_dim = hidden_dim * 2
            elif i == num_layers - 1:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
            else:
                layers.extend([
                    nn.Linear(current_dim, current_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, x, mask):
        mask_expanded = mask.unsqueeze(-1).float()
        masked_x = x * mask_expanded
        pooled = masked_x.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)

        output = self.mlp(pooled)
        return output


# ============ GRU Temporal Encoder (unchanged) ============
class GRUTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=False)
        self.output_dim = hidden_dim

    def forward(self, x, mask):
        gru_out, _ = self.gru(x)
        mask_expanded = mask.unsqueeze(-1).float()
        masked_out = gru_out * mask_expanded
        pooled = masked_out.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        #pooled = masked_out.max(dim=1)[0]  # Use max pooling
        return pooled


# ============ Raw Tabular Processor (identity transform) ============
class RawTabularProcessor(nn.Module):
    """
    No transformation applied to tabular features; returns raw features directly
    """

    def __init__(self, input_dim, output_dim=None):
        super().__init__()
        self.output_dim = input_dim

    def forward(self, x):
        return x


# ============ FullStackPredictor - Supports multiple adapters and temporal models ============
class FullStackPredictor(nn.Module):
    def __init__(self, tabular_dim, hidden_dim=HIDEEN_DIM, num_classes=2,
                 adapter_type="mobilenet", temporal_model="gru",
                 ablation_config=None, feature_config=None,
                 adapter_output_dim=80, adapter_patch_dim=64,
                 adapter_cls_dim=16, use_cls_in_adapter=True,
                 cross_modal_fusion="concat",
                 tabular_output_dim=32,
                 tabular_processing="linear",
                 center_aware=False, center_weight_init="gaussian"):
        super().__init__()
        self.adapter_output_dim = adapter_output_dim
        self.adapter_patch_dim = adapter_patch_dim
        self.adapter_cls_dim = adapter_cls_dim
        self.gps_dim = 16
        self.temporal_model_type = temporal_model
        self.num_classes = num_classes
        self.adapter_type = adapter_type
        self.cross_modal_fusion_type = cross_modal_fusion
        self.tabular_processing = tabular_processing
        self.feature_config = feature_config or FEATURE_CONFIG
        feature_dim = self.feature_config["feature_dim"]

        if ablation_config is None:
            ablation_config = {k: True for k in [
                "use_svi", "use_satellite", "use_gps",
                "use_env_exposure", "use_urban_function",
                "use_activity", "use_personal_ses"
            ]}
        self.ablation = ablation_config

        adapter_map = {
            "mobilenet": MobileNetAdapter,
            "cnn": CNNAdapter,
            "attention": AttentionAdapter,
            "no_adapter": NoAdapter
        }

        if adapter_type not in adapter_map:
            raise ValueError(f"Unknown adapter_type: {adapter_type}. "
                             f"Choose from {list(adapter_map.keys())}")

        adapter_class = adapter_map[adapter_type]
        sat_adapter_kwargs = {
            "center_aware": center_aware,
            "center_weight_init": center_weight_init
        }

        if self.ablation["use_satellite"]:
            self.sat_adapter = adapter_class(
                feature_dim, adapter_output_dim,
                patch_dim=adapter_patch_dim,
                cls_dim=adapter_cls_dim,
                grid_size=self.feature_config["sat_grid_size"],
                has_cls=self.feature_config["sat_has_cls"],
                use_cls_in_adapter=use_cls_in_adapter,
                **sat_adapter_kwargs
            )

        if self.ablation["use_gps"]:
            self.gps_enc = CoordsRFFEncoder(output_dim=self.gps_dim)

        # Get original slices (for extracting features from raw tabular data)
        self.tabular_slices = self._get_tabular_slices(tabular_dim)
        total_tabular_dim = sum(end - start for start, end in self.tabular_slices.values())

        if total_tabular_dim > 0:
            if tabular_processing == "raw":
                self.tabular_output_dim = total_tabular_dim
                self.tab_processor = RawTabularProcessor(total_tabular_dim)
            else:
                self.tabular_output_dim = tabular_output_dim

                if tabular_processing == "grouped":
                    # FIX: Use recomputed processor_slices instead of original tabular_slices,
                    # because _extract_tabular_features concatenates enabled feature groups into
                    # a new tensor with column indices starting from 0 consecutively,
                    # which differs from the column indices of the original tabular data
                    processor_slices = self._get_processor_slices()
                    self.tab_processor = create_tabular_processor(
                        'grouped',
                        total_tabular_dim,
                        self.tabular_output_dim,
                        feature_groups=processor_slices
                    )
                elif tabular_processing in ["linear", "mlp", "film", "attention", "glu",
                                            "tokenizer", "fm", "dcn", "resnet", "nonlinear_preserving",
                                            "neural_tree", "tree_mlp"]:
                    self.tab_processor = create_tabular_processor(
                        tabular_processing,
                        total_tabular_dim,
                        self.tabular_output_dim
                    )
                else:
                    raise ValueError(f"Unknown tabular_processing: {tabular_processing}")
        else:
            self.tabular_output_dim = 0

        modality_dims = []
        if self.ablation["use_satellite"]:
            modality_dims.append(adapter_output_dim)
        if self.ablation["use_gps"]:
            modality_dims.append(self.gps_dim)
        if total_tabular_dim > 0:
            modality_dims.append(self.tabular_output_dim)

        self.modality_names = []
        if self.ablation["use_satellite"]:
            self.modality_names.append("satellite")
        if self.ablation["use_gps"]:
            self.modality_names.append("gps")
        if total_tabular_dim > 0:
            self.modality_names.append("tabular")

        if cross_modal_fusion == "concat":
            fusion_dim = sum(modality_dims)
            self.cross_modal_fusion = None
        elif cross_modal_fusion == "hierarchical_gating":
            fusion_dim = adapter_output_dim if self.ablation["use_satellite"] else self.tabular_output_dim
            self.cross_modal_fusion = HierarchicalGating(
                modality_dims=modality_dims,
                output_dim=fusion_dim,
                dropout=0.3
            )
        elif cross_modal_fusion == "context_aware_gating":
            fusion_dim = adapter_output_dim if self.ablation["use_satellite"] else self.tabular_output_dim
            self.cross_modal_fusion = ContextAwareGating(
                modality_dims=modality_dims,
                output_dim=fusion_dim,
                context_dim=64,
                dropout=0.3
            )
        else:
            raise ValueError(f"Unknown cross_modal_fusion: {cross_modal_fusion}")

        if temporal_model == "gru":
            self.temporal_encoder = GRUTemporalEncoder(fusion_dim, hidden_dim, num_layers=2)
        elif temporal_model == "lstm":
            self.temporal_encoder = LSTMTemporalEncoder(fusion_dim, hidden_dim, num_layers=2)
        elif temporal_model == "mlp":
            self.temporal_encoder = MLPTemporalEncoder(fusion_dim, hidden_dim, num_layers=2)
        else:
            raise ValueError(f"Unknown temporal_model: {temporal_model}. "
                             f"Choose from ['gru', 'lstm', 'mlp']")

        self.head = nn.Sequential(
            nn.Linear(self.temporal_encoder.output_dim, num_classes)
        )

    def _get_tabular_slices(self, tabular_dim):
        """Return column indices for each feature group in the original tabular data
        (used for extracting features from raw data)"""
        slices = {}

        if self.ablation["use_svi"]:
            slices["svi"] = (0, 8)
        if self.ablation["use_env_exposure"]:
            slices["env"] = (8, 12)
        if self.ablation["use_urban_function"]:
            slices["urban"] = (12, 21)
        if self.ablation["use_activity"]:
            slices["activity"] = (21, 26)
        if self.ablation["use_personal_ses"]:
            slices["ses"] = (26, 30)

        return slices

    def _get_processor_slices(self):
        """
        Recompute consecutive slices starting from 0 based on enabled feature groups.
        Used for GroupedTabularProcessor, because it receives the tensor concatenated
        by _extract_tabular_features, where column indices start from 0 consecutively.

        Example: if use_svi=False, tabular_slices = {"env": (8,12), "urban": (12,21), ...}
        After concatenation by _extract_tabular_features, tensor dim = [B, S, 4+9+...]
        processor_slices should be {"env": (0,4), "urban": (4,13), ...}
        """
        processor_slices = {}
        offset = 0
        for name, (start, end) in self.tabular_slices.items():
            dim = end - start
            processor_slices[name] = (offset, offset + dim)
            offset += dim
        return processor_slices

    def _extract_tabular_features(self, tabular):
        features = []
        for start, end in self.tabular_slices.values():
            features.append(tabular[:, :, start:end])
        if features:
            return torch.cat(features, dim=-1)
        else:
            return None

    def forward(self, batch, return_details=False):
        B, S = batch['mask'].shape
        model_device = next(self.parameters()).device
        feature_list = []

        if self.ablation["use_satellite"]:
            sat_in = batch['sat_feats'].to(model_device)
            B, S, SatSeqLen, SatDim = sat_in.shape
            sat_flat = sat_in.view(B * S, SatSeqLen, SatDim)
            sat_enc = self.sat_adapter(sat_flat)
            sat_emb = sat_enc.view(B, S, -1)
            feature_list.append(sat_emb)

        if self.ablation["use_gps"]:
            gps_emb = self.gps_enc(
                batch['gps_x'].to(model_device),
                batch['gps_y'].to(model_device)
            )
            if gps_emb.dim() == 2:
                gps_emb = gps_emb.unsqueeze(1).expand(-1, S, -1)
            feature_list.append(gps_emb)

        # ========== Enhanced nonlinearity while preserving temporal structure ==========
        tab_features = self._extract_tabular_features(batch['tabular'].to(model_device))
        if tab_features is not None:
            if self.tab_processor is not None:
                # [B, S, 50] -> tabular processor -> [B, S, 84]
                tab_emb = self.tab_processor(tab_features)

                # Enhanced nonlinearity: apply additional nonlinear transform at each timestep
                # This strengthens nonlinear patterns so they are not fully smoothed by GRU
                if hasattr(self.tab_processor, 'post_transform'):
                    # Apply enhancement to temporal features
                    B, S, D = tab_emb.shape
                    tab_emb_flat = tab_emb.reshape(B * S, D)
                    tab_emb_enhanced = self.tab_processor.post_transform(tab_emb_flat)
                    tab_emb = tab_emb_enhanced.reshape(B, S, D)
            else:
                tab_emb = tab_features
            feature_list.append(tab_emb)

        if not feature_list:
            raise ValueError("At least one modality must be enabled!")

        fusion_details = None
        if self.cross_modal_fusion_type == "concat":
            fused_feat = torch.cat(feature_list, dim=-1)
        else:
            if return_details:
                fused_feat, fusion_details = self.cross_modal_fusion(feature_list, return_details=True)
            else:
                fused_feat = self.cross_modal_fusion(feature_list, return_details=False)

        context = self.temporal_encoder(fused_feat, batch['mask'].to(model_device))
        pred = self.head(context)

        if return_details:
            return pred, {
                "fusion": fusion_details,
                "feature_list": feature_list,
                "fused_feat": fused_feat,
                "context": context
            }
        return pred


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============ Main Training Function ============
if __name__ == "__main__":
    # Initialize logging system
    logger = Logger(LOG_FILE)
    sys.stdout = logger

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 80)
    print("Training Binary Classification Model (Class 0 vs Class 1)")
    print("   SVI features (columns 0-16) included in tabular processing")
    print(f"   Adapter Type: {ADAPTER_TYPE}")
    print(f"   Tabular Processing: {TABULAR_PROCESSING}")
    print(f"   Cross-Modal Fusion: {CROSS_MODAL_FUSION}")
    print(f"   Temporal Model: {TEMPORAL_MODEL}")
    # Print L2 regularization configuration
    if L2_REGULARIZATION:
        print(f"   L2 Regularization: ENABLED")
        print(f"      - Weight Decay: {WEIGHT_DECAY}")
        if L2_LAMBDA > 0:
            print(f"      - Explicit L2 Lambda: {L2_LAMBDA}")
        else:
            print(f"      - Explicit L2 Lambda: Disabled (using weight_decay only)")
    else:
        print(f"   L2 Regularization: DISABLED")
    print(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load data
    train_ds = NPZDataset(
        os.path.join(DATA_DIR, "train.npz"),
        feature_config=FEATURE_CONFIG
    )
    tabular_min = train_ds.tabular_min
    tabular_max = train_ds.tabular_max
    class_weights = train_ds.class_weights

    val_ds = NPZDataset(
        os.path.join(DATA_DIR, "val.npz"),
        tabular_min=tabular_min,
        tabular_max=tabular_max,
        class_weights=class_weights,
        feature_config=FEATURE_CONFIG
    )

    train_loader = DataLoader(
        train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True,
        collate_fn=custom_collate, num_workers=NUM_WORKERS, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=False,
        collate_fn=custom_collate, num_workers=NUM_WORKERS, pin_memory=True
    )

    # Print ablation configuration
    print("\nAblation Configuration:")
    enabled_features = []
    if ABLATION_CONFIG["use_svi"]:
        enabled_features.append("SVI (cols 0-7)")
    if ABLATION_CONFIG["use_env_exposure"]:
        enabled_features.append("Env (cols 8-11)")
    if ABLATION_CONFIG["use_urban_function"]:
        enabled_features.append("Urban/POI (cols 12-20)")
    if ABLATION_CONFIG["use_activity"]:
        enabled_features.append("Activity (cols 21-25)")
    if ABLATION_CONFIG["use_personal_ses"]:
        enabled_features.append("SES (cols 26-29)")
    if ABLATION_CONFIG["use_satellite"]:
        enabled_features.append("Satellite")
    if ABLATION_CONFIG["use_gps"]:
        enabled_features.append("GPS")

    print(f"   Enabled features: {', '.join(enabled_features)}")
    print("-" * 80)

    # Create model
    model = FullStackPredictor(
        train_ds.tabular_dim,
        num_classes=NUM_CLASSES,
        adapter_type=ADAPTER_TYPE,
        temporal_model=TEMPORAL_MODEL,
        ablation_config=ABLATION_CONFIG,
        feature_config=FEATURE_CONFIG,
        adapter_output_dim=ADAPTER_OUTPUT_DIM,
        adapter_patch_dim=ADAPTER_PATCH_DIM,
        adapter_cls_dim=ADAPTER_CLS_DIM,
        use_cls_in_adapter=USE_CLS_IN_ADAPTER,
        cross_modal_fusion=CROSS_MODAL_FUSION,
        tabular_output_dim=TABULAR_OUTPUT_DIM,
        tabular_processing=TABULAR_PROCESSING,
        center_aware=CENTER_AWARE,
        center_weight_init=CENTER_WEIGHT_INIT
    ).to(device)

    model.apply(init_weights)
    print(f"Model Parameters: {count_parameters(model):,}")

    # Loss function and optimizer
    class_weights_tensor = torch.tensor(class_weights).float().to(device)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=GAMMA, label_smoothing=LABEL_SMOOTHING)

    # CE Loss for validation evaluation
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Set optimizer based on L2 configuration
    if L2_REGULARIZATION:
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=WEIGHT_DECAY)
        print(f"Using AdamW optimizer with weight_decay={WEIGHT_DECAY}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=WEIGHT_DECAY)
        print(f"L2 regularization disabled ({WEIGHT_DECAY})")

    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_plateau_epochs, gamma=lr_decay_factor)
    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    best_val_acc = 0.0
    best_val_auc = 0.0
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_focal_loss = 0  # Track focal loss and L2 loss separately
        total_l2_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
            label = batch['label'].to(device)

            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
                pred = model(batch, return_details=False)
                focal_loss = criterion(pred, label)

                # Compute explicit L2 loss
                if L2_REGULARIZATION and L2_LAMBDA > 0:
                    l2_loss = compute_l2_loss(model)
                    loss = focal_loss + L2_LAMBDA * l2_loss
                else:
                    l2_loss = torch.tensor(0.0).to(device)
                    loss = focal_loss

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            # Record losses
            total_focal_loss += focal_loss.item()
            total_l2_loss += l2_loss.item()

            # Update every GRAD_ACCUM_STEPS steps
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS

        # Handle the last incomplete accumulation batch
        if len(train_loader) % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        val_focal_loss = 0.0
        val_ce_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []  # Collect positive class probabilities

        with torch.no_grad():
            for batch in val_loader:
                label = batch['label'].to(device)
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
                    pred = model(batch, return_details=False)

                    # Compute Focal Loss
                    batch_focal_loss = criterion(pred, label)
                    val_focal_loss += batch_focal_loss.item()

                    # Compute CE Loss
                    batch_ce_loss = ce_criterion(pred, label)
                    val_ce_loss += batch_ce_loss.item()

                    _, predicted = torch.max(pred, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())
                    all_probs.extend(torch.softmax(pred, dim=1)[:, 1].cpu().numpy())  # Collect positive class probabilities

        avg_val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        avg_val_focal_loss = val_focal_loss / len(val_loader)
        avg_val_ce_loss = val_ce_loss / len(val_loader)

        # Compute per-class precision, recall, f1
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=[0, 1]
        )

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        # Print L2 loss
        if L2_REGULARIZATION and L2_LAMBDA > 0:
            print(f"  Train Loss (Total): {total_loss / len(train_loader):.4f} "
                  f"(Focal: {total_focal_loss / len(train_loader):.4f}, "
                  f"L2: {total_l2_loss / len(train_loader):.6f})")
        else:
            print(f"  Train Loss (Focal): {total_loss / len(train_loader):.4f}")

        print(f"  Val Loss (Focal): {avg_val_focal_loss:.4f} | Val Loss (CE): {avg_val_ce_loss:.4f}")
        print(f"  Val Acc: {avg_val_acc:.4f}")
        print(f"  Class 0 (Low):  P={precision[0]:.3f}, R={recall[0]:.3f}, F1={f1[0]:.3f}, N={support[0]}")
        print(f"  Class 1 (High): P={precision[1]:.3f}, R={recall[1]:.3f}, F1={f1[1]:.3f}, N={support[1]}")

        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            model_name = f"best_{epoch}_{FEATURE_TYPE}_{ADAPTER_TYPE}_{TEMPORAL_MODEL}_binary_l2.pth"
            torch.save(model.state_dict(), model_name)
            print(f"  Saved: {model_name} (Acc: {avg_val_acc:.4f})")
        #avg_val_auc = roc_auc_score(all_labels, all_preds)
        # avg_val_auc = roc_auc_score(all_labels, all_probs)
        # if avg_val_auc > best_val_auc:
        #     best_val_auc = avg_val_auc
            #model_name = f"best_{epoch}_{FEATURE_TYPE}_{ADAPTER_TYPE}_{TEMPORAL_MODEL}_binary_l2.pth"
            #torch.save(model.state_dict(), model_name)
            #print(f"  Saved: {model_name} (AUC: {avg_val_auc:.4f})")

        scheduler.step()
        print("-" * 80)

    print("=" * 80)
    print(f"Training Complete! Best Val Acc: {best_val_acc:.4f}")
    #print(f"Training Complete! Best Val AUC: {best_val_auc:.4f}")
    print(f"End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Close log file
    logger.close()