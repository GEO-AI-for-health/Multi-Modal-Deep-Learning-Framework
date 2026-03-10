# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

HIDDEN_DIM = 64
# ============ 1. Feature-wise Linear Modulation (FiLM) ============
class FiLMTabularProcessor(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Feature embedding
        self.feature_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # FiLM generators (生成scale和shift参数)
        self.gamma_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.beta_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        """
        x: [B, S, input_dim] or [B, input_dim]
        """
        # Feature embedding
        h = self.feature_embed(x)

        # Generate modulation parameters
        gamma = self.gamma_generator(x)
        beta = self.beta_generator(x)

        # Apply FiLM: h_modulated = gamma * h + beta
        h_modulated = gamma * h + beta

        # Output
        output = self.output_proj(h_modulated)
        return output


# ============ 2. Self-Attention based Processor ============
class AttentionTabularProcessor(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.embed_dim = ((input_dim + num_heads - 1) // num_heads) * num_heads

        self.input_proj = nn.Linear(input_dim, self.embed_dim)

        # Multi-layer self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 2,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.embed_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        """
        x: [B, S, input_dim] or [B, input_dim]
        """
        # Reshape if needed
        original_shape = x.shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B, input_dim] -> [B, 1, input_dim]

        B, S, D = x.shape

        # Project to embed_dim
        x = self.input_proj(x)  # [B, S, embed_dim]

        # Reshape: [B, S, embed_dim] -> [B*S, embed_dim] -> treat embed_dim as sequence
        x = x.reshape(B * S, self.embed_dim)

        x = x.unsqueeze(1)  # [B*S, 1, embed_dim]

        # Actually, let's treat each feature dimension as a token
        x = x.squeeze(1)  # [B*S, embed_dim]
        # Treat as sequence of features
        num_features = self.embed_dim
        x = x.view(B * S, 1, num_features)

        # Apply attention (treating feature dims as sequence)
        x_t = x.transpose(1, 2)  # [B*S, num_features, 1]
        x_t = self.transformer(x_t)  # [B*S, num_features, 1]

        # Pool over features
        x_pooled = x_t.mean(dim=1)  # [B*S, 1]
        x_pooled = x_pooled.view(B, S, -1)  # [B, S, 1]

        # This approach is too complex, let me simplify
        # Just use attention to weight features
        x = x.reshape(B, S, self.embed_dim)

        # Simple feature-wise attention
        attn_weights = F.softmax(x, dim=-1)  # [B, S, embed_dim]
        x_attended = x * attn_weights  # [B, S, embed_dim]

        # Output
        output = self.output_proj(x_attended)

        # Restore shape if needed
        if len(original_shape) == 2:
            output = output.squeeze(1)

        return output


# ============ 3. Gated Linear Unit (GLU) based Processor ============
class GLUTabularProcessor(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build GLU layers
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            if i < num_layers - 1:
                next_dim = max(output_dim, current_dim // 2)
            else:
                next_dim = output_dim

            # GLU: split into two parts, one for value, one for gate
            layers.append(nn.Linear(current_dim, next_dim * 2))
            layers.append(GLU())

            if i < num_layers - 1:
                layers.append(nn.Dropout(0.1))

            current_dim = next_dim

        self.glu_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.glu_net(x)


class GLU(nn.Module):
    """Gated Linear Unit"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Split input into two halves
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


# ============ 4. Feature Tokenizer + Transformer ============
class FeatureTokenizer(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim, d_model=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        self.feature_embeddings = nn.Parameter(torch.randn(input_dim, d_model))

        # Value projection
        self.value_proj = nn.Linear(1, d_model)

        # Positional encoding for features
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, x):
        """
        x: [B, S, input_dim] or [B, input_dim]
        """
        original_shape = x.shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B, input_dim] -> [B, 1, input_dim]

        B, S, D = x.shape

        # Reshape to treat each feature as a token
        x = x.reshape(B * S, D, 1)  # [B*S, D, 1]

        # Project values
        x_proj = self.value_proj(x)  # [B*S, D, d_model]

        # Add feature embeddings
        x_embed = x_proj + self.feature_embeddings.unsqueeze(0)

        # Add positional encoding
        x_embed = x_embed + self.pos_encoding

        # Apply transformer
        x_transformed = self.transformer(x_embed)  # [B*S, D, d_model]

        # Flatten and project
        x_flat = x_transformed.reshape(B, S, -1)  # [B, S, D*d_model]
        output = self.output_proj(x_flat)

        if len(original_shape) == 2:
            output = output.squeeze(1)

        return output


# ============ 5. Factorization Machine (FM) based Processor ============
class FMTabularProcessor(nn.Module):

    def __init__(self, input_dim, output_dim, k=16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k  # latent dimension

        # Linear part
        self.linear = nn.Linear(input_dim, output_dim)

        # Factorization parameters for pairwise interactions
        self.V = nn.Parameter(torch.randn(input_dim, k))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim + k, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, x):
        """
        x: [B, S, input_dim] or [B, input_dim]
        """
        # Linear part
        linear_part = self.linear(x)  # [B, S, output_dim] or [B, output_dim]

        # FM part: 0.5 * sum((sum(x*V))^2 - sum((x*V)^2))
        # where V is the factorization matrix

        # Expand V for broadcasting
        V = self.V.unsqueeze(0)  # [1, input_dim, k]
        if len(x.shape) == 3:
            V = V.unsqueeze(0)  # [1, 1, input_dim, k]

        # Compute x * V
        xV = x.unsqueeze(-1) * V  # Broadcasting

        # Sum over features
        sum_xV = xV.sum(dim=-2)  # [B, S, k] or [B, k]

        # Square of sum
        sum_square = sum_xV ** 2

        # Sum of squares
        square_xV = xV ** 2
        square_sum = square_xV.sum(dim=-2)

        # FM interaction
        fm_part = 0.5 * (sum_square - square_sum)  # [B, S, k] or [B, k]

        # Combine
        combined = torch.cat([linear_part, fm_part], dim=-1)
        output = self.output_proj(combined)

        return output


# ============ 6. Cross Network (DCN) ============
class CrossNetwork(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers

        # Cross layers
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1)) for _ in range(num_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])

        # Deep network
        self.deep_net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, x):
        """
        x: [B, S, input_dim] or [B, input_dim]
        """
        x0 = x  # Save input

        # Cross network
        xl = x
        for i in range(self.num_layers):
            # xl+1 = x0 * (xl^T * w) + xl + b
            w = self.cross_weights[i]
            b = self.cross_biases[i]

            # Compute xl^T * w
            xl_w = torch.matmul(xl, w)  # [B, S, 1] or [B, 1]

            # Multiply by x0
            x0_xl_w = x0 * xl_w

            # Add residual and bias
            xl = x0_xl_w + xl + b

        cross_output = xl

        # Deep network
        deep_output = self.deep_net(x)

        # Concatenate and project
        combined = torch.cat([cross_output, deep_output], dim=-1)
        output = self.output_proj(combined)

        return output


# ============ 7. ResNet-style Processor ============
class ResNetTabularProcessor(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim, num_blocks=3, hidden_dim=128):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return self.output_proj(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.block(x))


# ============ 8. Feature Group Processor ============

class FeatureGroupProcessor(nn.Module):
    """
    """

    def __init__(self, feature_groups, output_dim):
        """
        feature_groups: dict, {group_name: (start_idx, end_idx)}
        """
        super().__init__()
        self.feature_groups = feature_groups

        # 为每个特征组创建独立的处理器
        self.group_processors = nn.ModuleDict()
        group_output_dims = []

        for group_name, (start, end) in feature_groups.items():
            group_dim = end - start
            group_output = max(16, group_dim // 2)  # 16
            group_output_dims.append(group_output)

            self.group_processors[group_name] = nn.Sequential(
                nn.Linear(group_dim, group_dim*2),
                #nn.LayerNorm(group_dim * 2),  #
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(group_dim*2, group_dim*2),
                # #nn.LayerNorm(group_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                # nn.Linear(group_dim*2, group_dim*2),
                # nn.LayerNorm(group_dim * 2),
                # nn.ReLU(),
                # nn.Dropout(0.1),
                nn.Linear(group_dim*2, group_output),
                nn.ReLU()
            )

        # 融合层
        total_group_output = sum(group_output_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_group_output, output_dim*2),
            #nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim*2, output_dim*2),
            # #nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Linear(output_dim*2, output_dim*2),
            # nn.LayerNorm(output_dim*2),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(output_dim*2, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: [B, S, total_input_dim] or [B, total_input_dim]
        """
        group_outputs = []

        for group_name, (start, end) in self.feature_groups.items():
            # Extract group features
            group_feat = x[..., start:end]

            # Process
            group_out = self.group_processors[group_name](group_feat)
            group_outputs.append(group_out)

        # Concatenate and fuse
        combined = torch.cat(group_outputs, dim=-1)
        output = self.fusion(combined)

        return output


# ============ 8. Nonlinear Preserving Processor ============
class NonlinearPreservingProcessor(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim, num_thresholds=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_thresholds = num_thresholds

        self.thresholds = nn.Parameter(
            torch.linspace(-3, 3, num_thresholds).unsqueeze(0).repeat(input_dim, 1)
        )

        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.nonlinear_enhancer = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim * 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        original_shape = x.shape
        has_seq = len(x.shape) == 3

        if not has_seq:
            x = x.unsqueeze(1)

        B, S, D = x.shape

        feat_linear = x
        feat_quad = x ** 2
        feat_inv_quad = -(x - 0.5) ** 2

        x_expanded = x.unsqueeze(-1)
        thresholds_expanded = self.thresholds.unsqueeze(0).unsqueeze(0)
        soft_thresholds = torch.sigmoid(20 * (x_expanded - thresholds_expanded))
        feat_threshold = soft_thresholds.mean(dim=-1)

        combined = feat_linear + feat_quad * 0.5 + feat_inv_quad * 0.5 + feat_threshold
        transformed = self.feature_transform(combined)
        enhanced = self.nonlinear_enhancer(transformed)
        output = enhanced + self.residual_weight * transformed

        if not has_seq:
            output = output.squeeze(1)

        return output


# ============ 9. Neural Decision Tree ============
class NeuralDecisionTree(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim, depth=3, temperature=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.temperature = temperature

        self.num_leaves = 2 ** depth
        self.num_internal = self.num_leaves - 1


        self.feature_selectors = nn.Parameter(torch.randn(self.num_internal, input_dim))
        self.split_thresholds = nn.Parameter(torch.randn(self.num_internal))


        self.leaf_embeddings = nn.Parameter(torch.randn(self.num_leaves, output_dim))

        self.post_transform = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        original_shape = x.shape
        has_seq = len(x.shape) == 3

        if not has_seq:
            x = x.unsqueeze(1)

        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)


        leaf_probs = self._compute_leaf_probabilities(x_flat)


        output = torch.matmul(leaf_probs, self.leaf_embeddings)
        output = self.post_transform(output)
        output = output.reshape(B, S, self.output_dim)

        if not has_seq:
            output = output.squeeze(1)

        return output

    def _compute_leaf_probabilities(self, x):
        batch_size = x.shape[0]
        node_probs = torch.ones(batch_size, 1, device=x.device)

        for level in range(self.depth):
            num_nodes_at_level = 2 ** level
            new_probs = []

            for node_idx in range(num_nodes_at_level):
                internal_idx = num_nodes_at_level - 1 + node_idx

                if internal_idx >= self.num_internal:
                    break

                current_prob = node_probs[:, node_idx]

                # 特征加权
                feature_weights = F.softmax(self.feature_selectors[internal_idx], dim=0)
                feature_value = torch.matmul(x, feature_weights)

                # 软分割
                split_prob = torch.sigmoid(
                    (feature_value - self.split_thresholds[internal_idx]) / self.temperature
                )

                left_prob = current_prob * (1 - split_prob)
                right_prob = current_prob * split_prob

                new_probs.append(left_prob.unsqueeze(1))
                new_probs.append(right_prob.unsqueeze(1))

            node_probs = torch.cat(new_probs, dim=1)

        return node_probs


# ============ 10. Tree-MLP Ensemble ============
class TreeMLPEnsemble(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim, tree_depth=3):
        super().__init__()

        tree_dim = output_dim // 2
        mlp_dim = output_dim - tree_dim

        self.tree = NeuralDecisionTree(input_dim, tree_dim, depth=tree_depth)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, mlp_dim),
            nn.Tanh()
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        tree_out = self.tree(x)
        mlp_out = self.mlp(x)
        combined = torch.cat([tree_out, mlp_out], dim=-1)
        output = self.fusion(combined)
        return output


# ============ Factory Function ============
def create_tabular_processor(processor_type, input_dim, output_dim, **kwargs):
    """
    """
    processors = {
        'linear': lambda: nn.Linear(input_dim, output_dim),

        'mlp': lambda: nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        ),

        'film': lambda: FiLMTabularProcessor(input_dim, output_dim),

        'attention': lambda: AttentionTabularProcessor(
            input_dim, output_dim,
            num_heads=kwargs.get('num_heads', 4),
            num_layers=kwargs.get('num_layers', 2)
        ),

        'glu': lambda: GLUTabularProcessor(
            input_dim, output_dim,
            num_layers=kwargs.get('num_layers', 3)
        ),

        'tokenizer': lambda: FeatureTokenizer(
            input_dim, output_dim,
            d_model=kwargs.get('d_model', 64),
            num_heads=kwargs.get('num_heads', 4),
            num_layers=kwargs.get('num_layers', 2)
        ),

        'fm': lambda: FMTabularProcessor(
            input_dim, output_dim,
            k=kwargs.get('k', 16)
        ),

        'dcn': lambda: CrossNetwork(
            input_dim, output_dim,
            num_layers=kwargs.get('num_layers', 3)
        ),

        'resnet': lambda: ResNetTabularProcessor(
            input_dim, output_dim,
            num_blocks=kwargs.get('num_blocks', 3),
            hidden_dim=kwargs.get('hidden_dim', 128)
        ),

        'grouped': lambda: FeatureGroupProcessor(
            kwargs.get('feature_groups'),
            output_dim
        ),

        'nonlinear_preserving': lambda: NonlinearPreservingProcessor(
            input_dim, output_dim,
            num_thresholds=kwargs.get('num_thresholds', 3)
        ),

        'neural_tree': lambda: NeuralDecisionTree(
            input_dim, output_dim,
            depth=kwargs.get('depth', 3),
            temperature=kwargs.get('temperature', 1.0)
        ),

        'tree_mlp': lambda: TreeMLPEnsemble(
            input_dim, output_dim,
            tree_depth=kwargs.get('tree_depth', 3)
        )
    }

    if processor_type not in processors:
        raise ValueError(f"Unknown processor type: {processor_type}. "
                         f"Choose from {list(processors.keys())}")

    return processors[processor_type]()
