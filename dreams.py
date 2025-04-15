import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# Implementation adapted from https://github.com/pluskal-lab/DreaMS


class FourierFeatures(nn.Module):
    def __init__(self, strategy, x_min, x_max, trainable=True, funcs='both', sigma=10, num_freqs=512):

        assert strategy in {'random', 'voronov_et_al', 'lin_float_int'}
        assert funcs in {'both', 'sin', 'cos'}
        assert x_min < 1

        super().__init__()
        self.funcs = funcs
        self.strategy = strategy
        self.trainable = trainable
        self.num_freqs = num_freqs

        if strategy == 'random':
            self.b = torch.randn(num_freqs) * sigma
        if self.strategy == 'voronov_et_al':
            self.b = torch.tensor(
                [1 / (x_min * (x_max / x_min) ** (2 * i / (num_freqs - 2))) for i in range(1, num_freqs)],
            )
        elif self.strategy == 'lin_float_int':
            self.b = torch.tensor(
                [1 / (x_min * i) for i in range(2, math.ceil(1 / x_min), 2)] +
                [1 / (1 * i) for i in range(2, math.ceil(x_max), 1)],
            )
        self.b = self.b.unsqueeze(0)

        self.b = nn.Parameter(self.b, requires_grad=self.trainable)
        self.register_parameter('Fourier frequencies', self.b)

    def forward(self, x):
        x = 2 * torch.pi * x @ self.b
        if self.funcs == 'both':
            x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        elif self.funcs == 'cos':
            x = torch.cos(x)
        elif self.funcs == 'sin':
            x = torch.sin(x)
        return x

    def num_features(self):
        return self.b.shape[1] if self.funcs != 'both' else 2 * self.b.shape[1]


class FeedForward(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim: int, depth=None, act_last=True, act=nn.ReLU, bias=True,
                 dropout=0.0):
        super().__init__()

        if isinstance(hidden_dim, int):
            assert depth is not None
            hidden_dim = [hidden_dim] * depth
        elif hidden_dim == 'interpolated':
            assert depth is not None
            hidden_dim = self.interpolate_interval(a=in_dim, b=out_dim, n=depth - 1, only_inter=True, rounded=True)
        elif isinstance(hidden_dim, Sequence):  # e.g. is List or Tuple
            depth = len(hidden_dim)
        else:
            raise ValueError

        self.ff = nn.ModuleList([])
        for l in range(depth):
            d1 = hidden_dim[l - 1] if l != 0 else in_dim
            d2 = hidden_dim[l] if l != depth - 1 else out_dim
            self.ff.append(nn.Linear(d1, d2, bias=bias))
            if l != depth - 1:
                self.ff.append(nn.Dropout(p=dropout))
            if l != depth - 1 or act_last:
                self.ff.append(act())
        self.ff = nn.Sequential(*self.ff)

    def forward(self, x):
        return self.ff(x)

    @staticmethod
    def interpolate_interval(a, b, n, only_inter=False, rounded=False):
        x_min, x_max = min(a, b), max(a, b)
        res = [x_min + i * (x_max - x_min) / (n + 1) for i in
               range(1 if only_inter else 0, n + 1 if only_inter else n + 2)]
        if x_max == a:
            res.reverse()
        if rounded:
            res = [round(x) for x in res]
        return res


class MultiheadAttention(nn.Module):

    def __init__(self, d_model, n_heads, att_dropout, no_transformer_bias, attn_mech, d_graphormer_params):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = att_dropout
        self.use_transformer_bias = not no_transformer_bias
        self.attn_mech = attn_mech
        self.d_graphormer_params = d_graphormer_params

        if self.d_model % self.n_heads != 0:
            raise ValueError('Required: d_model % n_heads == 0.')

        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim ** -0.5

        # Parameters for linear projections of queries, keys, values and output
        self.weights = torch.nn.Parameter(torch.Tensor(4 * self.d_model, self.d_model))
        if self.use_transformer_bias:
            self.biases = torch.nn.Parameter(torch.Tensor(4 * self.d_model))

        if self.d_graphormer_params:
            self.lin_graphormer = nn.Linear(self.d_graphormer_params, self.n_heads, bias=False)

        # initializing
        # If we do Xavier normal initialization, std = sqrt(2/(2D))
        # but it's too big and causes un-stability in PostNorm,
        # so we use the smaller std of feedforward module, i.e. sqrt(2/(5D))
        mean = 0
        std = (2 / (5 * self.d_model)) ** 0.5
        nn.init.normal_(self.weights, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.biases, 0.)

        if self.attn_mech == 'additive_v':
            self.additive_v = torch.nn.Parameter(torch.Tensor(self.n_heads, self.head_dim))
            nn.init.normal_(self.additive_v, mean=mean, std=std)

    def forward(self, q, k, v, mask, graphormer_dists=None, do_proj_qkv=True):
        bs, n, d = q.size()

        def _split_heads(tensor):
            bsz, length, d_model = tensor.size()
            return tensor.reshape(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)

        if do_proj_qkv:
            q, k, v = self.proj_qkv(q, k, v)

        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)

        if self.attn_mech == 'dot-product':
            att_weights = torch.einsum('bhnd,bhdm->bhnm', q, k.transpose(-2, -1))
        elif self.attn_mech == 'additive_v' or self.attn_mech == 'additive_fixed':
            att_weights = (q.unsqueeze(-2) - k.unsqueeze(-3))
            if self.attn_mech == 'additive_v':
                att_weights = (att_weights * self.additive_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))
            att_weights = att_weights.sum(dim=-1)
        else:
            raise NotImplementedError(f'"{self.attn_mech}" attention mechanism is not implemented.')
        att_weights = att_weights * self.scale

        if graphormer_dists is not None:
            if self.d_graphormer_params:
                # (bs, n, n, dists_d) -> (bs, n, n, n_heads) -> (bs, n_heads, n, n) = A.shape
                att_bias = self.lin_graphormer(graphormer_dists).permute(0, 3, 1, 2)
            else:
                # (bs, n, n, dists_d) -> (bs, 1, n, n) broadcastable with A
                att_bias = graphormer_dists.sum(dim=-1).unsqueeze(1)
            att_weights = att_weights + att_bias

        if mask is not None:
            att_weights.masked_fill_(mask.unsqueeze(1).unsqueeze(-1), -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)
        _att_weights = att_weights.reshape(-1, n, n)
        output = torch.bmm(_att_weights, v.reshape(bs * self.n_heads, -1, self.head_dim))
        output = output.reshape(bs, self.n_heads, n, self.head_dim).transpose(1, 2).reshape(bs, n, -1)
        output = self.proj_o(output)

        return output, att_weights

    def proj_qkv(self, q, k, v):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()

        if qkv_same:
            q, k, v = self._proj(q, end=3 * self.d_model).chunk(3, dim=-1)
        elif kv_same:
            q = self._proj(q, end=self.d_model)
            k, v = self._proj(k, start=self.d_model, end=3 * self.d_model).chunk(2, dim=-1)
        else:
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)

        return q, k, v

    def _proj(self, x, start=0, end=None):
        weight = self.weights[start:end, :]
        bias = None if not self.use_transformer_bias else self.biases[start:end]
        return F.linear(x, weight=weight, bias=bias)

    def proj_q(self, q):
        return self._proj(q, end=self.d_model)

    def proj_k(self, k):
        return self._proj(k, start=self.d_model, end=2 * self.d_model)

    def proj_v(self, v):
        return self._proj(v, start=2 * self.d_model, end=3 * self.d_model)

    def proj_o(self, x):
        return self._proj(x, start=3 * self.d_model)


class TokenWiseFeedForward(nn.Module):

    def __init__(self, ff_dropout, d_model, no_transformer_bias):
        super(TokenWiseFeedForward, self).__init__()
        self.dropout = ff_dropout
        self.d_model = d_model
        self.ff_dim = 4 * d_model
        self.use_transformer_bias = not no_transformer_bias

        self.in_proj = nn.Linear(self.d_model, self.ff_dim, bias=self.use_transformer_bias)
        self.out_proj = nn.Linear(self.ff_dim, self.d_model, bias=self.use_transformer_bias)

        # initializing
        mean = 0
        std = (2 / (self.ff_dim + self.d_model)) ** 0.5
        nn.init.normal_(self.in_proj.weight, mean=mean, std=std)
        nn.init.normal_(self.out_proj.weight, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.in_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):
        # my preliminary experiments show all RELU-variants
        # work the same and slower, RELU FTW!!!
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self.dropout, training=self.training)
        return self.out_proj(y)


class ScaleNorm(nn.Module):

    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class TransformerEncoder(nn.Module):

    def __init__(self, n_layers, pre_norm, d_model, scnorm, n_heads, att_dropout, no_transformer_bias, attn_mech,
                 d_graphormer_params, ff_dropout, residual_dropout):
        super(TransformerEncoder, self).__init__()
        self.residual_dropout = residual_dropout
        self.n_layers = n_layers
        self.pre_norm = pre_norm

        self.atts = nn.ModuleList(
            [MultiheadAttention(d_model, n_heads, att_dropout, no_transformer_bias, attn_mech, d_graphormer_params) for
             _ in range(self.n_layers)])
        self.ffs = nn.ModuleList(
            [TokenWiseFeedForward(ff_dropout, d_model, no_transformer_bias) for _ in range(self.n_layers)])

        num_scales = self.n_layers * 2 + 1 if self.pre_norm else self.n_layers * 2
        if scnorm:
            self.scales = nn.ModuleList([ScaleNorm(d_model ** 0.5) for _ in range(num_scales)])
        else:
            self.scales = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_scales)])

    def forward(self, src_inputs, src_mask, graphormer_dists=None):
        pre_norm = self.pre_norm
        post_norm = not pre_norm

        x = F.dropout(src_inputs, p=self.residual_dropout, training=self.training)
        for i in range(self.n_layers):
            att = self.atts[i]
            ff = self.ffs[i]
            att_scale = self.scales[2 * i]
            ff_scale = self.scales[2 * i + 1]

            residual = x
            x = att_scale(x) if pre_norm else x
            x, _ = att(q=x, k=x, v=x, mask=src_mask, graphormer_dists=graphormer_dists)
            x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
            x = att_scale(x) if post_norm else x

            residual = x
            x = ff_scale(x) if pre_norm else x
            x = ff(x)
            x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
            x = ff_scale(x) if post_norm else x

        x = self.scales[-1](x) if pre_norm else x
        return x


class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha=None, binary=False, return_softmax_out=False):
        """
        https://arxiv.org/pdf/1708.02002v2.pdf
        :param alpha: A vector summing up to one for multi-class classification, a positive-class scalar from (0, 1)
                      range for binary classification.
        :param return_softmax_out: If True, the return value of forward method is `(loss, softmax probabilities)` instead
               of `loss`.
        """
        super(FocalLoss, self).__init__()
        self.binary = binary
        self.gamma = gamma
        assert gamma >= 0
        self.alpha = alpha
        if self.alpha is not None:
            if not self.binary:
                raise NotImplementedError('Alpha weighting is currently implemented only for binary classification.')
            assert (isinstance(alpha, float) and 0 < alpha < 1) or (
                        isinstance(alpha, torch.Tensor) and alpha.sum() == 1)
        self.return_softmax_out = return_softmax_out

    def forward(self, inputs, targets):
        """
        :param inputs: Class logits of shape (..., num_classes).
        :param targets: One-hot class labels (..., num_classes).
        :return: Unreduced focal loss of shape (...).
        """

        if not self.binary:

            # Compute cross-entropy
            p = F.softmax(inputs, dim=-1)
            loss = F.nll_loss(p.log(), torch.argmax(targets, dim=-1), reduction='none')
            if self.gamma == 0:
                if self.return_softmax_out:
                    return loss, p
                return loss

            # Weight with focal loss terms
            p_t = (targets * p).sum(dim=-1)
            fl_term = (1 - p_t) ** self.gamma
            loss = fl_term * loss

            if self.return_softmax_out:
                return loss, p
            return loss

        else:
            weight = torch.ones(targets.shape, dtype=targets.dtype, device=targets.device)
            targets_mask = targets > 0

            if self.alpha is not None:
                weight[targets_mask] = self.alpha
                weight[~targets_mask] = 1 - self.alpha

            if self.gamma > 0:
                weight[targets_mask] *= (1 - inputs[targets_mask]) ** self.gamma
                weight[~targets_mask] *= inputs[~targets_mask] ** self.gamma

            return F.binary_cross_entropy(inputs, targets, weight=weight.detach())


class DreaMS(nn.Module):

    def __init__(
            self,
            fourier_strategy='lin_float_int',
            fourier_num_freqs=None,
            fourier_trainable=False,
            max_tbxic_stdev=0.0001,
            max_mz=1000,
            d_fourier=980,
            dropout=0.1,
            no_ffs_bias=False,
            ff_fourier_depth=5,
            ff_fourier_d=512,
            d_peak=44,
            ff_peak_depth=1,
            d_model=1024,
            ff_out_depth=1,
            hot_mz_bin_size=0.05,
            focal_loss_gamma=5.0,
            n_layers=7,
            pre_norm=True,
            scnorm=False,
            n_heads=8,
            att_dropout=0.1,
            no_transformer_bias=True,
            attn_mech='dot-product',
            d_graphormer_params=0,
            ff_dropout=0.1,
            residual_dropout=0.1,
            top_n=100,
    ):
        super().__init__()

        self.max_mz = max_mz
        self.d_model = d_model
        token_dim = 2  # m/z and intensity
        self.top_n = top_n

        self.fourier_enc = FourierFeatures(
            strategy=fourier_strategy, num_freqs=fourier_num_freqs, trainable=fourier_trainable,
            x_min=max_tbxic_stdev, x_max=max_mz,
        )

        self.ff_fourier = FeedForward(
            in_dim=self.fourier_enc.num_features(), out_dim=d_fourier,
            bias=not no_ffs_bias, dropout=dropout,
            depth=ff_fourier_depth, hidden_dim=ff_fourier_d,
        )

        self.ff_peak = FeedForward(
            in_dim=token_dim, hidden_dim=d_peak, out_dim=d_peak,
            depth=ff_peak_depth,
            dropout=dropout, bias=not no_ffs_bias,
        )

        self.transformer_encoder = TransformerEncoder(n_layers, pre_norm, d_model, scnorm, n_heads, att_dropout,
                                                      no_transformer_bias, attn_mech, d_graphormer_params, ff_dropout,
                                                      residual_dropout)

        self.ff_out = FeedForward(
            in_dim=d_model, hidden_dim=d_model, depth=ff_out_depth, act_last=False,
            out_dim=int(max_mz / hot_mz_bin_size),
            dropout=dropout, bias=True,
        )

        self.mz_masking_loss = FocalLoss(gamma=focal_loss_gamma, return_softmax_out=True)

        self.ro_out = nn.Linear(2 * self.d_model, 1, bias=False)

    def forward(self, spec):
        # Generate padding mask
        padding_mask = spec[:, :, 0] == 0

        # Lift peaks to d_peak (m/z's are normalized)
        peak_embeds = self.ff_peak(self.__normalize_spec(spec))

        # Concatenate with fourier features (d_peak -> d_peak + d_fourier ("num_fourier_features" -> d_fourier))
        fourier_features = self.ff_fourier(self.fourier_enc(spec[..., [0]]))
        spec = torch.cat([peak_embeds, fourier_features], dim=-1)

        graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)

        # Transformer encoder blocks
        spec = self.transformer_encoder(spec, padding_mask, graphormer_dists)

        return spec

    def __normalize_spec(self, spec):
        """
        Normalizes raw m/z values. Notice, that it is not in dataset `__getitem__ `because raw m/z values are still needed
        for Fourier features. Intensities are supposed to be normalized in `__getitem__`.
        """
        return spec / torch.tensor([self.max_mz, 1.], device=spec.device, dtype=spec.dtype)

    @staticmethod
    def to_classes(vals: torch.Tensor, max_val: float, bin_size: float, special_vals=(), return_num_classes=False):
        special_masks = [vals == v for v in special_vals]
        num_classes = int(max_val / bin_size)
        classes = torch.round(vals / bin_size).long()
        classes = classes.clamp(max=num_classes - 1)  # clamp not to have a separate class for max_mz
        for i, m in enumerate(special_masks):
            classes[m] = num_classes + i
        if return_num_classes:
            return classes, num_classes + len(special_vals)
        return classes
