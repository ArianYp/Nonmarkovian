import math
import typing
import contextlib

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# flash-attn is GPU-only and may not be installed (e.g. the Non-Markovian venv).
# Fall back to a pure-PyTorch scaled-dot-product-attention + rotary path so the
# model is importable and runnable (incl. CPU smoke tests) without it. The two
# paths are numerically equivalent for full (non-causal) attention.
try:
  import flash_attn
  import flash_attn.layers.rotary
  _HAS_FLASH_ATTN = True
except Exception:  # ImportError, or CUDA/Triton load failure
  flash_attn = None
  _HAS_FLASH_ATTN = False

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def _amp_bf16(device: torch.device | str):
  """bf16 autocast on CUDA, no-op elsewhere (CPU has no bf16 flash path)."""
  dev = torch.device(device) if not isinstance(device, torch.device) else device
  if dev.type == "cuda":
    return torch.cuda.amp.autocast(dtype=torch.bfloat16)
  return contextlib.nullcontext()


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


def modulate(x, shift, scale):
  # print(x.shape, shift.shape, scale.shape)
  if shift.shape == x.shape:
    return x * (1 + scale) + shift
  else:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
  
# function overload
# def modulate(x: torch.Tensor,
#              shift: torch.Tensor,
#              scale: torch.Tensor) -> torch.Tensor:
#   return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def _attn_fallback(qkv, rotary_cos_sin):
  """Pure-PyTorch replacement for the flash-attn rotary + varlen attention path.

  ``qkv``: ``[b, s, 3, h, d]``. ``rotary_cos_sin``: the ``(cos, sin)`` cache from
  :class:`Rotary` with shape ``[1, s, 3, 1, d]`` (``cat(freqs, freqs)`` layout, with
  the value slot set to identity). Applies rotary to q and k, then runs full
  (non-causal) scaled-dot-product attention. Returns ``[b, s, h, d]``.
  """
  cos, sin = rotary_cos_sin
  # q/k share the same rotary (slot 0); slot 2 (v) is identity by construction.
  cos_q = cos[0, :, 0, 0, :].to(qkv.dtype)  # [s, d]
  sin_q = sin[0, :, 0, 0, :].to(qkv.dtype)  # [s, d]
  cos_q = cos_q[None, :, None, :]  # [1, s, 1, d]
  sin_q = sin_q[None, :, None, :]
  q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each [b, s, h, d]
  q = q * cos_q + rotate_half(q) * sin_q
  k = k * cos_q + rotate_half(k) * sin_q
  # [b, s, h, d] -> [b, h, s, d] for SDPA
  q, k, v = (t.transpose(1, 2) for t in (q, k, v))
  out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
  return out.transpose(1, 2)  # [b, s, h, d]


# function overload



#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.cuda.amp.autocast(enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    seqlen = None
    if len(t.shape) != 1:
      seqlen = t.shape[1]
      t = rearrange(t, 'b h -> (b h)')
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    if seqlen != None:
      t_emb = rearrange(t_emb, '(b h) d -> b h d', h=seqlen)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, rotary_cos_sin, c, seqlens=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    # print(c.shape,"c---------")
    if len(c.shape) == 2:
      c = c.unsqueeze(1).repeat(1, seq_len, 1)
    (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=2)

    # attention operation
    x_skip = x
    x = modulate(self.norm1(x), shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = rearrange(qkv,
                    'b s (three h d) -> b s three h d',
                    three=3,
                    h=self.n_heads)
    if _HAS_FLASH_ATTN:
      with torch.cuda.amp.autocast(enabled=False):
        cos, sin = rotary_cos_sin
        qkv = apply_rotary_pos_emb(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      qkv = rearrange(qkv, 'b s ... -> (b s) ...')
      if seqlens is None:
        cu_seqlens = torch.arange(
          0, (batch_size + 1) * seq_len, step=seq_len,
          dtype=torch.int32, device=qkv.device)
      else:
        cu_seqlens = seqlens.cumsum(-1)
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0., causal=False)
      x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
    else:
      with torch.cuda.amp.autocast(enabled=False):
        x = _attn_fallback(qkv, rotary_cos_sin)  # [b, s, h, d]
      x = rearrange(x, 'b s h d -> b s (h d)')

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x



class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    batch_size, seq_len = x.shape[0], x.shape[1]
    if len(c.shape) == 2:
      c = c.unsqueeze(1).repeat(1, seq_len, 1)
    shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
    x = modulate(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x


class CustomEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomEmbedding, self).__init__()
        # Initialize weights and bias manually
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.out_features= out_features
        # self.bias = nn.Parameter(torch.Tensor(out_features))
        # Initialize weights with a specific variance
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with variance of 0.001
        nn.init.normal_(self.weight, mean=0.0, std=0.0316)  # std = sqrt(0.001)
        # nn.init.zeros_(self.bias)

    def forward(self, input):
        batch_size = input.shape[0]
        input = rearrange(input, 'b s v -> (b s) v')
        normalized_embedding = F.normalize(self.weight, p=2, dim=0) #noralization over 
        normalized_embedding = normalized_embedding * math.sqrt(self.out_features) # make the variance as zero. 
        output = torch.nn.functional.linear(input, normalized_embedding)
        output = rearrange(output, '(b s) v -> b s v',b=batch_size)
        # print(output.shape)
        return output


class BFN_DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size

    # self.vocab_embed = EmbeddingLayer(config.model.hidden_size,
    #                                   vocab_size)
    if self.config.embedding_nml:
      self.prob_adapter = CustomEmbedding(vocab_size,config.model.hidden_size) #add a normalization option.
    else:
      self.prob_adapter = nn.Linear(vocab_size,config.model.hidden_size,bias=False) #the pro_adapter for bfn only. 
      torch.nn.init.kaiming_uniform_(self.prob_adapter.weight, a=math.sqrt(5)) #add weight initialization. 

    

    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(DDiTBlock(config.model.hidden_size,
                              config.model.n_heads,
                              config.model.cond_dim,
                              dropout=config.model.dropout))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim)
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, input_para, sigma):
    x = self.prob_adapter(input_para)
    if self.config.entropy_condition:
      entropy = -(input_para * torch.log(input_para)).sum(-1) #batch * seqlen.
      sigma = 1 - entropy / math.log(self.vocab_size) #batch * seqlen, as weight larger and important. 
      # print(sigma,"--------")

    c = F.silu(self.sigma_map(sigma))

    rotary_cos_sin = self.rotary_emb(x)

    with _amp_bf16(x.device):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
      x = self.output_layer(x, c)

    return x