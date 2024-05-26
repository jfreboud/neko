import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TransformerArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_leads: int
    embedding_dim: int
    norm_eps: float = 1e-5
    rope_theta: float = 10000


class RMSNorm(torch.nn.Module):
    """
    Root mean squared norm.

    Parameters
    ----------
    dims: int
        Embedding dimension.
    eps: float
        Epsilon value to avoid 0 division.
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dims))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdims=True) + self.eps)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.

        Returns
        -------
        _: torch.Tensor
            The output tensor.
        """
        output = self._norm(x.type(torch.float32)).type(x.dtype)
        return self.weight * output


class Attention(torch.nn.Module):
    """
    Module that can handle contextual information thanks to attention.

    Parameters
    ----------
    args: ModelArgs
        Model parameters.
    """

    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = torch.nn.Linear(
            args.dim, args.n_heads * args.head_dim, bias=False
        )
        self.wk = torch.nn.Linear(
            args.dim, args.n_heads * args.head_dim, bias=False
        )
        self.wv = torch.nn.Linear(
            args.dim, args.n_heads * args.head_dim, bias=False
        )
        self.wo = torch.nn.Linear(
            args.n_heads * args.head_dim, args.dim, bias=False
        )

    @staticmethod
    def create_additive_causal_mask(
        context_len: int, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Create causal mask.

        Parameters
        ---------
        context_len: int
            Context length.
        dtype: torch.dtype
            Precision type.

        Returns
        -------
        mask: torch.Tensor
            The causal mask.
        """
        indices = torch.cat([torch.zeros((1,)), torch.arange(context_len)])
        mask = torch.tensor(indices[:, None] < indices[None])
        # usually inf but 1e9 is as good and softmax(full(1e9)) != nan
        mask = mask.type(dtype) * -1e9
        return mask

    @staticmethod
    def create_rotation_matrix(
        positions: torch.Tensor,
        embedding_dim: int,
        rope_theta: float,
        device
    ) -> torch.Tensor:
        """
        Generate the rotary matrix for RoPE.

        Parameters
        ----------
        context_len: int
            The context length.
        embedding_dim: int
            Embedding dimension.
        rope_theta: float
            RoPE theta.

        Returns
        -------
        R: torch.Tensor
            The rotary matrix of dimension
            (context_len, embedding_dim, embedding_dim).
        """
        R = torch.zeros(
            (len(positions), embedding_dim, embedding_dim),
            requires_grad=False,
            device=device
        )
        # positions = torch.arange(1, context_len+1).unsqueeze(1)
        # Create matrix theta (shape: context_len, embedding_dim // 2).
        slice_i = torch.arange(0, embedding_dim // 2, device=device)
        theta = rope_theta ** (-2.0 * (slice_i.float()) / embedding_dim)
        m_theta = positions * theta
        # Create sin and cos values.
        cos_values = torch.cos(m_theta)
        sin_values = torch.sin(m_theta)
        # Populate the rotary matrix R using 2D slicing.
        R[:, 2*slice_i, 2*slice_i] = cos_values
        R[:, 2*slice_i, 2*slice_i+1] = -sin_values
        R[:, 2*slice_i+1, 2*slice_i] = sin_values
        R[:, 2*slice_i+1, 2*slice_i+1] = cos_values
        return R

    def forward(
        self,
        x: torch.Tensor,
        rotation_matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.
        mask: torch.Tensor
            Causal mask.
        cache: (key_cache, value_cache): (torch.Tensor, torch.Tensor)
            cache for keys and values
            for generating tokens with past context.

        Returns
        -------
        (output, (keys, values)): (torch.Tensor, (torch.Tensor, torch.Tensor))
            output: the output tensor
            (keys, values): cache for keys and values
        """
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation.
        queries = queries.reshape(
            B, L, self.n_heads, -1
        ).transpose(1, 2)
        keys = keys.reshape(
            B, L, self.n_heads, -1
        ).transpose(1, 2)
        values = values.reshape(
            B, L, self.n_heads, -1
        ).transpose(1, 2)

        if cache is not None:
            key_cache, value_cache = cache

            queries = torch.einsum("bhlj,lij->bhli", [queries, rotation_matrix])
            keys = torch.einsum("bhlj,lij->bhli", [keys, rotation_matrix])

            keys = torch.concat([key_cache, keys], dim=2)
            values = torch.concat([value_cache, values], dim=2)

        else:
            queries = torch.einsum("bhlj,lij->bhli", [queries, rotation_matrix])
            keys = torch.einsum("bhlj,lij->bhli", [keys, rotation_matrix])

        scores = torch.matmul(queries, keys.transpose(2, 3)) * self.scale
        if mask is not None:
            scores += mask
        scores = torch.softmax(
            scores.type(torch.float32), dim=-1
        ).type_as(scores)

        output = torch.matmul(scores, values)  # (B, n_local_heads, L, head_dim)
        output = output.transpose(1, 2).contiguous().reshape(B, L, -1)

        return self.wo(output), (keys, values)


class FeedForward(torch.nn.Module):
    """
    MLP module.

    Parameters
    ----------
    args: ModelArgs
        Model parameters.
    """

    def __init__(self, args: TransformerArgs):
        super().__init__()

        self.w1 = torch.nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = torch.nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.

        Returns
        -------
        _: torch.Tensor
            The output tensor.
        """
        return self.w2(torch.nn.SiLU()(self.w1(x)) * self.w3(x))


class TransformerBlock(torch.nn.Module):
    """
    Transformer module.

    Parameters
    ----------
    args: ModelArgs
        Model parameters.
    """

    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(
        self,
        x: torch.Tensor,
        rotation_matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[
            Tuple[torch.Tensor,
                  Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.
        mask: torch.Tensor
            Causal mask.
        cache: (key_cache, value_cache): (torch.Tensor, torch.Tensor)
            cache for keys and values
            for generating tokens with past context.

        Returns
        -------
        (output, (keys, values)): (torch.Tensor, (torch.Tensor, torch.Tensor))
            output: the output tensor
            (keys, values): cache for keys and values
        """
        r, cache = self.attention(
            self.attention_norm(x),
            rotation_matrix=rotation_matrix,
            mask=mask,
            cache=cache
        )
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


class Transformer(torch.nn.Module):
    """
    Large Language Model module.

    Parameters
    ----------
    args: ModelArgs
        Model parameters.
    """

    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.args = args
        self.n_layers = args.n_layers
        self.input = torch.nn.Linear(
            args.n_leads, args.dim, bias=False
        )
        self.embedding = torch.nn.Linear(
            args.embedding_dim, args.dim, bias=False
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = torch.nn.Linear(args.dim, 12, bias=False)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(args=args) for _ in range(args.n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        cache=None,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.
        cache: (key_cache, value_cache): (torch.Tensor, torch.Tensor)
            cache for keys and values
            for generating tokens with past context.

        Returns
        -------
        (output, cache): (torch.Tensor, list)
            output: the output tensor
            cache: cache for keys and values for each layer
        """
        h1 = h
        h2 = self.input(x)

        mask = None
        if h1 is not None:
            mask = Attention.create_additive_causal_mask(h2.shape[1])
            mask = mask.type(h2.dtype)
            mask = mask.to(h2.device)

            h1 = h1[:, None, :]
            h1 = self.embedding(h1)
            h = torch.cat([h1, h2], dim=1)

        else:
            h = h2

        if cache is None:
            cache = [None] * len(self.layers)
            positions = torch.arange(
                1, h.shape[1] + 1, device=h.device
            ).unsqueeze(1)

        else:
            key_cache = cache[0][0]
            positions = torch.tensor(
                [key_cache.shape[2] + 1], device=h.device
            ).unsqueeze(1)

        rotation_matrix = Attention.create_rotation_matrix(
            positions=positions,
            embedding_dim=self.args.head_dim,
            rope_theta=self.args.rope_theta,
            device=h.device
        )

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(
                h,
                rotation_matrix=rotation_matrix,
                mask=mask,
                cache=cache[e]
            )

        if h1 is not None:
            h = h[:, 1:, :]

        return self.output(self.norm(h)), cache
