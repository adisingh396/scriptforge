"""ScriptForge v2 — text-sequence conditioned handwriting synthesis.

This is the sequence-to-image model. No font rasterisation is involved: the generator is
conditioned on the *characters themselves*, so there is no synthetic-font / real-handwriting
pixel mismatch to fight.

Tensor shapes for the default canvas (H=40, W=192, L<=24 characters, cols = W/4 = 48):

    text  (B,)                  -> ids            (B, L)          int64, pad = vocab
    ids                         -> embedding      (B, L, 64)
    embedding                   -> BiGRU          (B, L, 256)
    (queries (B,48,256), keys (B,L,256), mask (B,L))
                                -> attention a    (B, 48, L)
    a @ values                  -> context        (B, 48, 256)
    context                     -> 1x1 conv       (B, 24, 1, 48)
    -> bilinear to (H, W)       ->                (B, 24, 40, 192)
    -> 2x conv refiner + pos    -> cond           (B, 16, 40, 192)
    cond [+ FiLM(z)] -> U-Net   -> image          (B, 1, 40, 192) in [-1, 1]

Why attention instead of upsampling the text embedding: handwritten characters overlap and
connect, so the horizontal position of a character is not fixed. Each output column learns its
own query over the character sequence, which lets the alignment be soft and learned.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sf_models import CHARSET, _ConvBlock


# --------------------------------------------------------------------------- text encoder

class TextEncoder(nn.Module):
    """characters -> per-character contextual features."""

    def __init__(self, vocab, embed_dim=64, hidden=128, layers=1, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden * 2

    def forward(self, ids, lengths=None):
        """ids (B,L) -> (B,L,2*hidden) and a validity mask (B,L) bool."""
        emb = self.embed(ids)
        enc, _ = self.rnn(emb)
        if lengths is None:
            mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
        else:
            steps = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            mask = steps < lengths.unsqueeze(1)
        enc = enc * mask.unsqueeze(-1).to(enc.dtype)
        return enc, mask


class WriterEmbedding(torch.nn.Module):
    """Maps a real writer id to the style vector the conditioner consumes."""

    def __init__(self, n_writers, style_dim, scale=0.5):
        super().__init__()
        self.embed = nn.Embedding(int(n_writers), int(style_dim))
        nn.init.normal_(self.embed.weight, std=scale)

    def forward(self, writer):
        return self.embed(writer)


class TextCondition(nn.Module):
    """character features -> spatial condition map, one attention query per output column."""

    def __init__(self, enc_dim=256, cond_ch=16, height=40, width=192, cols=None,
                 style_dim=0, mid_ch=24, n_writers=0, writer_cond=False):
        super().__init__()
        self.height, self.width = int(height), int(width)
        self.cols = int(cols or max(8, width // 4))
        self.enc_dim = enc_dim
        self.cond_ch = cond_ch
        # learned per-column queries: one row per horizontal region of the canvas
        self.q = nn.Parameter(torch.randn(self.cols, enc_dim) * 0.02)
        self.proj = nn.Conv2d(enc_dim, mid_ch, 1)
        self.refine = nn.Sequential(
            _ConvBlock(mid_ch, cond_ch),
            _ConvBlock(cond_ch, cond_ch),
        )
        # absolute position prior so the encoder knows where it is on the canvas
        self.pos = nn.Parameter(torch.zeros(1, cond_ch, self.height, self.width))
        self.style_dim = int(style_dim or 0)
        self.writer_cond = bool(writer_cond)
        self.n_writers = int(n_writers or 0)
        self.wembed = WriterEmbedding(self.n_writers, self.style_dim) if self.writer_cond else None
        # Non-zero init for film: zero-init froze the FiLM path for thousands of steps
        # (style_delta stayed 0.000 in every probe). Small kaiming init activates it.
        self.film = nn.Linear(self.style_dim, cond_ch * 2) if self.style_dim > 0 else None
        if self.film is not None:
            nn.init.kaiming_uniform_(self.film.weight, a=0.1)
            nn.init.zeros_(self.film.bias)

    def attention(self, enc, mask):
        b = enc.shape[0]
        q = self.q.unsqueeze(0).expand(b, -1, -1)               # (B,cols,D)
        logits = torch.bmm(q, enc.transpose(1, 2)) / (self.enc_dim ** 0.5)  # (B,cols,L)
        logits = logits.masked_fill(~mask.unsqueeze(1), -1e4)
        return logits.softmax(dim=-1)                           # (B,cols,L)

    def forward(self, enc, mask, z=None, writer=None):
        a = self.attention(enc, mask)                           # (B,cols,L)
        ctx = torch.bmm(a, enc)                                 # (B,cols,D)
        ctx = ctx.transpose(1, 2).unsqueeze(2)                  # (B,D,1,cols)
        ctx = self.proj(ctx)                                    # (B,mid,1,cols)
        ctx = F.interpolate(ctx, size=(self.height, self.width),
                            mode="bilinear", align_corners=False)  # (B,mid,H,W)
        cond = self.refine(ctx) + self.pos                       # (B,cond,H,W)
        if self.writer_cond and writer is not None:
            z = self.wembed(writer)                              # (B,style_dim)
        if self.film is not None and z is not None:
            scale, bias = self.film(z).chunk(2, dim=-1)
            cond = cond * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        return cond


# --------------------------------------------------------------------------- generator

class UNetGeneratorV2(nn.Module):
    """Same U-Net skeleton as v1, but the input is the text-condition map, not a font image."""

    def __init__(self, cond_ch=16, base=32):
        super().__init__()
        self.cond_ch = cond_ch
        self.base = base
        self.enc0 = _ConvBlock(cond_ch, base, stride=1)
        self.down1 = _ConvBlock(base, base * 2, stride=2)
        self.down2 = _ConvBlock(base * 2, base * 4, stride=2)
        self.down3 = _ConvBlock(base * 4, base * 8, stride=2)

        self.up1 = nn.ConvTranspose2d(base * 8, base * 4, 4, stride=2, padding=1)
        self.dec1 = _ConvBlock(base * 8, base * 4, stride=1)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = _ConvBlock(base * 4, base * 2, stride=1)
        self.up3 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec3 = _ConvBlock(base * 2, base, stride=1)
        self.final = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, cond, z=None):
        e0 = self.enc0(cond)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)

        d1 = self.up1(e3)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, e0], dim=1))
        return torch.tanh(self.final(d3))


# --------------------------------------------------------------------------- discriminator

class CondPatchDiscriminator(nn.Module):
    """PatchGAN that also sees the requested text.

    Image + cond + text-embedding means the critic scores "does this look like handwriting of
    THIS text", not just "does this look like handwriting". Feeding mismatched (image, text)
    pairs with a fake label teaches the discriminator that correspondence matters.

    Stability (from BigGAN / PatchGAN practice):
    - spectral norm on every conv (keeps Lipschitz 1, prevents D explosion)
    - InstanceNorm is deliberately NOT used here: per-instance stats leak real/fake
    - LeakyReLU 0.2 throughout, no dropout (keeps gradients flowing to G)
    - optional writer head: aux CE on the real writer id (AC-GAN style) — forces the
      generator to actually encode writer identity, not ignore the FiLM path.
    """

    def __init__(self, cond_ch=16, text_dim=256, text_ch=16, base=32, use_sn=True,
                 with_cond=True, n_writers=0):
        super().__init__()
        self.cond_ch = int(cond_ch) if with_cond else 0
        self.text_dim = text_dim
        self.text_ch = text_ch
        self.n_writers = int(n_writers or 0)
        self.base = base
        in_ch = 1 + self.cond_ch + text_ch
        self.in_ch = in_ch

        def sn(conv):
            return nn.utils.parametrizations.spectral_norm(conv) if use_sn else conv

        self.c1 = sn(nn.Conv2d(in_ch, base, 4, stride=2, padding=1))
        self.c2 = sn(nn.Conv2d(base, base * 2, 4, stride=2, padding=1))
        self.c3 = sn(nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1))
        self.c4 = sn(nn.Conv2d(base * 4, base * 8, 4, stride=1, padding=1))
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_ch * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(text_ch * 2, text_ch),
        )
        c_out = base * 8
        c3_out = base * 4
        self.writer_head = nn.Sequential(
            nn.Linear(c_out + c3_out, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.n_writers),
        ) if self.n_writers > 0 else None

    def forward(self, image, cond=None, text_feat=None, return_features=False,
                return_writer=False):
        b, _, h, w = image.shape
        if text_feat is None:
            text_feat = image.new_zeros(b, self.text_dim)
        tmap = self.text_proj(text_feat.to(image.dtype)).view(b, self.text_ch, 1, 1)
        tmap = tmap.expand(b, self.text_ch, h, w)
        parts = [image, tmap]
        if self.cond_ch and cond is not None:
            parts.insert(1, cond[:, :self.cond_ch])
        elif self.cond_ch:
            parts.insert(1, image.new_zeros(b, self.cond_ch, h, w))
        x = torch.cat(parts, dim=1)

        f1 = self.act(self.c1(x))
        f2 = self.act(self.c2(f1))
        f3 = self.act(self.c3(f2))
        out = self.c4(f3)  # (B, base*8, H_out, W_out)
        if return_writer and self.writer_head is not None:
            pooled = out.mean(dim=(2, 3))  # (B, base*8)
            # also pool features for robustness: concat f3 pooled
            if f3 is not None:
                pooled = torch.cat([pooled, f3.mean(dim=(2, 3))], dim=1)
            writer_logits = self.writer_head(pooled)
            if return_features:
                return out, [f1, f2, f3, out], writer_logits
            return out, writer_logits
        if return_features:
            return out, [f1, f2, f3, out]
        return out


# --------------------------------------------------------------------------- losses

def shift_l1(fake, real, max_shift=4):
    """Alignment-tolerant reconstruction loss.

    Generated and real handwriting contain the same word but not at the same pixel offsets,
    so a strict pixel L1 punishes placement rather than content. Take the best of a small
    window of horizontal shifts, per sample. max_shift=0 gives plain L1.
    """
    if max_shift <= 0:
        return F.l1_loss(fake, real)
    best = None
    for s in range(-int(max_shift), int(max_shift) + 1):
        if s == 0:
            a, b = fake, real
        elif s > 0:                      # generated shifted right vs real
            a, b = fake[..., s:], real[..., :-s]
        else:                            # generated shifted left vs real
            a, b = fake[..., :s], real[..., -s:]
        v = (a - b).abs().mean(dim=(1, 2, 3))
        best = v if best is None else torch.minimum(best, v)
    return best.mean()


def feature_matching_loss(f_real, f_fake, weights=None):
    """L1 between discriminator intermediate features — realism without pixel alignment."""
    if weights is None:
        weights = [1.0] * len(f_real)
    total = 0.0
    for w, a, b in zip(weights, f_real, f_fake):
        total = total + w * F.l1_loss(b, a.detach())
    return total / max(1e-6, sum(weights))


def text_feature(enc, mask):
    """Masked mean of the character features -> (B, D). Used to condition the critic."""
    m = mask.unsqueeze(-1).to(enc.dtype)
    return (enc * m).sum(1) / m.sum(1).clamp(min=1.0)


# --------------------------------------------------------------------------- factory

def build_v2(height, width, vocab, cond_ch=16, base=32, style_dim=8, text_hidden=128,
             text_embed=64, d_text_ch=16, n_writers=0, writer_cond=False, use_sn=True):
    """Return (text_encoder, conditioner, generator, discriminator) for a v2 checkpoint."""
    enc = TextEncoder(vocab, embed_dim=text_embed, hidden=text_hidden)
    cond = TextCondition(enc.out_dim, cond_ch=cond_ch, height=height, width=width,
                         style_dim=style_dim, n_writers=n_writers, writer_cond=writer_cond)
    G = UNetGeneratorV2(cond_ch=cond_ch, base=base)
    D = CondPatchDiscriminator(cond_ch=cond_ch, text_dim=enc.out_dim, text_ch=d_text_ch,
                               base=base, use_sn=use_sn, n_writers=n_writers if writer_cond else 0)
    return enc, cond, G, D
