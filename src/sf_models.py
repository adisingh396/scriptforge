"""Neural-network modules for ScriptForge."""

import torch
import torch.nn as nn
import torch.nn.functional as F

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "


def build_charset():
    return CHARSET


def text_to_ids(text, charset=CHARSET):
    idx = {c: i for i, c in enumerate(charset)}
    return [idx[c] for c in str(text).upper() if c in idx]


def ids_to_text(ids, charset=CHARSET):
    out = []
    blank = len(charset)
    for i in ids:
        i = int(i)
        if i == blank:
            continue
        if 0 <= i < len(charset):
            out.append(charset[i])
    return "".join(out)


def greedy_ctc_decode(logits, charset=CHARSET):
    """Decode (B, T, C) logits. The layout is explicit — no guessing."""
    t = logits.detach().cpu() if isinstance(logits, torch.Tensor) else torch.as_tensor(logits).cpu()
    assert t.dim() == 3, f"expected (B,T,C) logits, got shape {tuple(t.shape)}"
    assert t.shape[-1] == len(charset) + 1, \
        f"last dim {t.shape[-1]} != classes+blank {len(charset) + 1} (wrong tensor layout?)"
    blank = len(charset)
    results = []
    for b in range(t.shape[0]):
        prev = -1
        chars = []
        for step in range(t.shape[1]):
            idx = int(t[b, step].argmax().item())
            if idx != blank and idx != prev:
                if 0 <= idx < len(charset):
                    chars.append(charset[idx])
            prev = idx
        results.append("".join(chars))
    return results


def crnn_log_probs(images, model):
    logits = model(images)
    return F.log_softmax(logits, dim=-1)


class StyleEncoder(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.fc = nn.Linear(z_dim, 8 * 6 * 24)

    def forward(self, z, height, width):
        b = z.shape[0]
        h = self.fc(z).view(b, 8, 6, 24)
        out = F.interpolate(h, size=(height, width), mode="nearest")
        return out


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UNetGenerator(nn.Module):
    def __init__(self, cond_ch=2, style_ch=8, base=48):
        super().__init__()
        self.cond_ch = cond_ch
        self.style_ch = style_ch
        self.base = base
        in_ch = cond_ch + style_ch
        self.enc0 = _ConvBlock(in_ch, base, stride=1)
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

    def forward(self, cond, style_map):
        if cond.shape[1] != self.cond_ch:
            cond = cond[:, :self.cond_ch]
        x = torch.cat([cond, style_map], dim=1)
        e0 = self.enc0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)

        d1 = self.up1(e3)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e0], dim=1)
        d3 = self.dec3(d3)
        return torch.tanh(self.final(d3))


class PatchDiscriminator(nn.Module):
    def __init__(self, cond_ch=2, base=48, use_sn=True):
        super().__init__()
        self.cond_ch = cond_ch
        self.base = base
        in_ch = 1 + cond_ch
        self.in_ch = in_ch

        def maybe_sn(conv):
            # NOTE: spectral norm + fp16 autocast can drive the weights to ~0, after which
            # the w/sigma parametrization pins them there (D collapses to a constant).
            # Keep it optional and prefer R1 regularization in fp32.
            return nn.utils.parametrizations.spectral_norm(conv) if use_sn else conv

        self.c1 = maybe_sn(nn.Conv2d(in_ch, base, 4, stride=2, padding=1))
        self.c2 = maybe_sn(nn.Conv2d(base, base * 2, 4, stride=2, padding=1))
        self.c3 = maybe_sn(nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1))
        self.c4 = maybe_sn(nn.Conv2d(base * 4, base * 8, 4, stride=1, padding=1))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, image, cond=None):
        if cond is not None:
            x = torch.cat([image, cond], dim=1)
        else:
            x = image
        if x.shape[1] != self.in_ch:
            if x.shape[1] > self.in_ch:
                x = x[:, :self.in_ch]
            else:
                # pad channels
                pad = self.in_ch - x.shape[1]
                x = torch.cat([x, x[:, :1].repeat(1, pad, 1, 1)], dim=1)
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.act(self.c3(x))
        x = self.c4(x)
        return x


class CRNN(nn.Module):
    def __init__(self, num_classes, height=48, width=192):
        super().__init__()
        self.num_classes = num_classes
        self.height = height
        self.width = width
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.lstm1 = nn.LSTM(256, 256, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes + 1)

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] != 1:
            x = x[:, :1]
        feat = self.cnn(x)
        # height-agnostic: collapse whatever height remains (40px, 48px, ...) into 1 row
        feat = F.adaptive_avg_pool2d(feat, (1, None))
        feat = feat.squeeze(2)
        feat = feat.permute(0, 2, 1)
        out, _ = self.lstm1(feat)
        out, _ = self.lstm2(out)
        logits = self.fc(out)
        return logits
