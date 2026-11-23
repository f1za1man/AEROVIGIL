"""Small LSTM models (CPU friendly): a one-class autoencoder and a supervised classifier."""

import copy
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMAutoencoder(nn.Module):
    """Encoder LSTM -> latent vector -> decoder LSTM. Trained on NORMAL windows only."""

    def __init__(self, n_features, hidden=32, layers=1, latent=16, dropout=0.0):
        super().__init__()
        inner = dropout if layers > 1 else 0.0
        self.enc = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=inner)
        self.to_z = nn.Linear(hidden, latent)
        self.drop = nn.Dropout(dropout)
        self.from_z = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=inner)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.drop(self.to_z(h[-1]))
        d = self.from_z(z).unsqueeze(1).expand(-1, x.shape[1], -1).contiguous()
        y, _ = self.dec(d)
        return self.out(y)


class LSTMClassifier(nn.Module):
    """LSTM over the window, then one logit: attack (1) or normal (0)."""

    def __init__(self, n_features, hidden=32, layers=1, dropout=0.1):
        super().__init__()
        inner = dropout if layers > 1 else 0.0
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=inner)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(self.drop(h[-1])).squeeze(-1)


def build_model(kind, n_features, params, seed=None):
    """Build a model. Pass `seed` so the initial weights are reproducible."""
    if seed is not None:
        set_seed(seed)
    if kind == "ae":
        return LSTMAutoencoder(
            n_features,
            params["hidden"],
            params["layers"],
            params["latent"],
            params.get("dropout", 0.0),
        )
    if kind == "clf":
        return LSTMClassifier(
            n_features, params["hidden"], params["layers"], params.get("dropout", 0.1)
        )
    raise ValueError(f"Unknown model kind: {kind}")


def _loss_fn(kind, y_train):
    if kind == "ae":
        return lambda out, x, y: ((out - x) ** 2).mean()
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    weight = torch.tensor(neg / max(pos, 1.0))
    bce = nn.BCEWithLogitsLoss(pos_weight=weight)
    return lambda out, x, y: bce(out, y)


def fit(
    kind, model, X_train, y_train, X_val, y_val, params, seed, max_epochs, patience
):
    """Train with Adam and early stopping on the validation loss. Returns the best model."""
    set_seed(seed)
    lossf = _loss_fn(kind, y_train)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params.get("weight_decay", 1e-5),
    )
    xt = torch.from_numpy(X_train)
    xv = torch.from_numpy(X_val)
    yt = torch.from_numpy(y_train.astype(np.float32)) if y_train is not None else None
    yv = torch.from_numpy(y_val.astype(np.float32)) if y_val is not None else None
    gen = torch.Generator().manual_seed(seed)
    bs = int(params.get("batch_size", 256))
    best, best_state, bad, hist = np.inf, None, 0, {"train": [], "val": []}
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        total = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i : i + bs]
            opt.zero_grad()
            out = model(xt[b])
            loss = lossf(out, xt[b], yt[b] if yt is not None else None)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params.get("grad_clip", 1.0))
            opt.step()
            total += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, len(xv), 1024):
                out = model(xv[i : i + 1024])
                vl += float(
                    lossf(
                        out,
                        xv[i : i + 1024],
                        yv[i : i + 1024] if yv is not None else None,
                    )
                ) * len(xv[i : i + 1024])
            vl /= max(len(xv), 1)
        hist["train"].append(total / len(xt))
        hist["val"].append(vl)
        if vl < best - 1e-6:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    hist["best_epoch"] = int(np.argmin(hist["val"]))
    hist["best_val"] = float(best)
    return model, hist


@torch.no_grad()
def predict(kind, model, X, batch=1024):
    """AE: (window score = mean squared error, per-feature error). Classifier: logits."""
    model.eval()
    xt = torch.from_numpy(X)
    if kind == "ae":
        errs = []
        for i in range(0, len(xt), batch):
            b = xt[i : i + batch]
            errs.append(((model(b) - b) ** 2).mean(dim=1).numpy())
        fe = (
            np.concatenate(errs)
            if errs
            else np.empty((0, X.shape[-1]), dtype=np.float32)
        )
        return fe.mean(axis=1), fe
    out = [model(xt[i : i + batch]).numpy() for i in range(0, len(xt), batch)]
    return (np.concatenate(out) if out else np.empty(0, dtype=np.float32)), None
