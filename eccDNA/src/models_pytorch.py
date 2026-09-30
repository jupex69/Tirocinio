"""3 architetture PyTorch per il confronto sano/malato, a complessita' crescente
e con meccanismi diversi (non solo varianti piu' profonde della stessa rete):

- DeepMLP: classificatore feed-forward diretto (BatchNorm + Dropout).
- AttentionGatedMLP: come DeepMLP, ma con un gate (sigmoid) che impara a
  pesare le 13 feature per ciascun campione prima dell'MLP - i pesi di
  attenzione sono anche interpretabili (quali descrittori contano di piu').
- SiameseEncoder: non classifica direttamente, impara un embedding via
  triplet loss (ancore/positivi/negativi campionati con bilanciamento
  anti-bias tra malattie) e classifica per prototipo (centroide sano vs
  centroide malato nello spazio embedding) - un paradigma di metric learning,
  non discriminativo diretto.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def _make_mlp_backbone(n_features, hidden, dropout, out_dim):
    layers = []
    prev = n_features
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class DeepMLP(nn.Module):
    def __init__(self, n_features, hidden=(64, 32, 16), dropout=0.3):
        super().__init__()
        self.net = _make_mlp_backbone(n_features, hidden, dropout, out_dim=1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AttentionGatedMLP(nn.Module):
    def __init__(self, n_features, hidden=(64, 32, 16), dropout=0.3):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(n_features, n_features), nn.Tanh(),
            nn.Linear(n_features, n_features), nn.Sigmoid(),
        )
        self.net = _make_mlp_backbone(n_features, hidden, dropout, out_dim=1)

    def forward(self, x, return_attention=False):
        gate = self.attn(x)
        out = self.net(x * gate).squeeze(-1)
        if return_attention:
            return out, gate
        return out


class SiameseEncoder(nn.Module):
    def __init__(self, n_features, hidden=(32,), embedding_dim=16, dropout=0.2):
        super().__init__()
        self.net = _make_mlp_backbone(n_features, hidden, dropout, out_dim=embedding_dim)

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


class SiameseWithHead(nn.Module):
    """Encoder siamese + testa di classificazione lineare. L'embedding e'
    modellato da una triplet loss (batch-hard) e contemporaneamente ottimizzato
    da una BCE tramite la testa: la componente contrastiva da' struttura metrica
    allo spazio (campioni simili vicini), la testa spinge direttamente sul
    confine sano/malato. E' l'architettura siamese piu' forte e legittima per
    questo problema, dove il metric learning puro (solo embedding + prototipo)
    resta limitato dalle poche feature scalari.

    L'embedding NON e' L2-normalizzato qui (a differenza di SiameseEncoder):
    la testa lineare sfrutta anche la magnitudine, non solo la direzione.
    """
    def __init__(self, n_features, hidden=(64, 32), embedding_dim=32, dropout=0.3):
        super().__init__()
        self.encoder = _make_mlp_backbone(n_features, hidden, dropout, out_dim=embedding_dim)
        self.head = nn.Linear(embedding_dim, 1)

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)


def _to_tensor(X, device):
    return torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)


def train_binary_classifier(model, X_train, y_train, X_val, y_val, epochs=200, lr=1e-3,
                             batch_size=512, patience=15, device="cpu", seed=42):
    """Training standard per DeepMLP/AttentionGatedMLP: BCEWithLogitsLoss (con
    pos_weight per lo sbilanciamento), Adam, early stopping su AUC di validation
    (si tiene il miglior stato del modello, non l'ultimo epoch)."""
    torch.manual_seed(seed)
    model.to(device)
    Xt, yt = _to_tensor(X_train, device), _to_tensor(y_train, device)
    Xv, yv = _to_tensor(X_val, device), np.asarray(y_val)

    n_pos, n_neg = (yt == 1).sum().item(), (yt == 0).sum().item()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = Xt.shape[0]
    best_auc, best_state, epochs_no_improve = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(Xt[idx])
            loss = criterion(logits, yt[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv).cpu().numpy()
        val_auc = roc_auc_score(yv, val_logits)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


@torch.no_grad()
def predict_proba(model, X, device="cpu"):
    model.eval()
    logits = model(_to_tensor(X, device))
    return torch.sigmoid(logits).cpu().numpy()


def _sample_triplet_indices(y, disease, rng, n):
    """Campiona n triplette (anchor, positive, negative) bilanciate: meta' con
    ancora malata, meta' con ancora sana. Tra i malati, la malattia specifica
    viene scelta uniformemente PRIMA dell'id (logica anti-bias: senza, il
    sottotipo piu' numeroso dominerebbe le triplette e la rete imparerebbe
    quello, non 'malato' in generale).
    """
    sano_idx = np.where(y == 0)[0]
    malato_idx = np.where(y == 1)[0]
    disease_malato = np.asarray(disease)[malato_idx]
    diseases = np.unique(disease_malato)
    idx_by_disease = {d: malato_idx[disease_malato == d] for d in diseases}

    def pick_malato():
        d = diseases[rng.integers(len(diseases))]
        pool = idx_by_disease[d]
        return pool[rng.integers(len(pool))]

    anchors, positives, negatives = [], [], []
    half = n // 2

    for _ in range(half):
        a = pick_malato()
        p = pick_malato()
        tries = 0
        while p == a and tries < 5:
            p = pick_malato()
            tries += 1
        neg = sano_idx[rng.integers(len(sano_idx))]
        anchors.append(a); positives.append(p); negatives.append(neg)

    for _ in range(n - half):
        a = sano_idx[rng.integers(len(sano_idx))]
        p = sano_idx[rng.integers(len(sano_idx))]
        tries = 0
        while p == a and tries < 5:
            p = sano_idx[rng.integers(len(sano_idx))]
            tries += 1
        neg = pick_malato()
        anchors.append(a); positives.append(p); negatives.append(neg)

    return np.array(anchors), np.array(positives), np.array(negatives)


def _pk_sample(class_labels, rng, n_classes, n_per_class):
    """Campionamento PK per il batch-hard mining: sceglie n_classes classi e
    n_per_class campioni per classe. Le classi sono le 18 (17 malattie + Sano):
    trattare ogni malattia come classe separata (non un unico 'malato') e' cio'
    che permette al mining di trovare triplette difficili sensate e sfrutta
    l'eterogeneita' invece di subirla. Ritorna gli indici del batch."""
    classi = np.unique(class_labels)
    n_classes = min(n_classes, len(classi))
    scelte = rng.choice(classi, size=n_classes, replace=False)
    idx = []
    for c in scelte:
        pool = np.where(class_labels == c)[0]
        repl = len(pool) < n_per_class
        idx.extend(rng.choice(pool, size=n_per_class, replace=repl))
    return np.array(idx)


def _batch_hard_triplet_loss(embeddings, labels, margin):
    """Triplet loss batch-hard (Hermans et al. 2017): per ogni ancora del batch
    prende il positivo PIU' LONTANO della stessa classe e il negativo PIU'
    VICINO di classe diversa - le triplette piu' informative, le uniche che
    danno gradiente utile. Risolve il fallimento delle triplette casuali (quasi
    sempre gia' facili -> gradiente ~0 -> l'encoder non impara)."""
    # matrice delle distanze euclidee a coppie nel batch
    dist = torch.cdist(embeddings, embeddings, p=2)
    labels = torch.as_tensor(labels, device=embeddings.device)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=embeddings.device)

    # positivo piu' lontano (stessa classe, escluso se stesso)
    pos_mask = same & ~eye
    dist_pos = dist.masked_fill(~pos_mask, float("-inf"))
    hardest_pos = dist_pos.max(dim=1).values

    # negativo piu' vicino (classe diversa)
    dist_neg = dist.masked_fill(same, float("inf"))
    hardest_neg = dist_neg.min(dim=1).values

    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    if valid.sum() == 0:
        return embeddings.sum() * 0.0  # nessuna tripletta valida in questo batch
    loss = torch.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return loss.mean()


def train_siamese_combined(model, X_train, y_train, X_val, y_val, epochs=200,
                           batch_size=256, lr=1e-3, margin=1.0, triplet_weight=0.5,
                           patience=20, device="cpu", seed=42):
    """Training della siamese con loss combinata: BCE (tramite la testa) +
    triplet batch-hard binaria (sull'embedding). La componente contrastiva
    struttura lo spazio, la BCE ottimizza direttamente il confine sano/malato.
    Early stopping su AUC di validation (probabilita' della testa). Ritorna
    (model, best_auc)."""
    torch.manual_seed(seed)
    model.to(device)
    Xt, yt = _to_tensor(X_train, device), _to_tensor(y_train, device)
    Xv, yv = _to_tensor(X_val, device), np.asarray(y_val)

    n_pos, n_neg = (yt == 1).sum().item(), (yt == 0).sum().item()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = Xt.shape[0]
    best_auc, best_state, epochs_no_improve = -1.0, None, 0
    y_codes = yt.long().cpu().numpy()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            emb = model.embed(Xt[idx])
            logits = model.head(emb).squeeze(-1)
            loss = bce(logits, yt[idx]) + triplet_weight * _batch_hard_triplet_loss(
                emb, y_codes[idx.cpu().numpy()], margin
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv).cpu().numpy()
        val_auc = roc_auc_score(yv, val_logits)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def train_siamese_batch_hard(encoder, X_train, class_train, y_train, X_val, y_val,
                             epochs=200, batches_per_epoch=40, n_classes=12, n_per_class=16,
                             lr=1e-3, margin=0.3, patience=20, device="cpu", seed=42):
    """Training della siamese con batch-hard mining su 18 classi (17 malattie +
    Sano). All'inferenza si classifica per prototipo piu' vicino
    (multi_prototype_score). Sostituisce le triplette casuali di train_siamese,
    che sulla task bilanciata (piu' piccola e piu' difficile) diventavano
    non-informative e facevano collassare l'AUC a ~0.5.

    class_train: etichetta a 18 classi ('Healthy' oppure il nome della malattia).
    y_train/y_val: etichetta binaria 0/1, usata solo per l'AUC di validation.
    Ritorna (encoder, centroid_healthy, disease_centroids, best_auc).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    encoder.to(device)
    Xt = _to_tensor(X_train, device)
    class_train = np.asarray(class_train)
    # le classi sono stringhe (nomi malattia / 'Healthy'): il PK-sampling e la
    # triplet loss lavorano su codici interi
    class_codes = pd.factorize(class_train)[0]
    yv = np.asarray(y_val)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    best_auc, best_state, epochs_no_improve = -1.0, None, 0

    for epoch in range(epochs):
        encoder.train()
        for _ in range(batches_per_epoch):
            idx = _pk_sample(class_codes, rng, n_classes, n_per_class)
            optimizer.zero_grad()
            emb = encoder(Xt[idx])
            loss = _batch_hard_triplet_loss(emb, class_codes[idx], margin)
            loss.backward()
            optimizer.step()

        ch, dc = compute_multi_prototypes(encoder, X_train, y_train, class_train, device)
        val_scores = multi_prototype_score(encoder, X_val, ch, dc, device)
        val_auc = roc_auc_score(yv, val_scores)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    ch, dc = compute_multi_prototypes(encoder, X_train, y_train, class_train, device)
    return encoder, ch, dc, best_auc


def compute_prototypes(encoder, X_train, y_train, device="cpu"):
    """Centroide (media delle embedding L2-normalizzate) per sano e malato nel
    train set - il classificatore del prototipo confronta ogni nuovo embedding
    a questi due punti fissi, invece di avere un head di classificazione."""
    with torch.no_grad():
        emb = encoder(_to_tensor(X_train, device)).cpu().numpy()
    y_train = np.asarray(y_train)
    centroid_healthy = emb[y_train == 0].mean(axis=0)
    centroid_disease = emb[y_train == 1].mean(axis=0)
    return centroid_healthy, centroid_disease


@torch.no_grad()
def prototype_score(encoder, X, centroid_healthy, centroid_disease, device="cpu"):
    """Punteggio continuo (piu' alto = piu' simile al prototipo malato) da
    usare per soglia/ROC-AUC: differenza di similarita' coseno ai due
    centroidi (le embedding sono gia' L2-normalizzate, quindi il prodotto
    scalare e' gia' una similarita' coseno)."""
    encoder.eval()
    emb = encoder(_to_tensor(X, device)).cpu().numpy()
    sim_disease = emb @ centroid_disease
    sim_healthy = emb @ centroid_healthy
    return sim_disease - sim_healthy


def compute_multi_prototypes(encoder, X_train, y_train, disease_train, device="cpu", min_per_disease=20):
    """Variante multi-prototipo: un centroide per CIASCUNA malattia specifica
    (non uno solo per 'malato') piu' un centroide sano.

    Motivo: schiacciare 17 malattie biologicamente diverse in un unico
    centroide 'malato' e' la debolezza della versione a 2 prototipi (la
    piu' debole in AUC nel confronto). Con un prototipo per malattia, una
    sequenza viene giudicata malata se somiglia a UNA QUALSIASI malattia
    (il suo cluster piu' vicino tra quelli di malattia), sfruttando
    l'eterogeneita' delle classi invece di subirla.

    Ritorna (centroid_healthy, disease_centroids) dove disease_centroids e'
    un dict {nome_malattia: centroide}. Le malattie con meno di
    min_per_disease campioni nel train vengono saltate (centroide instabile).
    """
    with torch.no_grad():
        emb = encoder(_to_tensor(X_train, device)).cpu().numpy()
    y_train = np.asarray(y_train)
    disease_train = np.asarray(disease_train)

    centroid_healthy = emb[y_train == 0].mean(axis=0)
    disease_centroids = {}
    for malattia in np.unique(disease_train[y_train == 1]):
        mask = (y_train == 1) & (disease_train == malattia)
        if mask.sum() >= min_per_disease:
            disease_centroids[malattia] = emb[mask].mean(axis=0)
    return centroid_healthy, disease_centroids


@torch.no_grad()
def multi_prototype_score(encoder, X, centroid_healthy, disease_centroids, device="cpu"):
    """Punteggio continuo multi-prototipo: (similarita' coseno al centroide di
    malattia PIU' VICINO) - (similarita' al centroide sano). Piu' alto = piu'
    vicino ad almeno un cluster di malattia rispetto al sano.

    Il max sulle malattie e' il cuore dell'idea: non serve somigliare alla
    'malattia media', basta somigliare a una qualunque delle malattie note.
    """
    encoder.eval()
    emb = encoder(_to_tensor(X, device)).cpu().numpy()
    disease_matrix = np.stack(list(disease_centroids.values()))  # (n_malattie, dim)
    sim_disease_max = (emb @ disease_matrix.T).max(axis=1)
    sim_healthy = emb @ centroid_healthy
    return sim_disease_max - sim_healthy


def train_siamese(encoder, X_train, y_train, disease_train, X_val, y_val, epochs=200,
                   triplets_per_epoch=4000, lr=1e-3, batch_size=256, margin=0.3,
                   patience=15, device="cpu", seed=42, multi_prototype=False):
    """Training con triplet loss, triplette ricampionate a ogni epoch (non
    precalcolate/salvate: con soli ~18 valori scalari per riga il
    ricampionamento e' economico e da' varieta' extra rispetto a un insieme
    di triplette fisso). Early stopping sull'AUC di validation della
    classificazione per prototipo.

    multi_prototype=False (default): un solo centroide 'malato' vs uno 'sano'.
    multi_prototype=True: un centroide per ciascuna malattia + uno sano, e il
    punteggio usa il centroide di malattia PIU' VICINO (vedi
    compute_multi_prototypes/multi_prototype_score). L'encoder e le triplette
    sono identici tra le due varianti: cambia solo come le embedding apprese
    vengono trasformate in un punteggio sano/malato.

    Ritorna (encoder, centroid_healthy, disease_proto, best_auc), dove
    disease_proto e' un singolo centroide (2-proto) o un dict
    {malattia: centroide} (multi-proto) - il chiamante usa la funzione di
    scoring corrispondente (prototype_score o multi_prototype_score).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    encoder.to(device)
    Xt = _to_tensor(X_train, device)
    y_train_arr = np.asarray(y_train)
    disease_train_arr = np.asarray(disease_train)
    yv = np.asarray(y_val)

    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    def _val_auc():
        if multi_prototype:
            ch, dc = compute_multi_prototypes(encoder, X_train, y_train, disease_train, device)
            scores = multi_prototype_score(encoder, X_val, ch, dc, device)
        else:
            ch, cd = compute_prototypes(encoder, X_train, y_train, device)
            scores = prototype_score(encoder, X_val, ch, cd, device)
        return roc_auc_score(yv, scores)

    best_auc, best_state, epochs_no_improve = -1.0, None, 0

    for epoch in range(epochs):
        encoder.train()
        a_idx, p_idx, n_idx = _sample_triplet_indices(y_train_arr, disease_train_arr, rng, triplets_per_epoch)
        perm = rng.permutation(len(a_idx))
        for start in range(0, len(perm), batch_size):
            batch = perm[start:start + batch_size]
            optimizer.zero_grad()
            emb_a = encoder(Xt[a_idx[batch]])
            emb_p = encoder(Xt[p_idx[batch]])
            emb_n = encoder(Xt[n_idx[batch]])
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

        val_auc = _val_auc()
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state)

    if multi_prototype:
        centroid_healthy, disease_proto = compute_multi_prototypes(encoder, X_train, y_train, disease_train, device)
    else:
        centroid_healthy, disease_proto = compute_prototypes(encoder, X_train, y_train, device)
    return encoder, centroid_healthy, disease_proto, best_auc
