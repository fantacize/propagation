"""Word embeddings via skip-gram-style backprop on a tiny corpus.

Implements the math from the project worksheet:
  forward:  z = u . v ;  yhat = sigmoid(z)
  loss:     L = -y log yhat - (1-y) log (1-yhat)
  grad:     dL/dz = yhat - y
  update:   w_new = w - eta * (yhat - y) * partner

Positive pairs = co-occurring words within the same sentence (stopword 'the'
removed). Negative pairs = random word pairs not seen in the corpus.
"""

import numpy as np
import matplotlib.pyplot as plt

CORPUS = [
    "the cat sleeps",
    "the dog sleeps",
    "the bird sleeps",
    "the cat runs",
    "the dog runs",
    "the bird flies",
    "the student studies",
    "the teacher teaches",
    "the student learns",
    "the dog chases the cat",
    "the cat chases the bird",
    "the teacher teaches the student",
]
STOPWORDS = {"the"}


def tokens(line):
    return [w for w in line.split() if w not in STOPWORDS]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cos(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def build_vocab(corpus):
    vocab = sorted({w for line in corpus for w in tokens(line)})
    return vocab, {w: i for i, w in enumerate(vocab)}


def build_pairs(corpus):
    """All ordered (a,b) pairs of distinct words within the same sentence."""
    pairs = []
    for line in corpus:
        toks = tokens(line)
        for i, a in enumerate(toks):
            for j, b in enumerate(toks):
                if i != j:
                    pairs.append((a, b))
    return pairs


def train(corpus=CORPUS, dim=2, epochs=2000, lr=0.1, neg_per_pos=2, seed=42):
    vocab, w2i = build_vocab(corpus)
    V = len(vocab)
    rng = np.random.default_rng(seed)

    # Init: nonzero, components in [-1, 1].
    E = rng.uniform(-1.0, 1.0, size=(V, dim))
    tiny = np.linalg.norm(E, axis=1) < 1e-3
    E[tiny] = rng.uniform(0.2, 1.0, size=(tiny.sum(), dim))

    positives = build_pairs(corpus)
    pos_set = set(positives)

    def neg_sample():
        while True:
            a = vocab[rng.integers(V)]
            b = vocab[rng.integers(V)]
            if a != b and (a, b) not in pos_set:
                return a, b

    losses = []
    for epoch in range(epochs):
        total = 0.0
        for k in rng.permutation(len(positives)):
            a, b = positives[k]
            ia, ib = w2i[a], w2i[b]
            va, vb = E[ia].copy(), E[ib].copy()
            z = float(np.dot(va, vb))
            yhat = sigmoid(z)
            total += -np.log(yhat + 1e-12)
            g = yhat - 1.0
            E[ia] = va - lr * g * vb
            E[ib] = vb - lr * g * va

            for _ in range(neg_per_pos):
                na, nb = neg_sample()
                ina, inb = w2i[na], w2i[nb]
                vna, vnb = E[ina].copy(), E[inb].copy()
                zn = float(np.dot(vna, vnb))
                yhn = sigmoid(zn)
                total += -np.log(1.0 - yhn + 1e-12)
                gn = yhn - 0.0
                E[ina] = vna - lr * gn * vnb
                E[inb] = vnb - lr * gn * vna

        losses.append(total)
        if (epoch + 1) % 200 == 0:
            print(f"epoch {epoch+1:5d}  loss={total:.4f}")

    return vocab, w2i, E, losses


def plot_embeddings(vocab, w2i, E, path="embeddings.png"):
    fig, ax = plt.subplots(figsize=(9, 9))
    for w in vocab:
        v = E[w2i[w]]
        ax.arrow(
            0, 0, v[0], v[1],
            head_width=0.05, length_includes_head=True,
            color="steelblue", alpha=0.65,
        )
        ax.text(v[0] * 1.06, v[1] * 1.06, w, fontsize=12)
    mx = float(np.max(np.abs(E))) * 1.25
    ax.set_xlim(-mx, mx)
    ax.set_ylim(-mx, mx)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal")
    ax.set_title("Trained 2D word embeddings")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    print(f"saved {path}")


def print_similarities(vocab, w2i, E):
    print("\nPairwise cosine similarities:")
    rows = []
    for i, w1 in enumerate(vocab):
        for w2 in vocab[i + 1:]:
            rows.append((w1, w2, cos(E[w2i[w1]], E[w2i[w2]])))
    rows.sort(key=lambda r: -r[2])
    for w1, w2, c in rows:
        print(f"  {w1:8s} ~ {w2:8s} : {c:+.3f}")


def manual_demo():
    """Reproduce the by-hand calculations from Q2-Q5."""
    print("=== Manual demo: cat sleeps and cat teaches ===")
    cat = np.array([1.0, 1.0])
    sleeps = np.array([0.1, -0.1])
    teaches = np.array([-0.5, -0.5])

    for name, partner, y in [("cat-sleeps", sleeps, 1), ("cat-teaches", teaches, 0)]:
        z = float(np.dot(cat, partner))
        yhat = sigmoid(z)
        L = -y * np.log(yhat + 1e-12) - (1 - y) * np.log(1 - yhat + 1e-12)
        g = yhat - y
        grad_cat = g * partner
        grad_partner = g * cat
        cat_new = cat - 0.1 * grad_cat
        partner_new = partner - 0.1 * grad_partner
        print(f"\n[{name}] y={y}")
        print(f"  z={z:+.4f}  yhat={yhat:.4f}  L={L:.4f}  dL/dz={g:+.4f}")
        print(f"  grad cat     = {grad_cat}")
        print(f"  grad partner = {grad_partner}")
        print(f"  cat_new      = {cat_new}")
        print(f"  partner_new  = {partner_new}")


if __name__ == "__main__":
    manual_demo()
    print("\n=== Training on expanded corpus ===")
    vocab, w2i, E, losses = train()
    print("\nFinal embeddings:")
    for w in vocab:
        print(f"  {w:10s} = {E[w2i[w]]}")
    print_similarities(vocab, w2i, E)
    plot_embeddings(vocab, w2i, E)
