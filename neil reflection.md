### (c) Reflection

**i. Why do we require nonzero initial choices?**

Two reasons. First, if both words in a pair start at the zero vector, then $z=0$, $\hat{y}=0.5$, and both gradients are multiplied by a zero partner vector, so neither embedding moves. If only one word is zero, the zero vector can still move because its gradient uses the nonzero partner, but the partner gets no update from that pair because its gradient is multiplied by zero. So nonzero initialization lets every pair update both words from the start. Second, the initial vectors should be different from each other. If many words started identical, they'd receive identical gradients across the same positive/negative pairs and never differentiate from each other. Random, nonzero initial choices get learning started and break the symmetry between words.

**ii. Why are embeddings helpful?**

Words on their own are just labels, you can't subtract, average, or take derivatives of them. Embeddings turn each word into a vector in $\mathbb{R}^d$, which is a space where math works. Once a word is a vector you can measure similarity (dot product), do gradient descent, and feed the word into any neural network layer that expects numbers. They also give you continuity: the model can learn that small changes in meaning correspond to small changes in vector position, which lets it generalize from words it's seen a lot to words it's seen less.

**iii. What is backprop doing, conceptually?**

Backprop is *blame assignment by chain rule*. The loss at the output tells you how wrong the whole computation was. Backprop walks backward through every operation that produced the loss and uses the chain rule to figure out how much each parameter contributed to the error. Each parameter then takes a small step opposite its gradient, which is the direction that locally reduces the loss the fastest. For embeddings: when a positive pair is too far apart, the gradient pulls them together; when a negative pair is too close, it pushes them apart. Same machinery, just driven by the sign of $\hat{y}-y$.

**iv. Did we ever tell the model categories or meanings or anything about the words?**

No. Nothing. We never said cat is an animal, that runs is a verb, or that teacher and student are people. The only information the model ever saw was: "these two words appeared in the same sentence" (positive) or "these two were sampled at random and probably don't go together" (negative). Everything else, the clusters, the directions, the analogies, came out of that one signal.

**v. How or what is this model learning?**

It's learning the **distributional hypothesis**: words that show up in similar contexts mean similar things. The training objective directly encodes that idea, words that co-occur get pulled toward the same direction, words that don't co-occur get pushed apart. Over many epochs, geometry in the embedding space ends up reflecting the structure of the corpus. Rough groupings can emerge when words share partners or sentence roles, but with this tiny corpus and only two dimensions those groupings are approximate rather than guaranteed clean categories.

**vi. If our embeddings were vectors with 100 components instead:**

**A. How would the math change?** It wouldn't, basically. $z = \vec{u}\cdot\vec{v}$ is still a dot product, just summing over 100 components instead of 2. The sigmoid, the loss, the derivation that $\frac{dL}{dz} = \hat{y}-y$, and the update rule $w_{\text{new}} = w - \eta(\hat{y}-y)\cdot(\text{partner})$ all stay exactly the same. The training code in this notebook is dimension-agnostic, the only change is `D = 2` becomes `D = 100`. The plotting code wouldn't work directly though, since you can't draw a 100-dimensional arrow; you'd need a dimensionality reduction technique like PCA or t-SNE to visualize. One other practical detail: a random vector in $[-1, 1]^{100}$ has a much larger expected norm than in $[-1, 1]^{2}$ (norm grows like $\sqrt{D}$), so to keep initial dot products small you'd usually scale the initial values down by something like $\frac{1}{\sqrt{D}}$.

**B. What would the model gain?** Capacity. In 2D every word has to share the same two axes with everyone else, so different *kinds* of relationships (subject vs verb, animal vs person, singular vs plural, tense, sentiment, topic) all have to compete for the same two directions and end up colliding. With 100 dimensions, different relationships can occupy different subspaces independently. Clusters separate more cleanly, and you start to get more interesting structure, like word arithmetic (the famous example: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$), which doesn't really fit in 2D. Higher dimensions are also what makes embeddings useful as inputs to bigger models like LLMs.