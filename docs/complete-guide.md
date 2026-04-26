# The Complete Guide to Our NLP Project

## From Subspaces to Features: Mechanistic Analysis of Language-Reasoning Interference via Sparse Autoencoders

**A first-principles guide for the entire team. Read this before office hours.**

If you read nothing else, read this paragraph: We are going to look inside a language model's brain while it solves math problems in different languages, figure out which internal "neurons" are doing language stuff versus math stuff, surgically remove the language ones, and show that the model actually gets better at math when we do this. Nobody has done this at the individual-feature level before. The tools are free, the compute fits on a laptop, and the result, if it works, is publishable at a top venue.

---

# Part 1: Why This Project Exists and Why You Should Care

## 1.1 The paradox that started everything

Here is a fact that should bother you. If you take a language model, say Gemma 3 4B, and you ask it to solve a math word problem in Swahili, it performs worse than if you ask the same problem in English. That part is not surprising. English dominates the training data, so of course the model is better at English.

But here is the part that should bother you: if you take the model's internal representations while it is processing the Swahili problem and you mathematically remove the "Swahili-ness" from those representations, the model gets BETTER at solving the math. Not just a little better. For some languages, the accuracy jumps by more than 10 percentage points.

Think about that for a second. The model learned something about Swahili during training. That something is actively stored in its internal representations while it processes Swahili text. And that something is actively making the model worse at reasoning. Removing it helps.

This is like discovering that a student who speaks both English and Swahili actually does better on a math test if you temporarily suppress their knowledge of Swahili while they're solving problems. It makes no intuitive sense. Why would knowing a language make you worse at math?

That is the question we are answering.

## 1.2 Who discovered this and why it matters

A team led by Weixiang Zhao at the Harbin Institute of Technology published a paper called "When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners." It was accepted as a Spotlight at NeurIPS 2025, which means the top 3.2% of all submissions. This is a big deal paper.

What Zhao et al. did was relatively simple in concept. They took the hidden states of an LLM (the internal vectors the model computes while processing text), decomposed them into two parts using linear algebra (SVD, which we will explain later), and projected out the part that was language-specific. The remaining representation, which they called "language-agnostic," was then used for the rest of the computation. And performance went up.

They tested this across 10 different LLMs and 11 different languages. It worked almost every time. The gains were largest for low-resource languages like Swahili and Bengali, where the model presumably has the most "confusion" between language processing and reasoning.

## 1.3 What Zhao et al. did NOT explain

Here is the critical gap. Zhao et al. showed THAT removing language-specific representations improves reasoning. They did not explain WHY. Their method operates at the level of entire subspaces (big mathematical directions in a high-dimensional space). It cannot tell you which specific internal features are causing the problem, or how those features interfere with reasoning.

This is like knowing that removing a certain region of the brain improves a patient's ability to do arithmetic, but not knowing which specific neurons in that region were causing the problem, or whether they were competing for resources, sending wrong signals, or something else entirely.

## 1.4 What we are going to do

We are going to take a microscope to this phenomenon. Instead of working at the subspace level (Zhao et al.'s approach), we are going to work at the individual feature level using sparse autoencoders (SAEs). SAEs decompose a model's internal representations into individual, interpretable features, things like "this feature activates when the text is in Chinese" or "this feature activates when the model is doing addition."

Using pre-trained SAEs from Google DeepMind (called Gemma Scope 2), we will:

1. Identify which specific features are language-specific and which are reasoning-specific
2. Surgically remove only the language-specific features and see if reasoning improves (like Zhao et al.'s result but with a scalpel instead of a sledgehammer)
3. Figure out HOW the language features interfere with reasoning (do they compete for space? send wrong signals? mess up attention?)

Nobody has done step 3 before. That is our contribution to the field.

## 1.5 What we get out of this

Let us be concrete about the payoffs.

For the course: This is a well-scoped project with clear milestones, pre-built tools (no SAE training needed), and a guaranteed interesting result regardless of outcome. Even if our surgical ablation does not beat Zhao et al.'s method, the feature taxonomy we build is a contribution. The proposal is already written and strong.

For publication: ACL 2026 has "Explainability of NLP Models" as its special theme. Our project is an almost exact fit. EMNLP 2026 is another strong target. The intersection of mechanistic interpretability and multilingual NLP is explicitly identified as underexplored in a major EMNLP 2025 survey (Resck et al.). We are filling a named gap.

For understanding: If this works, it tells us something fundamental about how neural networks organize knowledge. The finding that language knowledge actively hurts reasoning suggests that models might be better off if they could "think" in a language-free space and only convert to/from language at the input and output. That has implications for how we train and deploy multilingual AI systems.

For your career: Having a publication (or even a strong submission) at ACL, EMNLP, or NAACL as a masters student is genuinely rare and looks excellent for PhD applications, research positions, or ML engineering roles. Mechanistic interpretability is one of the hottest areas in AI safety and alignment, with major labs (Anthropic, Google DeepMind, OpenAI) actively hiring for it.

---

# Part 2: Background Concepts from First Principles

## 2.1 What happens inside a language model

If you have never thought about what actually happens inside a transformer, here is the simplified version.

A language model takes in text (like "What is 2 + 3?"), converts each word (technically each "token") into a vector of numbers (an "embedding"), and then passes these vectors through a series of layers. Each layer transforms the vectors a little bit, mixing information between tokens (via attention) and processing information within each token (via the MLP/feedforward network). After all the layers, the final vector is used to predict the next token.

The key insight for our project: at any point during this process, each token is represented by a vector of numbers. For Gemma 3 4B, each token's representation is a vector of 2,304 numbers. These 2,304 numbers encode everything the model "knows" about that token at that point in the computation: what the word means, what language it is in, what the sentence is about, whether the model is in the middle of a calculation, etc.

These vectors are called "hidden states" or "residual stream activations." They are the internal representations we care about.

## 2.2 The residual stream: a running sum

Gemma 3 4B has 34 transformer layers. Each layer reads the current hidden state, computes something, and adds its output back to the hidden state. This "read, compute, add back" pattern is called the residual stream.

Think of it like a shared whiteboard. Each layer reads what is on the whiteboard, does some computation, and writes its result back on the whiteboard. The whiteboard accumulates information as it passes through layers.

By the time a hidden state has passed through all 34 layers, it contains contributions from every layer's computation. The final hidden state is what determines the model's output.

For our project, we care about what is written on the whiteboard at each layer. Specifically, we want to know: at layer 17 (roughly the middle of the model), does the whiteboard contain "this is Swahili" information mixed together with "I need to add these two numbers" information? And if so, can we erase the Swahili part without erasing the math part?

## 2.3 How multilingual models process language

Here is what the research community has established about how multilingual LLMs process text in different languages. It follows a three-phase pattern:

**Phase 1 (Early layers, roughly layers 0-10):** The model encodes language-specific information. It figures out "this is Chinese" or "this is Spanish" and processes the input according to the grammar and vocabulary of that language. The hidden states at these layers are very different across languages, even for the same semantic content.

**Phase 2 (Middle layers, roughly layers 10-25):** The model operates in a shared "concept space" that is largely language-independent. At these layers, the representation of "two plus three" in English looks very similar to the representation of the same concept in Chinese or Swahili. This is where reasoning happens.

**Phase 3 (Late layers, roughly layers 25-34):** The model converts back from the shared concept space to the target language for output. It figures out how to express the answer in the same language as the input.

This three-phase model has been confirmed by multiple independent research groups using different methods:

- Dumas et al. (ACL 2025) showed it via cross-lingual activation patching (swapping hidden states between languages and seeing when meaning is preserved)
- Wu et al. (ICLR 2025) formalized it as the "Semantic Hub Hypothesis" and proved it with causal interventions
- Wendler et al. (ACL 2024) showed that intermediate representations decode to the correct next token but in English, regardless of input language
- Tang et al. (ACL 2024) showed that language-specific neurons follow a U-shaped distribution (concentrated in early and late layers, sparse in middle layers)

This three-phase model is crucial for our project because it predicts WHERE in the model we should find the most language-reasoning interference: in the middle layers, where language-specific information from Phase 1 has not yet been fully suppressed and is coexisting with the reasoning computations of Phase 2.

## 2.4 SVD: how Zhao et al. separate language from reasoning

Now let us understand the math behind Zhao et al.'s method. Do not panic. This is actually simple once you see what it does geometrically.

**Singular Value Decomposition (SVD)** is a way to decompose a matrix into its most important directions. If you have a matrix M, SVD gives you M = U S V^T, where U and V are rotation matrices and S is a diagonal matrix of "singular values" that tell you how important each direction is.

Here is what Zhao et al. do with this:

Step 1: Run the model on text in multiple languages. For each language, record the hidden state vector at the final token position at each layer. This gives you a collection of vectors.

Step 2: Compute the mean hidden state for each language. So you get one "average Chinese vector," one "average Spanish vector," etc.

Step 3: Stack these mean vectors into a matrix M (rows = languages, columns = hidden dimensions).

Step 4: Apply SVD to M. This decomposes M into directions. Some directions capture differences between languages (these are the "language-specific subspace" M_s). Other directions are shared across languages (the "language-agnostic subspace" M_a). They are orthogonal by construction.

Step 5: For any hidden state h during inference, project out the language-specific component:

h_hat = h - lambda * M_s^T * M_s * h

This is a projection. M_s^T * M_s * h computes the component of h that lies in the language-specific subspace, and then you subtract it (scaled by lambda, which controls how aggressive the projection is).

Geometrically, imagine h is a vector in 2,304-dimensional space. Some directions in this space encode "what language is this" and other directions encode "what math operation is happening." SVD finds the language directions. The projection removes h's component along those directions, leaving only the language-agnostic part.

The lambda parameter controls the strength. lambda = 0 means no projection (original model). lambda = 1 means full projection. Zhao et al. found that lambda around 1.0-1.5 works best.

**The limitation:** This projection removes an ENTIRE SUBSPACE. It is like saying "remove everything in this general direction." But what if only some features in that direction are causing problems, while others are actually helping? SVD cannot tell you that. That is why we need SAEs.

## 2.5 Sparse Autoencoders (SAEs): the microscope

This is the core tool of our project, so let us build up the intuition carefully.

**The superposition problem.** Neural networks are sneaky. They pack way more information into their hidden states than you might expect from the number of dimensions. A 2,304-dimensional vector can encode information about thousands of different concepts, not just 2,304 of them. This is called "superposition": the model superimposes multiple features onto the same dimensions, like multiple radio stations broadcasting on overlapping frequencies.

This means that if you look at individual dimensions of the hidden state (individual neurons), you will not find clean, interpretable features. Dimension 742 might partially encode "this is French," partially encode "there is a number here," and partially encode "the sentence has negative sentiment." The features are entangled.

**What SAEs do.** A sparse autoencoder learns to decompose the entangled hidden state into a larger set of clean, interpretable features. The key idea is simple:

1. Take the hidden state vector h (dimension 2,304 for Gemma 3 4B)
2. Map it through an encoder to a much higher-dimensional space (say, 64,000 dimensions): z = encode(h)
3. Force z to be sparse (most entries are zero, only a few are active)
4. Map z back to the original space through a decoder: h_reconstructed = decode(z)
5. Train so that h_reconstructed is as close to h as possible

The magic is in step 3. Because z is forced to be sparse, each of the 64,000 dimensions tends to correspond to a single, interpretable concept. Dimension 12,847 might fire only when the text is in Chinese. Dimension 45,231 might fire only when the model is doing addition. Dimension 3,902 might fire when there is a question mark.

The sparsity constraint forces the SAE to find the actual underlying features rather than tangled mixtures.

**Why pre-trained SAEs matter.** Training SAEs is expensive. Google DeepMind spent the equivalent of 15% of the compute used to train Gemma 2 9B just to train the SAEs for it. Fortunately, they released Gemma Scope 2, which provides pre-trained SAEs for every layer of every Gemma 3 model. We do not need to train our own SAEs. We just load them and use them.

**What SAE features look like in practice.** When you run text through Gemma 3 4B and then through the SAE at layer 17, you get a sparse vector of 64,000 values. Most are zero. Maybe 50-150 are nonzero. Each nonzero entry tells you "feature X is active with strength Y." By examining what kinds of text activate each feature, researchers have found that features correspond to remarkably specific concepts: specific languages, specific types of reasoning, specific grammatical structures, specific entities, etc.

For our project, the features we care about are:

- **Language-specific features:** Features that activate only for text in one language. "This fires when the input is Chinese" or "This fires when generating Swahili text."
- **Reasoning features:** Features that activate for mathematical reasoning regardless of language. "This fires when the model is performing addition" or "This fires when the model is carrying a digit."
- **Shared features:** Features that activate across languages and tasks. These might encode general things like "there is a number here" or "the sentence is asking a question."

## 2.6 The monolinguality metric: quantifying how language-specific a feature is

Deng et al. (ACL 2025) introduced a simple and clever way to measure how language-specific an SAE feature is. Here is the formula:

For feature s and language L:

v_s^L = mu_s^L - gamma_s^L

Where:
- mu_s^L is the mean activation of feature s on text in language L
- gamma_s^L is the mean activation of feature s across all OTHER languages

If v_s^L is large and positive, the feature activates much more for language L than for other languages. It is a "language L feature."

If v_s^L is close to zero for all languages, the feature activates equally across languages. It is language-agnostic (potentially a reasoning feature or a shared feature).

If v_s^L is large for one language and negative for others, the feature is strongly language-specific to that one language.

We will compute this metric for every feature at every layer we analyze. This gives us a complete map of which features are language-specific and which are not.

**Our extension:** Deng et al. used this metric to study language control (steering output language). We are extending it to study reasoning. Specifically, for each math problem and each layer, we compute the average monolinguality of all active features. Then we test whether this average monolinguality score predicts whether the model gets the problem right or wrong. If high monolinguality (lots of language-specific features active) predicts lower accuracy, that is direct evidence that language features interfere with reasoning.

## 2.7 What "ablation" means and how we do it with SAEs

"Ablation" just means "removing." When we say "ablate a feature," we mean: set that feature's activation to zero and see what happens to the model's output.

With SAEs, the process works like this:

1. Run the model forward until layer L
2. Take the hidden state h at layer L
3. Encode h through the SAE to get sparse features z
4. Identify which features in z are language-specific (using the monolinguality metric)
5. Set those features to zero: z_modified = z with language features zeroed out
6. Decode z_modified back to the hidden state space: h_modified = decode(z_modified)
7. Replace the original h with h_modified and continue the forward pass

If the model's accuracy improves after this ablation, we have evidence that those language-specific features were interfering with reasoning.

This is more surgical than Zhao et al.'s method. They remove an entire subspace. We remove individual features. We can test:

- What if we remove only the top 10 most language-specific features?
- What if we remove language features only at layer 17 but not other layers?
- What if we remove Chinese-specific features when processing Chinese text but leave Spanish-specific features alone?

This granularity is what makes our approach novel.

## 2.8 Sparse Feature Circuits: tracing information flow

Phase 3 of our project aims to explain HOW language features interfere with reasoning. For this, we use the Sparse Feature Circuits methodology from Marks et al. (ICLR 2025).

The idea is to trace causal relationships between SAE features. If feature A at layer 10 causally influences feature B at layer 17, we draw an arrow from A to B. The collection of all such arrows forms a "circuit" or "attribution graph."

The method uses integrated gradients (a technique from the interpretability literature) to measure the causal contribution of each feature to each downstream feature and to the model's output. This gives us a map of how information flows through the model.

For our project, we want to trace the flow between language-specific features and reasoning features. Three things could happen:

**(a) Capacity competition.** Language features and reasoning features might be competing for the same representational capacity. If the hidden state has limited "bandwidth," active language features might be crowding out reasoning features. If this is the case, we predict: when we ablate language features, reasoning features should get stronger (higher activation magnitudes). This is measurable.

**(b) Circuit interference.** Language features might be sending signals downstream that confuse the reasoning computation. For example, a "this is Swahili" feature might activate a downstream circuit that handles Swahili grammar, and that circuit might interfere with the arithmetic circuit. If this is the case, we predict: the attribution graph should show language features causally connected to non-reasoning output components. This is measurable.

**(c) Attention disruption.** Language features might be distorting the attention patterns that route information between tokens. For example, if the model needs to attend to "2" and "3" to compute "2 + 3," a language feature might cause some attention to be diverted to language-specific tokens instead. If this is the case, we predict: after ablating language features, the attention entropy over reasoning-relevant tokens should decrease (attention becomes more focused). This is measurable.

These three hypotheses are not mutually exclusive. The real answer might be a combination. But having distinct, measurable predictions for each hypothesis is what makes Phase 3 scientifically rigorous rather than hand-wavy.

---

# Part 3: The Key Papers in Detail

You do not need to read all 15 papers we cite. But you should understand the 5 most important ones. Here they are.

## 3.1 Zhao et al. (2025) -- "When Less Language is More"

**What they did:** Took hidden states from LLMs processing multilingual text, decomposed them via SVD into language-specific and language-agnostic subspaces, projected out the language-specific part at inference time.

**Key results:** Consistent accuracy improvements across 10 models, 11 languages, 3 benchmarks. Largest gains on low-resource languages (Swahili: sometimes 10%+ improvement). Method is training-free, works at inference time, no fine-tuning needed.

**What they did not do:** Explain WHY language representations interfere with reasoning. Identify which specific features cause the problem. Determine the mechanism of interference.

**Why it matters to us:** This is our starting point. We replicate their result as a baseline and then go deeper with SAEs.

## 3.2 Deng et al. (2025) -- "Unveiling Language-Specific Features"

**What they did:** Used Gemma Scope SAEs on Gemma 2B/9B to identify language-specific features using the monolinguality metric. Showed that ablating language-specific features selectively degrades performance in only that language.

**Key results:** Language-specific SAE features exist predominantly in middle-to-final layers. Single-feature ablation affects only the target language. Some languages have synergistic features (ablating them jointly has greater effect than individual ablation).

**What they did not do:** Connect any of this to reasoning. Their paper is entirely about language identification and language control (steering which language the model outputs in).

**Why it matters to us:** We use their monolinguality metric and their methodology for identifying language-specific features. We extend it to study the effect on reasoning.

## 3.3 Chou et al. (2025) -- "Causal Language Control"

**What they did:** Modified SINGLE SAE features to switch the output language of Gemma-2B and Gemma-9B. Achieved 90% success rate.

**Key results:** Individual SAE features can control output language with high precision. Specific attention heads amplify language-correlated directions. Language control is concentrated in specific layers.

**Why it matters to us:** Proves that language-specific SAE features are causally potent (not just correlational). If a single feature can switch the language, removing that feature might genuinely affect how the model processes language during reasoning.

## 3.4 Marks et al. (2025) -- "Sparse Feature Circuits"

**What they did:** Developed methods for discovering causal circuits composed of SAE features (not neurons or attention heads). Used integrated gradients and attribution patching to identify causally important features.

**Key results:** Circuits composed of SAE features are more interpretable and more causally precise than circuits composed of coarse-grained components. The SHIFT procedure (ablating task-irrelevant features) improves task performance.

**Why it matters to us:** This is our Phase 3 methodology. We will use their techniques to trace information flow between language and reasoning features.

## 3.5 Gemma Scope 2 (Google DeepMind, 2025)

**What it is:** A massive release of pre-trained SAEs and transcoders for all Gemma 3 model sizes (270M through 27B). Over 1 trillion total SAE parameters. Uses JumpReLU architecture with Matryoshka training.

**What is available for Gemma 3 4B:** Residual stream SAEs at all 34 layers in 16k and 256k widths. At 4 subset layers (roughly 25%, 50%, 65%, 85% depth), additional widths of 64k and 1M are available. Attention output and MLP output SAEs at all layers. Transcoders at all layers. Weakly causal crosscoders at the 4 subset layers.

**Why it matters to us:** We do not need to train any SAEs. We just load the pre-trained ones and use them. This saves weeks of compute and removes a major source of risk from the project.

---

# Part 4: What Makes Our Project Novel

Let us be very precise about why this project has not been done before and why it matters.

## 4.1 The gap we fill

There are two established research threads:

**Thread 1:** Zhao et al. showed that removing language-specific SUBSPACES improves reasoning. They work at the subspace level (SVD). They showed THAT language interferes with reasoning but not WHY or WHICH specific features are responsible.

**Thread 2:** Deng et al. and Chou et al. showed that SAE FEATURES naturally decompose into language-specific and language-agnostic categories, and that individual features can be ablated or steered to control language. They studied language CONTROL but not reasoning.

Nobody has connected these two threads. Nobody has asked: "Which specific SAE features are the ones causing language-reasoning interference, and through what mechanism?"

That is our contribution.

## 4.2 Why this was not possible before December 2025

Before Gemma Scope 2 was released in December 2025, there were no pre-trained SAEs for Gemma 3. You would have had to train your own SAEs, which requires significant compute and expertise. Gemma Scope 2 made this project feasible for a course project team.

Additionally, the Zhao et al. paper was only published in 2025. The empirical finding that language representations interfere with reasoning is new. The combination of "new empirical finding" plus "new tools to investigate it" creates a window of opportunity that we are exploiting.

## 4.3 The three contributions

Our project makes three contributions, each independently publishable:

**Contribution 1 (Phase 1): A feature-level taxonomy of language vs. reasoning representations.** For the first time, we classify SAE features in a multilingual model into language-specific, reasoning-specific, and shared categories across all layers, with a metric connecting monolinguality to reasoning performance. This taxonomy is a resource the community can use.

**Contribution 2 (Phase 2): Surgical ablation vs. subspace projection.** We show whether feature-level ablation achieves a better reasoning-fidelity trade-off than SVD-based projection. If it does, this is a practical improvement. If it does not, this tells us something fundamental about the nature of language-reasoning interference (it operates at the subspace level, not the feature level).

**Contribution 3 (Phase 3): The mechanism of interference.** We provide the first mechanistic account of HOW language features interfere with reasoning, distinguishing between capacity competition, circuit interference, and attention disruption. This is the contribution that turns a solid paper into a potentially great one.

---

# Part 5: The Experimental Design in Detail

## 5.1 Phase 1: Building the feature taxonomy

### What we are doing

For each of the 250 MGSM math problems, in each of the 5 languages (English, Chinese, Spanish, Bengali, Swahili), we run the problem through Gemma 3 4B and record which SAE features activate at each layer.

### The data

MGSM is a benchmark of 250 grade-school math problems. Each problem is a word problem like "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?" The problems are professionally translated into 11 languages. We use 5 of them.

The reason we chose these 5 languages: English (high-resource, Latin script), Chinese (high-resource, logographic script), Spanish (high-resource, Latin script), Bengali (medium-resource, Bengali script), Swahili (low-resource, Latin script). This gives us 3 language families, 3 scripts, and a range from high to low resource. The typological diversity matters because we want to make sure our findings generalize and are not specific to, say, languages with Latin script.

### The process

For each problem-language pair (250 x 5 = 1,250 total):

1. Format the problem as a prompt for Gemma 3 4B
2. Run inference and record the model's answer (right or wrong)
3. At each layer, extract the hidden state
4. Pass the hidden state through the pre-trained SAE for that layer
5. Record which features are active and their activation strengths

This gives us, for each problem-language pair, a sparse vector of feature activations at each layer. With 64k-width SAEs at the 4 subset layers, this is 4 vectors of 64,000 entries each (mostly zeros). With 16k-width SAEs at all 34 layers, this is 34 vectors of 16,000 entries each.

### The analysis

Once we have all the feature activations, we compute:

**Per-feature monolinguality.** For each feature at each layer, compute v_s^L = mu_s^L - gamma_s^L across the 250 problems. Features with high v for one language and near-zero for others are language-specific. Features with near-zero v for all languages are language-agnostic.

**Feature taxonomy.** Classify each feature as: language-specific (high monolinguality for at least one language), reasoning-specific (activates consistently across languages on math problems but not on non-math text), or shared (activates across languages and task types).

**Reasoning correlation.** For each problem, compute the mean monolinguality of all active features at each layer. Fit a logistic regression predicting whether the model gets the problem right or wrong from this mean monolinguality score. If the coefficient is negative (more monolinguality predicts lower accuracy), this is evidence that language-specific features interfere with reasoning.

**Layer-wise distribution.** Plot the distribution of language-specific vs. reasoning-specific features across layers. We predict a pattern consistent with the three-phase model: language-specific features concentrated in early and late layers, reasoning features concentrated in middle layers, with the most "interference" (both types active) in the transition zones.

### What this produces

A complete map of the model's internal organization for multilingual reasoning. This map tells us exactly where and what to ablate in Phase 2.

## 5.2 Phase 2: Causal validation

### What we are doing

We test whether surgically removing language-specific SAE features improves reasoning, and compare this against Zhao et al.'s method.

### The hypotheses

**H1:** SAE-targeted ablation of language-specific features improves MGSM accuracy (replicates Zhao et al.'s finding at the feature level).

**H2:** SAE-targeted ablation achieves a better trade-off between reasoning improvement and language fidelity than SVD-based projection (because it removes fewer things more precisely).

**H3:** Ablating features at layers 17 and 22 (middle of the model, roughly 50% and 65% depth) accounts for the majority of the improvement (consistent with the three-phase model).

### The experimental conditions

We run the model on all 1,250 problem-language pairs under each of these conditions:

1. **Baseline:** Unmodified Gemma 3 4B. No intervention.
2. **SVD projection (Zhao et al. replication):** Implement their method. Compute language subspace from training examples, project it out at inference time, with their recommended lambda.
3. **SAE ablation (ours):** At the chosen layers, encode hidden states through the SAE, zero out language-specific features (identified in Phase 1), decode back. We test this at each layer individually and at all layers together.
4. **Random feature ablation:** Same as condition 3 but instead of ablating language-specific features, ablate the same NUMBER of random features. This controls for the possibility that any ablation improves performance.
5. **Deng et al. ablation:** Ablate language-specific features using Deng et al.'s original method (which was designed for language control, not reasoning). This tests whether our reasoning-targeted selection matters.
6. **English-only prompting:** Translate all problems to English, solve in English, translate back. This is the "thinking in English" baseline from Etxaniz et al.
7. **Reasoning amplification:** Instead of ablating language features, amplify reasoning features (increase their activation magnitude). This tests whether the improvement comes from removing interference or from boosting signal.

### The metrics

**Accuracy:** Percentage of MGSM problems the model solves correctly under each condition. This is the primary metric.

**Language fidelity:** We want the model to still respond in the correct language after our intervention. We measure this using LaBSE similarity between the output and reference text in the target language. If we remove language features and the model starts responding in English instead of Swahili, that is a problem.

**Per-language breakdown:** We report accuracy separately for each of the 5 languages. We expect the gains to be largest for low-resource languages (Bengali, Swahili), consistent with Zhao et al.

**Per-layer breakdown:** We report the effect of ablation at each layer individually. This tells us which layers contain the most "interfering" language features.

### The key methodological consideration

Zhao et al.'s projection operates on the final input token across all layers. SAE features can activate at multiple token positions. We need to test both:

- Ablating language features at the final token only (matching Zhao et al.'s setup)
- Ablating language features at ALL positions where they activate

The difference tells us whether the interference is localized to specific token positions or distributed across the sequence.

### What this produces

A clear comparison showing whether feature-level ablation outperforms subspace-level projection, and if so, by how much and at which layers.

## 5.3 Phase 3: Explaining the mechanism

### What we are doing

Assuming Phases 1 and 2 confirm that language features interfere with reasoning, Phase 3 asks: HOW?

### The three competing hypotheses with measurable predictions

**(a) Capacity competition.** If language features and reasoning features are competing for representational capacity (like two signals trying to use the same channel), then:
- PREDICTION: When we ablate language features, the activation magnitudes of reasoning features at the same layer should INCREASE. The reasoning features "expand" into the freed capacity.
- MEASUREMENT: Compare reasoning feature magnitudes before and after language feature ablation. Compute paired t-tests or similar.

**(b) Circuit interference.** If language features are sending causal signals that confuse downstream reasoning, then:
- PREDICTION: Attribution graphs (using Marks et al.'s Sparse Feature Circuits method) should show language features causally connected to non-reasoning output components. There should be "cross-wiring" between language circuits and reasoning circuits.
- MEASUREMENT: Compute integrated gradients from language features to output logits. Compare the attribution of language features to reasoning-correct output tokens vs. reasoning-incorrect output tokens.

**(c) Attention disruption.** If language features are distorting attention patterns, pulling attention away from reasoning-relevant tokens:
- PREDICTION: After ablating language features, attention entropy over reasoning-relevant tokens (the numbers and operators in the math problem) should DECREASE (attention becomes more focused on the right tokens).
- MEASUREMENT: Compute attention entropy before and after language feature ablation. Compare across layers.

### What this produces

The first mechanistic explanation of WHY language representations interfere with multilingual reasoning. This is the most novel and most ambitious part of the project. Even preliminary results here would be significant.

---

# Part 6: The Tools We Use

## 6.1 Gemma 3 4B

This is the model we study. It is a dense decoder-only transformer with:

- 34 transformer layers
- 2,304 hidden dimension
- 9,216 MLP intermediate dimension
- 8 attention heads with 4 KV heads (grouped-query attention)
- 256 head dimension
- 262k vocabulary
- Supports 128k context length
- Trained on 4 trillion tokens with knowledge distillation from Gemini 2.0
- Supports 140+ languages in pre-training
- Apache 2.0 license (fully open)

At int4 quantization, the model weights require about 3.3 GB. At bf16 (full precision), about 8 GB for weights alone. For activation extraction, we need bf16 to get accurate hidden states.

The model can run on an M4 MacBook with 16 GB unified memory, though it will be tight at bf16 with a reasonable context length. For larger runs, we can use NDIF (National Deep Inference Fabric) via the nnsight library, which provides remote access to larger hardware.

To get the model locally: `ollama pull gemma3:4b` gives you the quantized version. For full precision with activation access, you need HuggingFace Transformers or a framework like TransformerLens.

## 6.2 Gemma Scope 2

This is the collection of pre-trained SAEs for Gemma 3. It was released by Google DeepMind in December 2025.

For Gemma 3 4B specifically, the available SAEs include:

- **Residual stream SAEs** at all 34 layers (16k and 256k widths)
- **Residual stream SAEs** at 4 subset layers (additional 64k and 1M widths)
- **Attention output SAEs** at all layers
- **MLP output SAEs** at all layers
- **Transcoders** at all layers
- **Weakly causal crosscoders** at 4 subset layers

The SAEs use JumpReLU activation with Matryoshka training (features ordered by importance).

To load a SAE (using SAELens):

```
from sae_lens import SAE
sae, cfg, sparsity = SAE.from_pretrained(
    release="gemma-scope-2-4b-pt-resid_post",
    sae_id="layer_17_width_64k_l0_medium"
)
```

**CRITICAL NOTE:** Before starting experiments, verify the exact layer indices and SAE identifiers on the HuggingFace repo: huggingface.co/google/gemma-scope-2. The subset layer indices {9, 17, 22, 29} are from the technical paper but might differ for the 4B model. Check this first.

## 6.3 SAELens

SAELens (github.com/decoderesearch/SAELens) is the standard library for working with sparse autoencoders. Version 6 supports any PyTorch model. It handles:

- Loading pre-trained SAEs from HuggingFace
- Encoding activations through SAEs
- Extracting feature activations
- Computing feature statistics

## 6.4 TransformerLens

TransformerLens (github.com/TransformerLensOrg/TransformerLens) is a library for mechanistic interpretability. It provides:

- Hook-based access to any intermediate activation in the model
- Easy activation patching and causal interventions
- Support for Gemma 3 (added in recent releases)

For our project, we use TransformerLens to:
- Run the model with hooks at every layer
- Replace hidden states with SAE-modified versions
- Measure attention patterns before and after interventions

## 6.5 nnsight

nnsight (nnsight.net) allows remote execution of interpretability experiments on NDIF infrastructure. If our local machine cannot handle bf16 inference on Gemma 3 4B with full activation extraction, nnsight lets us run the same code on remote hardware. Version 0.6 (February 2026) works with any PyTorch model.

## 6.6 Neuronpedia

Neuronpedia (neuronpedia.org) hosts interactive exploration of SAE features. You can browse features for Gemma 3 models, see what text activates each feature, and steer models using specific features. Useful for qualitative exploration and for building intuition about what specific features do.

## 6.7 MGSM dataset

Available at HuggingFace: juletxara/mgsm. Contains 250 math problems in 11 languages. We also note that MGSM-Rev2 (github.com/google-research-datasets/MGSM-Rev2) corrects some translation errors in the original, so check whether the corrected version is more appropriate.

---

# Part 7: How to Actually Build This Project

## 7.1 Project infrastructure

### GitHub repository

Create a private GitHub repository immediately. Structure:

```
nlp-project/
├── README.md                    # Project overview, setup instructions
├── requirements.txt             # Python dependencies
├── setup.py or pyproject.toml   # Package configuration
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── 01_model_sanity_check.ipynb
│   ├── 02_sae_exploration.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_monolinguality_analysis.ipynb
│   ├── 05_causal_ablation.ipynb
│   └── 06_circuit_analysis.ipynb
├── src/                         # Reusable code
│   ├── data/                    # Data loading, MGSM processing
│   ├── features/                # SAE feature extraction, monolinguality computation
│   ├── interventions/           # Ablation, steering, SVD projection
│   ├── evaluation/              # Accuracy computation, LaBSE similarity
│   └── visualization/           # Plots, figures for the paper
├── scripts/                     # Scripts for running experiments
│   ├── extract_features.py      # Extract SAE features for all problem-language pairs
│   ├── compute_monolinguality.py # Compute monolinguality scores
│   ├── run_ablation.py          # Run causal ablation experiments
│   └── run_circuits.py          # Run circuit analysis
├── results/                     # Saved results, checkpoints
├── figures/                     # Generated figures
└── paper/                       # LaTeX files for the paper
    ├── main.tex
    ├── references.bib
    └── figures/
```

### Collaboration practices

Use git branches for parallel work. Never push directly to main. Use pull requests with brief descriptions. This avoids the situation where someone's code breaks everyone else's setup.

Share a Google Doc or Notion page for daily progress notes. Quick updates like "Extracted features for English and Chinese at layers 9 and 17. Layer 17 shows clear monolinguality separation. Layer 9 is messier." This keeps everyone informed without needing meetings.

### Environment setup

Python 3.10+. Use a virtual environment or conda environment. Key dependencies:

- torch (PyTorch, for model inference)
- transformers (HuggingFace, for loading Gemma 3 4B)
- sae_lens (for loading Gemma Scope 2 SAEs)
- transformer_lens (for activation hooks and interventions)
- nnsight (for remote execution if needed)
- numpy, scipy (for linear algebra, statistics)
- matplotlib, plotly (for visualization)
- scikit-learn (for logistic regression in the monolinguality-accuracy analysis)
- sentence_transformers (for LaBSE similarity)
- datasets (HuggingFace, for loading MGSM)
- pandas (for data management)

## 7.2 Week-by-week execution plan

### Week 1 (Days 1-3): Sanity checks and infrastructure

This is the most important week. If the basic pipeline does not work, nothing else matters.

**Day 1: Model setup**

- Install all dependencies
- Load Gemma 3 4B in bf16 (if memory allows) or int8
- Run 10 MGSM problems in English. Does the model get them right? Record accuracy.
- Run 10 MGSM problems in Swahili. Does the model produce coherent output at all?
- Run 10 MGSM problems in Bengali. Same check.
- Print the model config and confirm: number of layers, hidden dimension, vocabulary size.

If the model gets 0% on Swahili or Bengali, we need to know this now so we can adjust our language selection.

**Day 2: SAE setup**

- Load a Gemma Scope 2 SAE for one layer (try the "middle" subset layer first)
- Verify the layer indices match the model. Print the available SAE identifiers.
- Run one English prompt through the model, extract the hidden state at that layer, encode through the SAE
- Print: how many features are active? What are their indices? What are their activation magnitudes?
- Repeat for a Chinese prompt. Are different features active? Some overlap? How much?

This is the moment of truth. If the SAE pipeline works, the rest of the project is execution. If it doesn't, we need to debug now.

**Day 3: Extraction pipeline**

- Build a function: given a prompt, return SAE feature activations at all subset layers
- Test it on 5 problems in 2 languages (10 total runs)
- Save the results to disk in a clean format (numpy arrays or pickled dictionaries)
- Verify that re-loading and re-analyzing produces identical results

### Week 1-2 (Days 4-7): Feature extraction at scale

Once the pipeline works on 10 examples, scale it up:

- Run feature extraction for all 250 problems x 5 languages = 1,250 runs
- At each subset layer, save the sparse feature vectors
- This will take some hours of compute. Monitor memory usage.
- Also run the model normally (without SAE intervention) to record baseline accuracy on all 1,250 problem-language pairs

**Parallel task: Zhao et al. replication**

- Implement the SVD-based projection from Zhao et al.
- This requires: running the model on a set of "calibration" texts in each language to compute the mean representations, then computing the SVD to find the language subspace
- Apply the projection at inference time and record accuracy
- Compare to their reported numbers. We do not need to match exactly (different model), but the trend (improvement for non-English languages) should be similar

### Week 2-3 (Days 8-14): Monolinguality analysis and taxonomy

Now we have feature activations for 1,250 problem-language pairs at multiple layers. Time to analyze.

**Computing monolinguality scores:**

- For each feature at each layer, compute v_s^L for each language
- Rank features by maximum monolinguality (most language-specific to least)
- Visualize the distribution: histogram of monolinguality scores, separated by layer
- Do you see the U-shaped pattern (more language-specific features in early/late layers)?

**Building the taxonomy:**

- Classify features into three categories:
  - Language-specific: max monolinguality > threshold (tune this threshold)
  - Reasoning-specific: low monolinguality AND high activation on math problems across all languages
  - Shared: everything else
- How many features fall into each category at each layer?
- Are there features that are BOTH language-specific AND high-activation during reasoning? These are the prime suspects for interference.

**Connecting to accuracy:**

- For each problem-language pair, compute the mean monolinguality of active features at each layer
- Fit logistic regression: accuracy ~ mean_monolinguality + layer + language
- Is the monolinguality coefficient negative? (More language-specific features active = lower accuracy?)
- Plot the relationship

### Week 3-4 (Days 15-21): Causal ablation experiments

This is the core experiment. We now have a list of language-specific features at each layer. Time to ablate them and see what happens.

**Building the intervention pipeline:**

- Write a function that: loads the model, hooks into a specific layer, encodes the hidden state through the SAE, zeros out specified features, decodes back, and replaces the hidden state
- Test this on a single example first. Does the model still produce coherent output? Does accuracy change?

**Running the experiments:**

For each of the 7 experimental conditions (baseline, SVD projection, SAE ablation, random ablation, Deng et al. ablation, English-only, reasoning amplification):

- Run all 1,250 problem-language pairs
- Record accuracy and output text
- Compute LaBSE similarity for language fidelity

**Analyzing the results:**

- Table: accuracy per language per condition
- Bar chart: overall accuracy per condition with 95% confidence intervals
- Scatter plot: accuracy improvement vs. number of features ablated (is there a sweet spot?)
- Layer-by-layer: which layer's ablation helps most? (We predict middle layers.)
- Final token vs. all positions: does the granularity of intervention matter?

### Week 4-5 (Days 22-28): Phase 3 and paper writing

**Circuit analysis (if time permits):**

- Select the most interesting cases from Phase 2 (the problems/languages where SAE ablation helped most)
- Apply Sparse Feature Circuits methodology: compute integrated gradients from language features to reasoning features and to output
- Build attribution graphs
- Test the three hypotheses:
  - (a) Do reasoning feature magnitudes increase after language ablation?
  - (b) Do language features have causal connections to non-reasoning outputs?
  - (c) Does attention entropy decrease after ablation?

**Paper writing:**

- Start with the results tables and figures (build the paper around the data)
- Write the abstract last (it should summarize what you found, not what you planned)
- The related work is mostly done (it is in the proposal)
- The methodology section should be clear enough that someone could replicate your work

## 7.3 What to do if things go wrong

### If the SAE pipeline does not work with Gemma 3 4B

Fall back to Gemma 3 1B, which is smaller and more likely to work smoothly with existing tools. Gemma Scope 2 provides SAEs for 1B as well. The results will be less impressive (smaller model) but the methodology and findings are the same.

### If SAE ablation does not improve reasoning

This is explicitly handled in our proposal as an informative negative result. It means the interference operates at the subspace level (Zhao et al.'s SVD captures it) rather than the individual feature level (SAEs miss it). This tells us something important about the nature of the interference: it is distributed and cannot be localized to individual features. The feature taxonomy from Phase 1 is still a contribution.

### If Bengali/Swahili produce garbage output

Replace with Hindi (medium-resource, Devanagari script) and French (high-resource, Latin script) or any combination that maintains typological diversity from the 11 MGSM languages.

### If we run out of time for Phase 3

Phase 3 is explicitly scoped as exploratory. Present whatever preliminary results you have. Even showing that one of the three hypotheses is supported by partial evidence is interesting. The paper can present it as "initial findings suggest capacity competition rather than circuit interference" with future work to fully resolve.

### If Zhao et al.'s replication does not work on Gemma 3 4B

Their paper tested specific models (Qwen-2.5, Qwen-3, DeepSeek-R1, GLM). They did not test Gemma 3. If their method does not work on Gemma 3, that is itself an interesting finding (model-dependent effect). But it is unlikely, they tested 10 diverse models and it worked on all of them.

## 7.4 Compute requirements

Let us be realistic about what compute we need.

**Feature extraction (Phase 1):** 1,250 forward passes through Gemma 3 4B at bf16. Each forward pass takes maybe 5-15 seconds on an M4 MacBook (depending on sequence length). So 1,250 runs = 2-5 hours. Plus SAE encoding at each layer adds some overhead but SAE inference is cheap. Budget 8-12 hours total for feature extraction.

**Causal ablation (Phase 2):** 7 conditions x 1,250 runs = 8,750 forward passes. Each pass involves hooks and SAE intervention, so maybe 10-20 seconds each. Budget 25-50 hours. This is the most compute-intensive part. If running on a laptop, this might need to run overnight for a few nights.

**Circuit analysis (Phase 3):** Integrated gradients require multiple forward and backward passes per example. This is more expensive. We probably cannot run it on all 1,250 examples. Select 50-100 representative examples (covering different languages and accuracy levels) and run circuit analysis on those. Budget 10-20 hours.

**Total:** Roughly 50-80 hours of compute. This is achievable on a laptop over 5 weeks, but only if you start early and let things run overnight. If you have access to a GPU (even a single A100 for a few hours), everything speeds up dramatically.

**NDIF alternative:** If local compute is insufficient, use nnsight to run on NDIF infrastructure. This requires internet access and an account, but gives you access to much more powerful hardware.

---

# Part 8: What to Know for Office Hours

## 8.1 Questions to ask the professor

These are the questions that would make the best use of office hours:

1. "We are planning to use Gemma Scope 2 pre-trained SAEs on Gemma 3 4B. Have you seen any issues with using pre-trained SAEs for causal interventions? Should we be worried about reconstruction error affecting our ablation results?"

2. "Our Phase 3 proposes three competing hypotheses for the mechanism of interference. Is there a hypothesis you think is most likely? We want to make sure we are not chasing a dead end."

3. "We are thinking about venue for publication. ACL 2026 has 'Explainability of NLP Models' as its special theme. Do you think our project fits better as an ACL submission or an EMNLP submission?"

4. "Are there any multilingual reasoning benchmarks beyond MGSM that you would recommend? We are using MGSM because Zhao et al. used it, but we want to know if reviewers would expect additional benchmarks."

5. "Do you have any concerns about the 5-week timeline? Our Phase 3 is explicitly scoped as exploratory, but we want to make sure Phases 1 and 2 are sufficient for a strong project even without Phase 3 results."

## 8.2 What to know if the professor asks tough questions

**"How is this different from just replicating Zhao et al. with different tools?"** Answer: Zhao et al. showed THAT language representations interfere with reasoning. They used a blunt tool (SVD projection of entire subspaces). We are asking WHICH specific features cause the interference and HOW they do it. The tool change (from SVD to SAEs) is not the contribution. The contribution is the mechanistic explanation.

**"Why Gemma 3 4B specifically?"** Because Gemma Scope 2 provides pre-trained SAEs for every layer, which eliminates the need to train our own SAEs. No other model family has this level of SAE coverage. Additionally, Gemma 3 4B is small enough to run locally but large enough to have genuine multilingual capability.

**"What if SAE features are not the right level of analysis?"** This is explicitly addressed in our risk mitigation. If SAE-targeted ablation does not improve reasoning but SVD projection does, that is an informative result: it means the interference is distributed across features and cannot be localized. The feature taxonomy from Phase 1 is still a contribution.

**"Is 250 problems enough?"** For aggregate accuracy comparisons with bootstrap confidence intervals, yes. The standard in the multilingual reasoning literature (Shi et al. 2022, Zhao et al. 2025) is to use MGSM's 250 problems. If we want per-layer, per-language breakdowns (which gives us 5 languages x 34 layers = 170 cells), the power per cell is thin. But we are looking at aggregate trends across layers, not per-cell significance.

**"How do you handle the fact that SAE reconstruction is lossy?"** Good question. When we encode through the SAE and decode back, the reconstructed hidden state is not identical to the original. This introduces noise. We control for this by including a condition where we encode and decode without any feature ablation (i.e., pass through the SAE and back). If this alone changes accuracy, we know the SAE reconstruction error is a confounder and we need to account for it.

---

# Part 9: The Bigger Picture

## 9.1 Why this matters beyond our project

If language-specific features genuinely interfere with reasoning in LLMs, this has several important implications:

**For model training:** Future models might benefit from architectures that explicitly separate language processing from reasoning. Imagine a model with a "language encoder" that converts any language into a shared representation, a "reasoning module" that operates on that shared representation, and a "language decoder" that converts the result back. This is essentially what the model already does approximately (the three-phase model), but making it explicit could be more efficient.

**For deployment:** Organizations deploying multilingual AI systems could use our ablation technique at inference time to improve reasoning performance for non-English users. This is a practical, training-free intervention.

**For AI safety:** Understanding how information flows between language and reasoning circuits is relevant to alignment. If a model can be "confused" by language-specific features during reasoning, similar interference might occur in other settings (e.g., safety training conflicting with capability). Mechanistic understanding of interference is a step toward safer systems.

**For cognitive science:** The parallel to human cognition is striking. Cognitive neuroscience has shown that the human brain's language network (Broca's area, Wernicke's area) is largely inactive during mathematical reasoning. Our finding that suppressing language features in LLMs improves reasoning echoes this result. It suggests that the tension between language and reasoning might be a fundamental feature of any system that learns both.

## 9.2 Where this research goes next

If our project succeeds, the natural next steps are:

- **More models:** Does the same pattern hold in Llama, Qwen, Mistral? The feature taxonomy might be model-specific.
- **More tasks:** Does language interference affect other reasoning tasks beyond math? Logical reasoning, causal reasoning, spatial reasoning?
- **Training interventions:** Can we modify the training process to reduce language-reasoning interference from the start?
- **Dynamic ablation:** Can we build a system that automatically detects when language features are interfering and ablates them on the fly?
- **Theoretical understanding:** Why does superposition lead to interference between language and reasoning? Is this a consequence of the training objective, the architecture, or something else?

Each of these is a separate paper. If our project works, there is a clear multi-year research agenda behind it.

---

# Part 10: Quick Reference

## 10.1 Key numbers to remember

- Gemma 3 4B: 34 layers, 2,304 hidden dim, 3.9B params
- MGSM: 250 problems, 11 languages, we use 5
- Gemma Scope 2 SAEs: available at all 34 layers, 4 subset layers with additional widths
- SAE feature width: 16k (all layers) or 64k (subset layers)
- Total forward passes needed: roughly 10,000 across all experiments
- Compute time estimate: 50-80 hours on laptop, much less with GPU

## 10.2 Key URLs

- Gemma 3 4B: huggingface.co/google/gemma-3-4b-it
- Gemma Scope 2: huggingface.co/google/gemma-scope-2
- SAELens: github.com/decoderesearch/SAELens
- TransformerLens: github.com/TransformerLensOrg/TransformerLens
- MGSM: huggingface.co/datasets/juletxara/mgsm
- nnsight: nnsight.net
- Neuronpedia: neuronpedia.org
- Zhao et al. code: github.com/MuyuenLP/Language-Reasoning-Disentangle
- Our proposal PDF: already submitted to course

## 10.3 Key papers (read at least the abstracts)

1. Zhao et al. "When Less Language is More" (NeurIPS 2025) -- our starting point
2. Deng et al. "Unveiling Language-Specific Features" (ACL 2025) -- our methodology
3. Marks et al. "Sparse Feature Circuits" (ICLR 2025) -- our Phase 3 method
4. Gemma Scope 2 Technical Paper (DeepMind 2025) -- our tools
5. Shi et al. "MGSM" (NeurIPS 2022) -- our benchmark

## 10.4 The one-sentence pitch

"We use sparse autoencoders to identify which specific features in a multilingual language model cause language-specific representations to interfere with reasoning, and we provide the first mechanistic explanation of why removing these features improves performance."

## 10.5 Priority order if we run out of time

1. Phase 1 feature taxonomy (minimum viable project, still a contribution)
2. Phase 2 with at least 3 conditions: baseline, SVD projection, SAE ablation (the comparison is the core result)
3. Phase 2 with all 7 conditions (strengthens the paper)
4. Phase 3 with at least one hypothesis tested (makes the paper great)
5. Phase 3 with all three hypotheses tested (makes the paper exceptional)

Stop wherever time runs out. Each level produces a stronger paper than the previous one, but even level 1 alone is a valid course project.