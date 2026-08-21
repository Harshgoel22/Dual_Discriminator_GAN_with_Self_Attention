# Dual Discriminator GAN with Self-Attention

A systematic study of **DCGAN, D2GAN, DCGAN with Self-Attention (DCGAN-SA), and D2GAN with Self-Attention (D2GAN-SA)** on the CIFAR-10 dataset.

**Course:** CS787 — Generative AI  
**Institution:** IIT Kanpur  
**Authors:** Harsh Goel, Kishan Kumar Mishra, Singampalli Sri Varshith, Thoram Venkata Madhu Sai Kiran

---

## Overview

Generative Adversarial Networks (GANs) are powerful generative models, but they can suffer from problems such as **mode collapse, unstable training, and poor sample diversity**.

This project investigates whether two ideas can improve GAN generation on CIFAR-10:

1. **Dual discriminators (D2GAN)** to provide complementary diversity and realism signals.
2. **Self-attention** to model long-range spatial dependencies between image regions.

We compare four architectures:

| Model | Description |
|---|---|
| **DCGAN** | Convolutional GAN baseline |
| **D2GAN** | DCGAN-style generator with two discriminators |
| **DCGAN-SA** | DCGAN with a self-attention block |
| **D2GAN-SA** | D2GAN combined with self-attention |

### Main Result

The experiments show that **D2GAN without self-attention performed best overall**.

- **Best FID:** 21.40
- **Best Inception Score:** 6.43
- **Best configuration:** D2GAN with `α = 0.10`, `β = 0.10`

Interestingly, self-attention improved the standard DCGAN but degraded D2GAN when the two techniques were combined.

---

## Results

### Main Benchmark

| Model | FID ↓ | Inception Score ↑ | Main Characteristics |
|---|---:|---:|---|
| **DCGAN** | 25.66 | 6.17 | Convolutional / transposed-convolutional GAN |
| **D2GAN** | **21.40** | **6.43** | Dual discriminators, `α=0.10`, `β=0.10` |
| **DCGAN-SA** | 23.62 | 6.34 | Self-attention, spectral normalization, hinge loss |
| **D2GAN-SA** | 27.55 | 6.09 | Dual discriminators + self-attention |

### Key Observation

D2GAN improved the baseline substantially:

- FID: **25.66 → 21.40**
- Inception Score: **6.17 → 6.43**

Adding self-attention to DCGAN also helped:

- FID: **25.66 → 23.62**
- IS: **6.17 → 6.34**

However, combining self-attention with D2GAN produced a significant degradation:

- FID: **21.40 → 27.55**
- IS: **6.43 → 6.09**

This suggests that architectural improvements that work independently do not necessarily combine constructively.

---

## Architectures

### 1. DCGAN

DCGAN is used as the baseline architecture.

The generator maps a latent vector:

```text
Input Latent Vector
                          z ∈ ℝ¹⁰⁰×¹×¹
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Transposed Convolution 1                    │
│   ConvT2d(100 ──> 512, kernel=4×4, stride=1, padding=0)      │
│   Batch Normalization  +  ReLU                               │
└──────────────────────────────┬───────────────────────────────┘
                               │ Feature Map: 512 × 4 × 4
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Refining Convolution 1                      │  <-- Texture Refinement
│   ConvT2d(512 ──> 512, kernel=3×3, stride=1, padding=1)      │      (Maintains 4×4 size)
│   Batch Normalization  +  ReLU                               │
└──────────────────────────────┬───────────────────────────────┘
                               │ Feature Map: 512 × 4 × 4
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Transposed Convolution 2                    │
│   ConvT2d(512 ──> 256, kernel=4×4, stride=2, padding=1)      │  <-- Spatial Upsampling (2x)
│   Batch Normalization  +  ReLU                               │
└──────────────────────────────┬───────────────────────────────┘
                               │ Feature Map: 256 × 8 × 8
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Refining Convolution 2                      │  <-- Texture Refinement
│   ConvT2d(256 ──> 256, kernel=3×3, stride=1, padding=1)      │      (Maintains 8×8 size)
│   Batch Normalization  +  ReLU                               │
└──────────────────────────────┬───────────────────────────────┘
                               │ Feature Map: 256 × 8 × 8
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Transposed Convolution 3                    │
│   ConvT2d(256 ──> 128, kernel=4×4, stride=2, padding=1)      │  <-- Spatial Upsampling (2x)
│   Batch Normalization  +  ReLU                               │
└──────────────────────────────┬───────────────────────────────┘
                               │ Feature Map: 128 × 16 × 16
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Refining Convolution 3                      │  <-- Texture Refinement
│   ConvT2d(128 ──> 128, kernel=3×3, stride=1, padding=1)      │      (Maintains 16×16 size)
│   Batch Normalization  +  ReLU                               │
└──────────────────────────────┬───────────────────────────────┘
                               │ Feature Map: 128 × 16 × 16
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Transposed Convolution 4                    │
│   ConvT2d(128 ──> 3,   kernel=4×4, stride=2, padding=1)      │  <-- Final Spatial Upsampling
│   Hyperbolic Tangent (Tanh)                                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
                       Generated Image
                          G(z) ∈ ℝ³×³²×³²
```

The final configuration uses:

- `ngf = 128`
- `ndf = 128`
- Transposed convolutions for upsampling
- Additional `3 × 3` refinement convolutions in the generator

The refinement layers improved the Inception Score from **5.90 to 6.17**.

---

### 2. D2GAN

D2GAN uses two discriminators instead of one:

![D2GAN Architecture](https://github.com/Harshgoel22/Dual_Discriminator_GAN_with_Self_Attention/blob/ac04a46ee0bdd511b06698742ee78705b7da003c/d2gan/d2gan_arch.png)

Conceptually:

- **D1** provides a signal associated with forward KL divergence:
  `KL(P_data || P_G)`
- **D2** provides a signal associated with reverse KL divergence:
  `KL(P_G || P_data)`

The combination aims to balance:

- **Mode coverage / diversity**
- **Sample quality / realism**

The D2GAN objective used in the project is:

$$\min_G \max_{D_1,D_2} J(G,D_1,D_2) = \alpha \mathbb{E}_{x\sim p_{\text{data}}}[\log D_1(x)] - \mathbb{E}_{z\sim p_z}[D_1(G(z))] - \mathbb{E}_{x\sim p_{\text{data}}}[D_2(x)] + \beta \mathbb{E}_{z\sim p_z}[\log D_2(G(z))]$$

At the theoretical optimum, the objective is related to:

$$
\alpha D_{KL}(P_{data}\parallel P_G)
+
\beta D_{KL}(P_G\parallel P_{data})
+
C
$$

where `α` and `β` control the relative contribution of the two divergence terms.

---

### 3. Self-Attention

The self-attention module follows the SAGAN-style formulation.

For an input feature map:

$$
x \in \mathbb{R}^{C\times H\times W}
$$

three projections are computed:

$$
f(x)=W_fx,\qquad
g(x)=W_gx,\qquad
h(x)=W_hx
$$

The attention weights are:

$$ \beta_{j,i} = \frac{\exp(f(x_i)^Tg(x_j))}{\sum_{i=1}^{N}\exp(f(x_i)^Tg(x_j))}$$

where:

$$N=H\times W$$

The output is:

$$
o_i =
\gamma
\sum_j \beta_{j,i}h(x_j)
+
x_i
$$

with `γ` initialized to zero.

The project evaluates self-attention in both DCGAN and D2GAN.

---

## Hyperparameter Studies

### DCGAN Feature-Map Depth

| `ngf / ndf` | Inception Score | Observation |
|---:|---:|---|
| 64 | 1.30 | Capacity-starved / unstable |
| **128** | **6.17** | Best balance |
| 256 | 6.00 | Stable but slower |

A feature-map width of **128** provided the best performance among the tested configurations.

---

### Generator Refinement Layers

| Configuration | IS |
|---|---:|
| Without refinement layers | 5.90 |
| With refinement layers | **6.17** |

The additional `3 × 3` convolutional layers improved local texture refinement.

---

## D2GAN α–β Grid Search

The project evaluated:

```text
α, β ∈ {0.01, 0.05, 0.10, 0.20}
```

### FID Results

| α \ β | 0.01 | 0.05 | 0.10 | 0.20 |
|---:|---:|---:|---:|---:|
| **0.01** | 23.77 | 23.49 | 23.33 | 21.88 |
| **0.05** | 24.37 | 24.37 | 25.20 | **21.54** |
| **0.10** | 26.51 | 25.03 | **21.40** | 24.80 |
| **0.20** | 25.45 | 25.40 | 23.42 | 22.93 |

The best configuration was:

```text
α = 0.10
β = 0.10
FID = 21.40
```

This grid search highlights the sensitivity of D2GAN to the balance between its two discriminator objectives.

---

## DCGAN-SA Ablation Study

The DCGAN-SA architecture was progressively modified to understand which design choices mattered most.

| Config | Modification | IS |
|---:|---|---:|
| 1 | Baseline DCGAN-SA | 5.77 |
| 2 | Remove spectral normalization from generator | 5.40 |
| 3 | Remove refinement layers + generator SN | 5.30 |
| 5 | Move self-attention to `16 × 16` | 5.75 |
| 6 | Hinge loss | 5.99 |
| 7 | Equal G/D learning rate (`2e-4`) | 6.05 |
| 8 | Increase discriminator LR to `4e-4` | 6.23 |
| 9 | Add 2 refinement layers | **6.34** |
| 10 | Extra refinement layer before SA | 5.74 |
| 11 | Extra refinement layer after SA | 6.13 |
| 12 | D updated 4× faster, `LR_G=1e-4` | 6.19 |
| 13 | Remove discriminator BatchNorm | 5.10 |

The best DCGAN-SA configuration achieved an **Inception Score of 6.34**.

---

## Why Did D2GAN Work Better?

A standard GAN discriminator provides a single learning signal. This can create a difficult trade-off:

- If the generator focuses strongly on realism, it may collapse to a small number of modes.
- If it focuses too heavily on diversity, sample quality can suffer.

D2GAN addresses this by using two complementary discriminators.

### D1 — Diversity / Mode Coverage

The forward-KL-related objective:

$$
D_{KL}(P_{data}\parallel P_G)
$$

strongly penalizes situations where the generator assigns insufficient probability mass to regions where real data exists.

This encourages broader coverage of the data distribution.

### D2 — Realism

The reverse-KL-related objective:

$$
D_{KL}(P_G\parallel P_{data})
$$

penalizes generated probability mass in regions that are not supported by the real distribution.

This encourages generated samples to remain realistic.

Together, the two objectives can provide a better balance between **diversity and fidelity**.

---

## Why Did D2GAN-SA Perform Worse?

Although self-attention improved DCGAN, the combination with D2GAN degraded performance.

The experimental result was:

```text
D2GAN     : FID 21.40, IS 6.43
D2GAN-SA  : FID 27.55, IS 6.09
```

Two plausible factors were identified.

### 1. Limited Need for Long-Range Attention

CIFAR-10 images are only `32 × 32`.

At this resolution, many object structures can already be modeled effectively using convolutional receptive fields. The additional global attention mechanism may therefore provide limited benefit relative to its optimization and parameter overhead.

### 2. More Complicated Optimization Dynamics

D2GAN already introduces a more complex adversarial game involving:

Adding self-attention, spectral normalization, and additional architectural interactions can make optimization more sensitive.

Therefore, the degradation should be interpreted as an **empirical result on this experimental setup**, rather than as evidence that self-attention is inherently incompatible with D2GAN.

---

## Evaluation Metrics

### Inception Score

The Inception Score measures both:

1. **Image quality / class confidence**
2. **Diversity across predicted classes**

It is defined as:

$$ IS(G) = \exp \left( \mathbb{E}_{x\sim p_g} \left[ D_{KL} \left( p(y|x)\parallel p(y) \right) \right] \right) $$

where:

- `p(y|x)` is the class distribution predicted for an image.
- `p(y)` is the marginal class distribution across generated samples.

Higher IS is better.

#### Limitations

- Can miss intra-class mode collapse.
- Depends on an ImageNet-pretrained classifier.
- Does not directly compare generated and real feature distributions.
- Does not reliably detect memorization of the training set.

---

### Fréchet Inception Distance

FID compares feature distributions of real and generated images.

Let:

- `(μ_r, Σ_r)` be real-image feature statistics.
- `(μ_g, Σ_g)` be generated-image feature statistics.

Then:

$$ FID = \|\mu_r-\mu_g\|_2^2 + Tr \left(\Sigma_r+\Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2} \right) $$

Lower FID indicates that the generated feature distribution is closer to the real distribution.

#### Limitations

- Assumes Gaussian statistics in feature space.
- Sensitive to sample size.
- Can be affected by preprocessing, resizing, blur, and compression.
- FID alone does not fully characterize image quality or diversity.

---

## Qualitative Evaluation

The project also evaluates generated samples visually.

A useful qualitative artifact is an `8 × 8` generated-image grid showing samples from D2GAN. The grid can be used to inspect:

- Object recognizability
- Visual sharpness
- Class diversity
- Evidence of mode collapse
- Texture quality
- Intra-class variation

Generated examples include recognizable CIFAR-10 categories such as animals, vehicles, and other object classes.

---

## Important Figures

### Figure 1 — D2GAN α–β FID Heatmap
![image alt](https://github.com/Harshgoel22/Dual_Discriminator_GAN_with_Self_Attention/blob/ac04a46ee0bdd511b06698742ee78705b7da003c/d2gan/Screenshot%202026-08-21%20170744.png)

### Figure 2 — D2GAN Most Confident Images
![image alt](https://github.com/Harshgoel22/Dual_Discriminator_GAN_with_Self_Attention/blob/ac04a46ee0bdd511b06698742ee78705b7da003c/d2gan/most_confident_images.png)

### Figure 3 — D2GAN Class Distribution
![image alt](https://github.com/Harshgoel22/Dual_Discriminator_GAN_with_Self_Attention/blob/ac04a46ee0bdd511b06698742ee78705b7da003c/d2gan/class_distribution_of_top_1000_imgs.png)

---

## Experimental Takeaways

The main conclusions from the experiments are:

1. **D2GAN outperformed the standard DCGAN baseline.**
2. **The best D2GAN configuration achieved FID 21.40 and IS 6.43.**
3. **Increasing DCGAN feature-map depth from 64 to 128 produced a major improvement.**
4. **Increasing the width further to 256 did not provide additional gains.**
5. **Generator refinement layers improved DCGAN image quality.**
6. **Self-attention improved DCGAN performance in the tested setup.**
7. **Self-attention did not improve D2GAN and instead caused a substantial degradation.**
8. **The interaction between GAN objectives and architectural modifications is highly non-linear.**
9. **FID and IS should be considered together with qualitative inspection rather than used in isolation.**

---

## Project Structure

A suggested repository structure is:

```text
.
├── README.md
├── fid.py
├── inception_scorer.py
|
├── dcgan/
│   ├── class_distribution_of_top_1000_imgs.png
│   ├── dcgan.py
│   ├── fake_images_epoch_75.png
|   ├── fid_scores.txt
|   ├── inception_scores.txt
|   ├── kl_divergence_scores.txt
|   ├── loss_plot_till_epoch_100.png
|   ├── most_confident_images.png
│   └── training_animation_till_epoch_100.mp4
|   
├── d2gan/
│   ├── class_distribution_of_top_1000_imgs.png
│   ├── d2gan.py
│   ├── fake_images_epoch_75.png
|   ├── fid_scores.txt
|   ├── inception_scores.txt
|   ├── loss_plot_till_epoch_150.png
|   ├── most_confident_images.png
│   └── training_animation_till_epoch_150.mp4
|
├── dcgan_with_self_attention/
│   ├── class_distribution_of_top_1000_imgs.png
│   ├── dcgan_with_attention.py
│   ├── fake_images_epoch_100.png
|   ├── fid_scores.txt
|   ├── inception_scores.txt
|   ├── loss_plot_till_epoch_150.png
|   ├── most_confident_images.png
│   └── training_animation_till_epoch_150.mp4
|
├── d2gan_with_self_attention/
│   ├── class_distribution_of_top_1000_imgs.png
│   ├── d2gan_with_self_attention.py
│   ├── fake_images_epoch_80.png
|   ├── fid_scores.txt
|   ├── inception_scores.txt
|   ├── loss_plot_till_epoch_100.png
|   ├── most_confident_images.png
│   └── training_animation_till_epoch_100.mp4
|
└── final_project_report.pdf
```

Adapt the structure above to match the actual repository implementation.

---

## Dataset

The experiments use **CIFAR-10**, containing:

- 10 object classes
- `32 × 32` RGB images
- 50,000 training images
- 10,000 test images

The dataset is commonly used as a benchmark for image-generation models because it is small enough for rapid experimentation while still containing substantial variation in object appearance.

---

## Reproducibility

For reproducible experiments, record at least:

- Random seed
- Number of training epochs / iterations
- Batch size
- Generator learning rate
- Discriminator learning rate
- Optimizer
- `ngf` / `ndf`
- `α` / `β` for D2GAN
- Self-attention resolution
- Spectral-normalization configuration
- Number of discriminator updates per generator update
- Number of generated samples used for FID and IS
- Image preprocessing and normalization

When comparing models, use the **same evaluation protocol and sample count** wherever possible.

---

## Limitations and Future Work

This study focuses on CIFAR-10 and therefore does not establish that the same architectural trends will hold for higher-resolution datasets.

Potential future directions include:

- Testing on higher-resolution datasets such as CelebA-HQ or ImageNet.
- Comparing multiple self-attention locations and attention dimensions.
- Evaluating alternative normalization techniques.
- Using modern GAN architectures such as StyleGAN variants.
- Performing multiple independent runs to report mean ± standard deviation.
- Evaluating Precision and Recall for generative models.
- Measuring diversity with class-conditional and feature-space metrics.
- Investigating whether attention is more useful at larger spatial resolutions.
- Studying the gradient interaction between the two D2GAN discriminators and the self-attention block.

---

## Citation / References

The project is based on ideas from the following areas:

- **DCGAN:** Radford, Metz, and Chintala, *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*.
- **D2GAN:** Nguyen et al., *Dual Discriminator Generative Adversarial Nets*.
- **Self-Attention GAN:** Zhang et al., *Self-Attention Generative Adversarial Networks*.
- **CIFAR-10:** Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*.
- **FID:** Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*.


---

## Final Summary

> **D2GAN was the strongest model in our CIFAR-10 experiments, achieving an FID of 21.40 and an Inception Score of 6.43. Self-attention improved the conventional DCGAN but unexpectedly hurt D2GAN, demonstrating that combining individually beneficial components does not necessarily lead to better generative performance.**

The project therefore highlights an important practical lesson in GAN design: **better architectural components are not automatically better when combined—their interaction with the underlying optimization objective matters.**
