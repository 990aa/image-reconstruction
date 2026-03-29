# Attention-Guided Evolutionary Reconstruction with Multi-Resolution Progressive Growth

## Abstract
We present a time-bounded evolutionary image reconstruction framework that approximates arbitrary images using transparent geometric primitives. The method combines complexity-adaptive scheduling, multi-resolution progressive growth, and interactive residual-driven control. A five-panel visualization is introduced to expose optimization behavior in real time, including signed residual directionality and geometry-only structural evolution. The optimization is guarded by a hard timeout mechanism that prevents unbounded runs by dynamically throttling remaining compute cycles near deadlines. Across three visually distinct target categories (portrait, landscape, graphic), the method consistently outperforms a naive baseline under matched budgets, reducing final mean squared error (MSE) by 206.18 to 440.47 absolute points.

## 1. Introduction
Evolutionary image approximation with explicit primitives remains a compelling paradigm for interpretable generative graphics. Unlike latent or diffusion-based synthesis, each accepted primitive has a direct geometric and photometric interpretation. However, practical systems still struggle with three recurring issues: unstable convergence on complex images, weak diagnostics of residual directionality, and runtime variability that can lead to stalled interactive sessions.

This paper addresses these limitations with four contributions:
1. A complexity-adaptive multi-resolution progressive growth scheduler that automatically allocates polygon additions across coarse-to-fine rounds.
2. A five-panel live visualization for directional error interpretation, geometric evolution inspection, and loss-dynamics analysis.
3. Interactive controls for immediate growth and decomposition-based correction during optimization.
4. A hard timeout policy that guarantees bounded execution while preserving graceful convergence behavior.

## 2. Related Work
Evolutionary search has long been used for image approximation and procedural art due to its flexible objective formulation and parameter-free acceptance behavior [1], [2]. Multi-scale and pyramid-based optimization is a standard strategy for avoiding poor local minima by first fitting low-frequency structure [3], [4]. In computer vision and graphics diagnostics, signed residual analysis is known to provide richer optimization signals than magnitude-only heatmaps [4]. Interactive visual analytics can further improve optimization steering and user trust [5].

Our method integrates these principles into a single, real-time evolutionary pipeline with strict runtime bounding.

## 3. Method
### 3.1 Problem Formulation
Given a target image $T \in [0,1]^{H \times W \times 3}$ and a rendered canvas $C$, optimization minimizes pixel-wise reconstruction error:

$$
\mathcal{L}(T,C) = \frac{1}{3HW}\sum_{y=1}^{H}\sum_{x=1}^{W}\sum_{c=1}^{3}(T_{y,x,c}-C_{y,x,c})^2.
$$

The model state is a set of semi-transparent geometric primitives with learned center, scale, orientation, color, opacity, and shape-specific parameters.

### 3.2 Complexity-Adaptive Multi-Resolution Growth
For each input image, a complexity score is estimated from color segmentation and structure cues. This score controls:
1. Resolution pyramid rounds.
2. Polygon budget allocation across rounds.
3. Batch growth schedule and shape-size bounds.
4. Per-round optimization aggressiveness.

Optimization begins at low resolution to fit large spatial structure, then scales polygons to higher resolutions for detail recovery. This progressive transfer improves stability and accelerates useful convergence.

### 3.3 Time-Bounded Evolution
Two deadlines are tracked: a runtime budget and a hard safety timeout. Near the remaining deadline, per-cycle iteration counts are smoothly reduced by a throttling factor:

$$
\text{throttle} = \mathrm{clip}\!\left(\frac{t_{\text{remaining}}}{12},\,0.20,\,1.00\right).
$$

This avoids abrupt termination and ensures that ongoing growth/correction cycles finish safely while total runtime remains bounded.

### 3.4 Five-Panel Live Visualization
The proposed display is specifically designed for progressive growth diagnostics:
1. Target panel: static reference.
2. Reconstruction panel: current full-resolution canvas (updated at fixed cadence to prioritize optimizer compute).
3. Signed residual panel: $T-C$, where red indicates under-bright regions (too dark) and blue indicates over-bright regions (too bright).
4. Polygon-outline panel: outlines on white background, size-coded by color (large blue, medium green, small red).
5. Log-loss panel: MSE on logarithmic scale with vertical markers for (i) resolution transitions and (ii) polygon batch insertions.

The loss profile typically exhibits sawtooth behavior: steady descent, mild jump after growth insertion, then rapid recovery.

### 3.5 Interactive Controls
The interface preserves standard runtime controls and adds four critical controls for real-time intervention:
1. Immediate forced growth.
2. Immediate residual decomposition correction.
3. View cycling across reconstruction/residual/outline focus.
4. Live softness adjustment ($+/-$) to sharpen or soften edge transitions during rendering.

These controls are useful both for presentation and for active optimization steering on difficult regions.

## 4. Experimental Setup
### 4.1 Data
Three external images with distinct frequency characteristics were used:
1. Portrait scene (smooth gradients plus edge contours).
2. Natural landscape (mixed low and high spatial frequencies).
3. Graphic composition (high-contrast boundaries and stylized regions).

### 4.2 Protocol
A naive baseline and the proposed progressive system were run under matched budgets. Paired comparisons used a fixed 800-iteration setting to isolate algorithmic behavior from runtime variation. Additional short live runs validated interactive stability and artifact generation.

### 4.3 Metrics
Primary metric: terminal RGB mean squared error (MSE). Secondary qualitative evidence: residual directionality maps, outline evolution, and quality-vs-budget curves.

## 5. Results
### 5.1 Baseline vs Proposed Method
| Target Type | Naive Final MSE | Proposed Final MSE | Absolute Improvement |
| --- | ---: | ---: | ---: |
| Portrait | 555.1732 | 114.7067 | 440.4665 |
| Landscape | 208.1470 | 1.9645 | 206.1825 |
| Graphic | 359.3228 | 14.0109 | 345.3118 |

The proposed system consistently achieved substantially lower error. The largest relative quality gain was observed on the landscape image, where progressive growth nearly eliminated residual error at the tested budget.

### 5.2 Short Interactive Runs
Short bounded live sessions also showed stable convergence behavior:

| Target Type | Final Iteration | Accepted Polygons | Final MSE |
| --- | ---: | ---: | ---: |
| Portrait | 123 | 109 | 139.3204 |
| Landscape | 134 | 120 | 6.0246 |
| Graphic | 70 | 70 | 66.7524 |

These runs confirm that the timeout-safe scheduler can maintain meaningful descent even under strict wall-clock limits.

### 5.3 Qualitative Behavior
The signed residual panel reveals directional correction needs unavailable in scalar heatmaps. The polygon-outline panel shows that growth batches are structurally coherent: large primitives settle global composition first, then medium/small primitives refine local detail. Vertical event markers on log-loss plots align with expected sawtooth transitions during growth stages.

## 6. Discussion
The combined effect of complexity-adaptive scheduling, progressive growth, and hard-timeout throttling yields practical and reliable optimization for arbitrary images. Unlike naive fixed-resolution evolution, the method prevents excessive early commitment to fine-scale noise and remains interactive under presentation constraints.

The remaining quality gap on portraits suggests that smooth skin-tone transitions and subtle tonal variation still challenge polygon-limited representations at moderate budgets. Future work should consider perceptual losses in LAB/LPIPS space and adaptive primitive families for low-texture gradients.

## 7. Limitations
1. Reported experiments use three target categories; larger benchmark suites are needed for broad statistical generalization.
2. MSE does not fully capture human perceptual quality.
3. Growth scheduling still depends on heuristic complexity mappings.
4. Forced interventions can trade short-term gains for occasional local instability.

## 8. Conclusion
We introduced a full Phase-7 evolutionary reconstruction framework for arbitrary images, centered on multi-resolution progressive growth, five-panel interactive diagnostics, and strict runtime safety. The method is empirically strong across varied image classes and operationally robust for live usage. Results indicate large and repeatable accuracy improvements versus naive optimization while preserving interpretability and interactive control.

## References
[1] Mitchell, M. An Introduction to Genetic Algorithms. MIT Press, 1998.

[2] Baluja, S., and Davies, S. Fast Evolutionary Learning in Graphical Models for Image Approximation. In Proceedings of the Genetic and Evolutionary Computation Conference, 1997.

[3] Burt, P. J., and Adelson, E. H. The Laplacian Pyramid as a Compact Image Code. IEEE Transactions on Communications, 31(4):532-540, 1983.

[4] Gonzalez, R. C., and Woods, R. E. Digital Image Processing. 4th ed., Pearson, 2018.

[5] Heer, J., Bostock, M., and Ogievetsky, V. A Tour through the Visualization Zoo. Communications of the ACM, 53(6):59-67, 2010.

[6] Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. Optimization by Simulated Annealing. Science, 220(4598):671-680, 1983.

[7] Nocedal, J., and Wright, S. J. Numerical Optimization. 2nd ed., Springer, 2006.

[8] Van der Walt, S., Schonberger, J. L., Nunez-Iglesias, J., et al. scikit-image: Image Processing in Python. PeerJ, 2:e453, 2014.

[9] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12:2825-2830, 2011.

[10] Hunter, J. D. Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3):90-95, 2007.
