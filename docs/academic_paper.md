# Sequential Perceptual Primitive Reconstruction: A Greedy Analytic-Color Approach for Interpretable Image Approximation

## Abstract
This report documents the final implementation state of a sequential primitive-based image reconstructor that approximates a target image using explicit alpha-blended geometric primitives. The system builds the image one primitive at a time through a greedy search loop that samples error-driven candidate locations, solves candidate colors analytically, refines geometry by mutation hill climbing, and commits only the single best primitive at each step. The final implementation uses a three-stage schedule at `100x100`, `150x150`, and `200x200` resolutions with progressively finer geometry, larger early alpha, and RGB-space candidate ranking. A complete seven-image benchmark at `200x200`, `1500` maximum shapes, and a `5` minute CPU budget per image yields average metrics of `rgb_mse = 0.007529`, `ssim = 0.74898`, `psnr = 25.900 dB`, `lab_mse = 36.489`, and `gradient_corr = 0.76832`. The final design outperformed both earlier joint optimizers and a later perceptual LAB-ranking trial.

## 1. Introduction
Primitive-based reconstruction is attractive because it produces interpretable approximations rather than opaque latent outputs. Each accepted primitive becomes a visible optimization decision: a location, a scale, a rotation, a color contribution, and a primitive family. This makes the method useful not only for image approximation, but also for understanding how approximation quality changes as the system allocates budget across color mass, structure, and fine detail.

The central challenge is balancing search quality with CPU throughput. A search rule that is perceptually elegant but too expensive can reduce the number of accepted primitives within a fixed wall-clock budget and therefore degrade the final reconstruction. The implementation described here was shaped by that tradeoff.

## 2. Final System Overview
The final system reconstructs a target image through a staged sequential search process:

![Pipeline Overview](C:\Users\ahada\Documents\abdulahad\evolutionary-art\docs\figures\pipeline_overview.png)

The primitive dictionary contains:
- ellipses
- quads
- triangles
- thin strokes

The canvas starts from the mean target color rather than a white or random initialization. This reduces the amount of early work spent filling the entire frame with a rough global color field.

## 3. Optimization Strategy

### 3.1 Sequential Growth
The system does not jointly optimize all existing primitives at once. Instead, it repeatedly solves a smaller local problem:
1. compute the current residual between target and canvas
2. build a guide map from that residual
3. sample candidate centers from the strongest regions of that guide map
4. generate candidate primitives near those centers
5. solve candidate colors analytically
6. rank candidates by reconstruction error
7. refine the best candidate by local mutation
8. commit only the winning primitive

This converts a difficult many-parameter global optimization problem into a sequence of bounded local decisions.

### 3.2 Analytic Color
Color is not searched by gradient descent. For each candidate primitive, the system computes its coverage mask and directly solves the optimal RGB value under alpha blending using the current canvas and target image. If `C` denotes the current canvas, `w` denotes the alpha-weighted coverage map of the new primitive, and `k` denotes the primitive color, the updated canvas is:

`C' = C + w * (k - C)`

The optimizer chooses `k` so that the resulting `C'` minimizes squared color error under the current visibility pattern. This removes RGB from the mutation search and lets the geometry optimizer spend its budget on placement, scale, and rotation.

### 3.3 Geometry Search
Geometry is refined through random mutation hill climbing. Starting from the best candidate found in the initial search pool, the system perturbs:
- horizontal position
- vertical position
- major axis size
- minor axis size
- rotation

Mutations are accepted only when they improve the candidate ranking objective.

### 3.4 Objective Choice
The final implementation ranks candidates by RGB mean squared error. This was not an arbitrary choice. A later experiment replaced detail-stage RGB ranking with LAB ranking while keeping RGB acceptance unchanged. That trial reduced throughput enough to produce worse final reconstructions, confirming that the fixed CPU budget favored the cheaper RGB objective.

### 3.5 Live Visualization
The final implementation includes a live Matplotlib dashboard and a recorded demo path built on the same optimizer state updates. The dashboard shows:
- the fixed target image
- the current reconstruction
- an absolute error heat map
- a log-scaled RGB loss curve with stage markers

Because the dashboard subscribes to the same state transitions that drive the optimizer, it can be verified rather than treated as a cosmetic overlay. The recorded demo path checks that update events arrive throughout the run and that the displayed loss decreases from the initial recorded state to the final one.

## 4. Multi-Stage Schedule
The final schedule is deliberately coarse-to-fine:

![Stage Schedule](C:\Users\ahada\Documents\abdulahad\evolutionary-art\docs\figures\stage_schedule.png)

The three stages differ in four important ways:
- resolution
- search budget
- geometry scale range
- allowable opacity

Early stages are intentionally more opaque so that broad color masses and shadows become stable quickly. Later stages reduce alpha and size, allowing finer corrections without completely overwriting the existing canvas.

The live dashboard layout used during interactive and recorded runs is shown below.

![Dashboard Layout](C:\Users\ahada\Documents\abdulahad\evolutionary-art\docs\figures\dashboard_layout.png)

## 5. Residual Routing and Primitive Bias
The guide map is based on absolute residual energy. The final stage uses a detail-aware version that adds a high-frequency component while still preserving part of the low-frequency residual:

`guide = high_frequency_residual + 0.40 * residual`

This was chosen because more aggressive structure-only routing caused the optimizer to neglect the smooth interiors of objects. The final implementation uses structure as guidance, not as the dominant objective.

Primitive selection is similarly biased but not over-constrained:
- ellipses dominate in smooth or curved regions
- quads and triangles appear when local structure is stronger
- thin strokes appear only on sufficiently linear edge-like regions

This preserves round organic forms while still allowing sharp accents where needed.

## 6. Throughput Considerations
Because the system is wall-time-limited, candidate throughput is a first-class concern. On the tuned best branch, a 5-minute grape benchmark processed approximately:
- `76,767` candidate evaluations
- `255.53` candidate evaluations per second

This measurement explains why apparently reasonable perceptual refinements can fail. A ranking objective that is more perceptually sophisticated but significantly slower may reduce accepted primitive count enough to lower final quality.

## 7. Experimental Protocol
The final evaluation sweep used:
- square resolution: `200x200`
- maximum primitive budget: `1500`
- runtime budget per image: `5 minutes`
- random seed: `42`
- fit mode: center crop

The target suite contained seven images:
- face
- grape
- heart
- internet graphic
- internet landscape
- internet portrait
- logo

Each target was reconstructed once after clearing all earlier output folders, leaving exactly one current run per target.

In addition, the three internet targets were reconstructed one after another through the live dashboard recorder, producing animated captures of the optimization process. The recorder verified that the visualization updated throughout the run and that the displayed loss decreased over time.

## 8. Results

### 8.1 Full-Suite Aggregate
Average metrics across the seven latest runs were:
- `rgb_mse = 0.007529`
- `ssim = 0.74898`
- `psnr = 25.900 dB`
- `lab_mse = 36.489`
- `gradient_mse = 0.002770`
- `gradient_mae = 0.02306`
- `gradient_corr = 0.76832`
- average accepted primitives = `377.43`

![Benchmark Summary](C:\Users\ahada\Documents\abdulahad\evolutionary-art\docs\figures\benchmark_summary.png)

### 8.2 Per-Image Results
| Target | Accepted Primitives | RGB MSE | SSIM | PSNR (dB) | LAB MSE | Gradient Corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Face | `362` | `0.003907` | `0.84210` | `24.081` | `19.201` | `0.92674` |
| Grape | `406` | `0.021227` | `0.43940` | `16.731` | `102.260` | `0.53904` |
| Heart | `375` | `0.001956` | `0.88783` | `27.087` | `11.986` | `0.93074` |
| Internet Graphic | `369` | `0.002101` | `0.71917` | `26.775` | `12.519` | `0.62341` |
| Internet Landscape | `391` | `0.000052` | `0.97600` | `42.827` | `0.563` | `0.85681` |
| Internet Portrait | `400` | `0.021521` | `0.43420` | `16.671` | `103.326` | `0.52817` |
| Logo | `339` | `0.001939` | `0.94415` | `27.125` | `5.569` | `0.97332` |

### 8.3 Interpretation
The easiest cases are flat, high-contrast, or low-detail targets such as the landscape, logo, and heart. Those images allow the staged primitive schedule to capture most of the visible content with relatively few accepted primitives and very high structural agreement.

The hardest cases are the grape and portrait targets. These require smooth organic shading, clustered edges, and medium-frequency texture all at once. They remain the best stress tests for the current approach.

## 9. Comparative Findings From Development
Three high-level lessons emerged from the experimental history:

1. Sequential local search beat global joint optimization.
   Joint parameter updates across many existing primitives underperformed a one-shape-at-a-time greedy search.

2. Strong early coverage mattered.
   Weak early alpha made the system underpaint broad color masses and spend later stages fixing problems that should have been solved earlier.

3. More perceptual ranking was not automatically better.
   The detail-stage LAB-ranking trial was computationally more expensive and produced worse final images because it reduced the number of accepted shapes within the same 5-minute budget.

4. Visualization works best when it is bound to optimizer state rather than post-hoc screenshots.
   The final dashboard and demo recorder share the same update stream, which made the live reconstruction process both informative and mechanically verifiable.

## 10. Limitations
The current system still has several limitations:
- highly textured natural scenes remain difficult
- runtime variability can noticeably change final quality under a hard wall-clock limit
- the detail stage is still constrained by CPU candidate throughput
- complex curved shading is approximated rather than fully matched

An isolated earlier benchmark on the same final implementation family produced a stronger grape result than the latest full-suite rerun, which indicates that the code is stable in design but still somewhat sensitive to runtime conditions under the fixed time budget.

## 11. Conclusion
The final implementation is a tuned sequential primitive reconstructor that combines analytic color solving, mutation-based geometry search, staged resolution growth, and error-driven candidate routing. It achieved strong results on simple and mid-complexity targets and remained competitive on harder organic scenes, while staying fully interpretable. The most important conclusion from development is that the winning improvements were not the most complicated ones. The best branch came from strengthening a simple greedy core rather than replacing it with a more elaborate global optimizer or a slower perceptual ranking rule.
