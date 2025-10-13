---
layout:
  width: default
  title:
    visible: false
  description:
    visible: false
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
  metadata:
    visible: false
---

# Diversity

<h2 align="center">Expression Diversity</h2>

Bitbox measures the diversity of expression-related activations by computing their entropy. If a person frequently activates only a few specific expression signals while rarely engaging others, the resulting entropy value is low.

This function only accepts [global](facial-expressions.md#expression-related-global-deformations) or [local](facial-expressions.md#localized-expression-units) facial expressions. It computes diversity across all expression coefficients together and produces a single score for the entire video and an additional score representing the average frame-wise entropy.

```python
from bitbox.expressions import diversity

# estimate global expression coefficients
exp_global, pose, lands3D = processor.fit()

# compute diversity
diversity_scores = diversity(exp_global, scales=6)
```

The computation is performed at multiple temporal scales, similar to the multiscale approach used for [Expressivity](expressivity.md). This allows Bitbox to capture expressions that unfold at different speeds, such as slow, moderate, or rapid changes in facial activity. A temporal scale represents the approximate duration of an expression event. For example, if the scale is 1 second, the algorithm identifies activations (peaks) in the expression signal that last about one second from start to finish. At each scale, a peak detection algorithm finds these activations, and entropy is then calculated based on their frequencies.&#x20;

Refer to the [Expressivity](expressivity.md) section for more details on temporal scales. All options available there—such as multiscale computation, single-scale analysis, and aggregation—are also supported for diversity calculations.

```python
# analysis using the original signal with no multiscale analysis
diversity_scores = diversity(exp_global, scales=None)

# using explicit scales
diversity_scores = diversity(exp_global, scales=[0.5, 1, 1.5, 2])

# aggregate over scales
diversity_scores = diversity(exp_global, scales=6, aggregate=True)

# setting fps
diversity_scores = diversity(exp_global, scales=6, fps=30)
```

The output is a Pandas `DataFrame` containing two scores for each temporal scale: overall entropy and average frame-wise entropy. Both scores range from 0 to 1.

{% hint style="success" %}
**Overall**: Entropy is calculated using the cumulative frequencies of peaks across the entire signal.\
**Frame-wise**: Entropy is calculated separately for each frame, and the results are then averaged over time.

Note that overall entropy will always be much higher than the entropy calculated per frame or their average. This is because, at any given frame, only a few expression coefficients are typically active, resulting in low entropy values.
{% endhint %}

```
   scale   overall   frame_wise
0  0.1     0.556535    0.01963
1  0.88    0.610186   0.004407
2  1.66    0.630256   0.002316
3  2.44    0.616686   0.001854
4  3.22    0.597459   0.001127
5   4.0    0.558529   0.000926
```

By default, Bitbox computes diversity using signal magnitudes. In this mode, peak frequencies are weighted by their magnitudes—stronger activations contribute more to the overall count. If you prefer to compute diversity in a binary manner, where only the presence or absence of an activation is considered (without weighting by magnitude), you can disable this behavior by setting `magnitude=False`.

```python
# compute diversity using binary peaks
diversity_scores = diversity(exp_global, scales=6, magnitude=False)
```
