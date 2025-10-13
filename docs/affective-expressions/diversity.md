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

Bitbox quantifies diversity of expression-related activations by computing the entropy of those activations. If a person activates only certain types of expression signals, and not the others, entropy is small.&#x20;

This function only accepts [global](facial-expressions.md#expression-related-global-deformations) or [local](facial-expressions.md#localized-expression-units) facial expressions. It computes diversity over all expression coefficients together and generate a single score for the entire video, as well as a score for the average frame-wise entropy.&#x20;

```python
from bitbox.expressions import diversity

# estimate global expression coefficients
exp_global, pose, lands3D = processor.fit()

# compute expressivity stats
diversity_scores = diversity(exp_global, scales=6)
```

The computation is performed at several temporal scales, similar to multiscale computations with [Expressivity](expressivity.md#temporal-scales). Multiscale computation allows capturing expressions that unfold at different speeds—for example, slow, moderate, or rapid changes in facial activity. A temporal scale represents the approximate duration of an expression event. For instance, if the scale is 1 second, the algorithm looks for activations (peaks) in the expression signal that last about one second from start to finish. At each scale, a peak detection algorithm identifies these activations, and the entrpopy is computed using peak frequencies. Please see [Expressivity](expressivity.md#temporal-scales) section for the details on temporal scales.&#x20;

The output is a Pandas `DataFrame`, containing two scores per temporal scale: overall entropy and average frame-wise entropy. The range for both scores is 0-1.

{% hint style="success" %}
**Overall**: Entropy is calculated using cumulative frequencies of peaks across the entire signal.

**Frame-wise**: Entropy is calculated at each frame individually and then average across time is computed.

Note that Overall entropy will be always much higher than thr entropy at each frame, as well as, their average. At each frame only a few expression coefficients are expected to be activated, which yields very low entropy values.&#x20;
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

