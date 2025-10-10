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

# Expressivity

<h2 align="center">Overall Expressivity</h2>

Bitbox measures overall expressivity of expressions by counting number of activations of expression signals, their magnitudes, ranges, etc. These stats can be used to study how expressive a face is on average. These metrics are also useful as covariates as certain other measures, such as [asymmetry](symmetry.md), can be affected by overall expressivity (higher range of expressions may lead to higher magnitudes of asymmetry).&#x20;

This function only accepts [global](localized-expression-units.md#expression-related-global-deformations) or [local](localized-expression-units.md#localized-expression-units) facial expressions. It computes expressivity stats for each  expression coefficient independently.&#x20;

```python
from bitbox.expressions import expressivity

# estimate global expression coefficients
exp_global, pose, lands3D = processor.fit()

# compute expressivity stats
expressivity_stats = expressivity(exp_global, scales=6)
```

The computation is performed at multiple temporal scales to capture expressions with different temporal characteristics (slowly evolving, fast, very rapid, etc.). At each scale, activations of the expression signal are detected using a peak detection algorithm. The number of such peaks and their magnitudes are used for computing metrics. The output is a list of Pandas `DataFrame`, one for each expression signal. Each `DataFrame` includes six metrics per temporal scale: _frequency_, _density_, _mean_, _std_, _min_, _max_.

{% hint style="success" %}
**Frequency**: Number of activations (peaks) observed in the expression signal.

**Density**: Cummulative sum of activation magnitudes divided by the signal length.

**Mean**: Mean magnitude of activations.

**Std**: Standard deviation of activation magnitudes.

**Min**: Minimum magnitude of activations.&#x20;

**Max**: Maximum magnitude of activations.&#x20;
{% endhint %}

<pre><code>
<strong>          eye	           brow	          nose	          mouth	         overall
</strong>0	0.292884	0.265071	0.041300	0.075939	0.211353
1	0.210291	0.260469	0.045354	0.097847	0.195869
2	0.283825	0.156247	0.000000	0.000000	0.114181
3	0.327324	0.119029	0.094368	0.040651	0.182186
4	0.196593	0.167545	0.196875	0.051681	0.171618
...	...	...	...	...	...
</code></pre>

### Temporal Scales

Bitbox can compute expressivity at multiple time scales. A scale corresponds to a different window of interest in seconds. For example, if the scale is 1 second, that means we are detecting activations (peaks) that takes roughly 1 second from start to end. In the figure below, you see an original expression signal and its decomposition into signals at different temporal scales, along with peaks detected at each scale.&#x20;

IMAGE: Multi-scale decomposition

You can define the scales of interests using the `scales` parameter. You can either define the scales using an integer, the number of equally-spaced scales between 0.1 second and 4 second, or using an explicit list of durations.&#x20;

```python
# defining the number of scales
# resulting scales: 0.1 , 0.88, 1.66, 2.44, 3.22, 4.
# computed using np.linspace(0.1, 4, 6)
expressivity_stats = expressivity(exp_global, scales=6)

# defining the scales explicitly
expressivity_stats = expressivity(exp_global, scales=[0.5, 1, 1.5, 2])
```

For the scales to correspond to actual seconds, the frame rate of the signal must be set accurately. The default value is 30 frames per second.

```python
# setting the frame rate of the signal
expressivity_stats = expressivity(exp_global, scales=6, fps=30)

```

If the `scales` parameter is skipped (default), the computation is performed at the lowest temporal scale (sampling rate of the original signal), and every single peak in the signal is considered. See the image below for an example.

```python
# working with the original signal with no temporal scales
expressivity_stats = expressivity(exp_global, scales=None)

# you can also simply skip the parameter, which generates the same results
expressivity_stats = expressivity(exp_global)
```

IMAGE: No-scales

You can also aggregate activations across multiple time scales and generate expressivity stats for the aggregate peaks. If there are very close (within the time window of the lowest scale) peaks at multiple scales, the underlying algorithm considers only the one with the highest relative magnitude (computed within its own scale).&#x20;

<pre class="language-python"><code class="lang-python"><strong># disable normalization
</strong>expressivity_stats = expressivity(exp_global, scales=[0.5, 1, 1.5, 2], aggregate=True)
</code></pre>

<pre><code>
<strong>          eye	           brow	          nose	          mouth	         overall
</strong>0	0.292884	0.265071	0.041300	0.075939	0.211353
1	0.210291	0.260469	0.045354	0.097847	0.195869
2	0.283825	0.156247	0.000000	0.000000	0.114181
3	0.327324	0.119029	0.094368	0.040651	0.182186
4	0.196593	0.167545	0.196875	0.051681	0.171618
...	...	...	...	...	...
</code></pre>

{% hint style="warning" %}
&#x20;Using the `aggregate` parameter is not same as simply adding stats from multiple scales. First, a new, combined set of peaks are generated by merging peaks of multiple scales and eliminating the redundant ones that are very close in time. Then, expressivity stats are generated for this single list of peaks.
{% endhint %}

{% hint style="warning" %}
Using the `aggregate` parameter does not produce the same result with setting the `scales` parameter `None`. The latter will generate stats simply using the original signal itself.&#x20;
{% endhint %}

IMAGE: Aggregate
