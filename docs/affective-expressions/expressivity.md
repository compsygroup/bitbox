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

Bitbox quantifies overall facial expressivity by analyzing expression-related activations, including their counts, magnitudes, and ranges. These statistics describe how expressive a face is on average. They are also useful as covariates, since certain measures, such as asymmetry, can be influenced by overall expressivity. For example, a wider range of expressions may lead to higher asymmetry magnitudes.

This function only accepts [global](facial-expressions.md#expression-related-global-deformations) or [local](facial-expressions.md#localized-expression-units) facial expressions. It computes expressivity stats for each  expression coefficient independently.&#x20;

```python
from bitbox.expressions import expressivity

# estimate global expression coefficients
exp_global, pose, lands3D = processor.fit()

# compute expressivity stats
expressivity_stats = expressivity(exp_global, scales=6)
```

The computation is performed at several temporal scales to capture expressions that unfold at different speeds—for example, slow, moderate, or rapid changes in facial activity. A temporal scale represents the approximate duration of an expression event. For instance, if the scale is 1 second, the algorithm looks for activations (peaks) in the expression signal that last about one second from start to finish. At each scale, a peak detection algorithm identifies these activations, and their number and magnitude are used to compute summary metrics. The output is a list of Pandas `DataFrame`s, one for each expression signal. Each `DataFrame` contains six metrics per temporal scale: _frequency_, _density_, _mean_, _standard deviation_, _minimum_, and _maximum_.

{% hint style="success" %}
**Frequency**: Number of activations (peaks) observed in the expression signal.

**Density**: Cummulative sum of activation magnitudes divided by the signal length.

**Mean**: Mean magnitude of activations.

**Std**: Standard deviation of activation magnitudes.

**Min**: Minimum magnitude of activations.&#x20;

**Max**: Maximum magnitude of activations.&#x20;
{% endhint %}

```
       scale   frequency      density      mean       std       min       max
0       0.1        11         0.003595   0.49841  0.142492  0.278971  0.731558
1      0.88         2         0.000888    0.6774   0.05611  0.621291   0.73351
2      1.66         2         0.000715  0.545349  0.032984  0.512366  0.578333
3      2.44         2         0.000829  0.632378  0.011088  0.621291  0.643466
4      3.22         2         0.000792  0.604034  0.039433  0.564601  0.643466
5       4.0         2         0.000693  0.528063   0.05027  0.477794  0.578333
```

### Temporal Scales

Bitbox can compute expressivity across multiple temporal scales, where each scale represents a specific time window in seconds. This allows it to capture both short, transient expressions and longer, sustained changes. For example, if the scale is 2 seconds, Bitbox looks for expression activations that evolve over roughly two seconds—such as a gradual smile forming and fading—rather than brief micro-expressions. The figure below shows an expression signal decomposed into several temporal scales, along with detected peaks that correspond to expression events occurring at those different speeds.

<figure><img src="../.gitbook/assets/multiscale (1).png" alt=""><figcaption></figcaption></figure>

{% hint style="info" %}
**What scales to consider?**

The choice of temporal scales depends on both your dataset and your research question. You should first decide what types of expressions you want to capture—whether you are interested in quick, subtle changes (like micro-expressions) or slower, more deliberate facial movements (such as a gradual smile or frown).

Shorter scales, such as 0.1–0.5 seconds, are better for detecting brief or fine-grained activations that reflect rapid muscle twitches or transient changes in expression. Longer scales, such as 1–3 seconds, capture sustained or slowly evolving expressions that may indicate intentional or emotional responses.

When in doubt, it is often useful to analyze multiple scales to see how expressivity patterns vary across them. You can later select the most informative scales for your analysis or report results across several scales to capture the full temporal dynamics of facial behavior.
{% endhint %}

You can specify the desired scales through the `scales` parameter. This can be done either by providing:

* an integer, which defines how many evenly spaced scales to generate between 0.1 and 4 seconds,
* a list of explicit durations, giving you full control over which time windows to analyze.

```python
# defining the number of scales
# resulting scales: 0.1 , 0.88, 1.66, 2.44, 3.22, 4.
expressivity_stats = expressivity(exp_global, scales=6)

# defining the scales explicitly
expressivity_stats = expressivity(exp_global, scales=[0.5, 1, 1.5, 2])
```

For the scales to represent real durations in seconds, the frame rate of the signal must be correctly specified. The default is 30 frames per second, but you should adjust this value to match the actual frame rate of your data to ensure accurate timing.

```python
# setting the frame rate of the signal
expressivity_stats = expressivity(exp_global, scales=6, fps=30)

```

If the `scales` parameter is not specified, the computation runs at the finest temporal scale, corresponding to the sampling rate of the original signal. In this case, every local peak in the signal is treated as an activation. See the image below for an illustration.

```python
# working with the original signal with no temporal scales
expressivity_stats = expressivity(exp_global, scales=None)

# you can also simply skip the parameter, which generates the same results
expressivity_stats = expressivity(exp_global)
```

```
       scale   frequency      density      mean       std       min       max
0.     None        4          0.001738     0.66254  0.063133  0.58123  0.731558
```

You can also combine activations detected at different time scales to produce overall expressivity statistics based on the aggregated peaks. When peaks from multiple scales occur very close in time—within the time window of the smallest scale—the algorithm keeps only the one with the highest relative magnitude, calculated within its own scale. This ensures that overlapping detections represent a single, dominant activation rather than multiple redundant ones.&#x20;

<pre class="language-python"><code class="lang-python"><strong># disable normalization
</strong>expressivity_stats = expressivity(exp_global, scales=[0.5, 1, 1.5, 2], aggregate=True)
</code></pre>

```
       scale       frequency      density       mean      std       min       max
0.   Agg 0.1-4.0       6          0.001901     0.48313  0.138442  0.278971  0.643466
```

{% hint style="warning" %}
Using the `aggregate` parameter is not the same as simply combining statistics from multiple scales. First, the algorithm creates a new, unified set of peaks by merging detections from different scales and removing redundant ones that occur too close in time. After this consolidation, expressivity statistics are computed from this single, combined list of peaks, ensuring that overlapping activations across scales are represented only once.
{% endhint %}

{% hint style="warning" %}
Using the `aggregate` parameter does not produce the same result as setting the `scales` parameter to `None`. When `scales=None`, statistics are computed directly from the original signal without considering multiple temporal scales or merging peaks. In contrast, the aggregate option first merges peaks detected across scales, removes redundant ones, and then computes expressivity statistics from this combined set.
{% endhint %}

{% hint style="warning" %}
If you only need a rough estimate of the average level of expression activation, you can simply take the mean of the expression signals themselves without using the multiscale analysis provided by the expressivity function:

```python
signals = exp_global['data']
rough_expressivity = signals.mean()
```

This approach is similar to using `scales=None` in the expressivity function, but it may produce slightly smaller values because it averages the entire signal, whereas the expressivity function computes the mean based only on detected peaks.
{% endhint %}
