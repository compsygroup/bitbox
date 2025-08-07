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

# Relative Motion

<h2 align="center">Relative Motion</h2>

Bitbox calculates displacement statistics relative to a reference frame. It measures how much an object moves with respect to its location at this reference frame, including metrics of minimum, average, standard deviation, and maximum displacement.

{% hint style="success" %}
This function can quantify, for example, how much the mouth corners move from their position in a neutral face.
{% endhint %}

Similar to [kinematics](kinematics.md) and [smoothness](smoothness.md) measures, you can compute these stats for face bounding boxes, head pose, facial landmarks, or body joints (coming soon). When using landmarks, variables are calculated for each landmark separately.

```python
# run the processor
rects, lands, exp_global, pose, lands_can, exp_local = processor.run_all()

from bitbox.biomechanics import relative_motion

# motion stats for facial landmarks relative to the neutral frame
mind, avgd, stdd, maxd = relative_motion(lands, reference=[0])
```

The `reference` parameter can either be a list of frame indices or the string "mean" (default), to calculate statistics based on the object's average coordinates over time. You can provide multiple indices, allowing statistics to be calculated for each reference frame individually; average statistics across these frames will be provided. This approach is particularly useful if you have several frames of a neutral face, which helps reduce pixel noise in your data.

```python
# motion stats for facial landmarks relative to multiple neutral frames
mind, avgd, stdd, maxd = relative_motion(lands, reference=[0, 13, 57])

# motion stats for facial landmarks relative to average coordinates
mind, avgd, stdd, maxd = relative_motion(lands, reference='mean')

# 'mean' is the default value, so you can skip it
mind, avgd, stdd, maxd = relative_motion(lands)
```

