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

# Smoothness

<h2 align="center">Smoothness of Motion</h2>

Bitbox computes common metrics of smoothness, including averge jerk and log dimensionless jerk.

{% hint style="success" %}
**Average jerk**: Mean magnitude of each object’s jerk (third-order derivative of position), summarizing average abruptness of motion.

**Log dimensionless jerk (LDJ)**: Negative natural log of the time-normalized, amplitude-normalized squared-acceleration integral, giving a dimensionless/unitless smoothness index.

$$\mathrm{LDJ} = -\ln\left(\frac{T^{3}}{v_{\text{peak}}^{2}}\displaystyle\int_{0}^{T}\lVert \mathbf{a}(t) \rVert^{2},dt\right)$$
{% endhint %}

Similar to [kinematics](kinematics.md) measures, you can compute these variables for face bounding boxes, head pose, facial landmarks, or body joints (coming soon). When using landmarks, variables are calculated for each landmark separately.

<pre class="language-python"><code class="lang-python">from bitbox.biomechanics import motion_smoothness

# run the processor
rects, lands, exp_global, pose, lands_can, exp_local = processor.run_all()

<strong># quantify abruptness of motion for face rectangles
</strong>jerk, ldj = motion_smoothness(rects)

# quantify abruptness of motion for head pose
jerk, ldj = motion_smoothness(pose)

# quantify abruptness of motion for facial landmarks
jerk, ldj = motion_smoothness(lands)
</code></pre>

With head pose, you can either use translation coordinates or rotation angles (yaw, pitch, roll).

```python
# using translation coordinates
jerk, ldj = motion_smoothness(pose)

# using rotation angles
jerk, ldj = motion_smoothness(pose, angular=True)
```

