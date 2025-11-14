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

# Coordination

<h2 align="center">Social Coordination</h2>

Bitbox can measure how closely two sets of signals move together over time. This functionality is useful for studying the coordination of facial expressions, head movements, or body actions (coming soon), either within a single person or between multiple people.

The main difference between coordination and [imitation](imitation.md) is that, in coordination, time lags are allowed in both directions—that is, each signal can lead or follow the other—making it suitable for analyzing mutual synchronization rather than one-way following.

This function accepts landmarks ([2D ](../overview/outputs.md#id-2d-face-landmarks)or [3D](../overview/outputs.md#id-3d-face-landmarks)), head pose, or facial expressions ([global](../affective-expressions/facial-expressions.md#expression-related-global-deformations) or [local](../affective-expressions/facial-expressions.md#localized-expression-units)). It computes a windowed, lagged cross-correlation between all pairs of signals.

```python
from bitbox.social import coordination

# output directory
output_dir = 'output'

# define a face processor
processor = FP(runtime='bitbox:latest')

# get all facial signals for person A
processor.io(input_file='participant_a.mp4', output_dir=output_dir)
rect_a, land_a, exp_glob_a, pose_a, land_can_a, exp_loc_a = processor.run_all(normalize=True)

# get all facial signals for person B
processor.io(input_file='participant_b.mp4', output_dir=output_dir)
rect_b, land_b, exp_glob_b, pose_b, land_can_b, exp_loc_b = processor.run_all(normalize=True)

# quantify coordination
corr_mean, corr_std, corr_lag = coordination(exp_glob_a, exp_glob_b, width=1.1, step=0.5, fps=30)
```

You can compute coordination within a single person and a single set of signals, such as coordination of different facial expressions. In this case, the output matrix is symmetric, since each signal is compared with every other signal in both directions.

```python
# quantify within-person coordination of facial expressions
corr_mean, corr_std, corr_lag = coordination(exp_glob_a, exp_glob_a)

# quantify within-person coordination of facial expressions vs head pose
corr_mean, corr_std, corr_lag = coordination(exp_glob_a, pose_a)
```

