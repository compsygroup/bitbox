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

# Visualization

<h2 align="center">Visualization and Plotting</h2>

Bitbox offers advanced visualization features designed specifically for facial and body analysis. Its plotting module uses Plotly, an open-source visualization library, to create interactive, web-based plots that make data exploration intuitive and flexible.

Currently, Bitbox supports visualizing face rectangles, facial landmarks (both 2D and 3D), head pose, and expressions (both global and localized).

<pre class="language-python"><code class="lang-python"># define input file and output directory
input_file = 'data/elaine.mp4'
output_dir = 'output'

# define a face processor
processor = FP(runtime='bitbox:latest')

# set input and output
processor.io(input_file=input_file, output_dir=output_dir)
<strong>
</strong><strong># detect faces
</strong>rects = processor.detect_faces()

# detect 2D landmarks
lands = processor.detect_landmarks()

# compute global expressions, pose, and 3D canonicalized landmarks
exp_global, pose, lands_can = processor.fit()
</code></pre>

The visualization output is saved as an HTML/JavaScript file in your specified output directory. To view it, simply open the file in any modern web browser such as Chrome, Edge, or Firefox.

When you open the file, you may notice a short delay before the visualization appears. This is normal—the included JavaScript code generates the video frames dynamically in your browser. Depending on your hardware and browser, it may take a few seconds to fully load. Once loaded, you can freely zoom, pan, and interact with the visualization to explore the data in detail.

```python
# visualize landmarks at random poses
processor.plot(lands, pose=pose)
```

<figure><img src="../.gitbook/assets/visual01.png" alt=""><figcaption></figcaption></figure>

```python
# visualize landmarks with rectangles overlayed
processor.plot(lands, overlay=[rects], video=True) 
```

<figure><img src="../.gitbook/assets/visual02.png" alt=""><figcaption></figcaption></figure>

```python
# visualize expressions
processor.plot(exp_global, overlay=[rects, lands], video=True)
```

<figure><img src="../.gitbook/assets/visual03.png" alt=""><figcaption></figcaption></figure>
