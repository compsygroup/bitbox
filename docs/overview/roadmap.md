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

# Roadmap

## Roadmap <a href="#contributing" id="contributing"></a>

Bitbox is continually evolving, with our team actively working on new features and enhancements. Given our limited resources, prioritization is key. Below are the current priorities for improvement, which we will keep updating as needed.

### Core Developments

We plan to implement the following updates over the next year:

* [ ] **Add support for OpenPose (body backend)**: We plan to add a new backend processor for analyzing body actions. OpenPose is the most widely used processor for body joint detection and tracking.
* [ ] **Add suport for OpenFace (face backend)**: A new face processing backend will be implemented. OpenFace, a commonly used alternative to 3DI, will enable the estimation of Action Unit activations.
* [ ] A**dd support for TalkNET (speech backend)**: We will implement a speech behavior processor, TalkNET, to determine who is speaking and when. This will also allow the study of turn-taking patterns.
* [ ] **Add support for CPU version of 3DI-Lite**: Current face-processing backends require GPUs, limiting their use on computers without one. Introducing a CPU version of 3DI-Lite will broaden the usability and adoption of Bitbox.
* [ ] **Add visualization and reporting feature**: We plan to enhance Bitbox by incorporating visualization features to inspect generated results. Additionally, summary statistics will be introduced for inspection and reporting purposes.

### Community Developments

We contantly look forward to new feature suggestions and implementations from the community. There are many things you can contribute to. Below, we list a few ideas that we would love to be implemented in Bitbox.

* **New backend processors for facial, body, speech analysis, and more**. Bitbox is open to all behavioral measurement modalities, including heart rate and various wearables. Our core team currently lacks expertise in this area and welcomes community contributions.
* **New measurement/analysis funtions.** If you have specific behavioral measurements you're interested in, feel free to share your ideas—or even better, your code—for implementation in Bitbox.



