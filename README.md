[![DOI](https://zenodo.org/badge/578652292.svg)](https://zenodo.org/doi/10.5281/zenodo.10982527)
# ConnectivityOnProjections

Official repo for the MICCAI 2022 paper: [Enforcing connectivity of 3D linear structures using their 2D projections](https://arxiv.org/pdf/2207.06832.pdf). Doruk Oner, Hussein Osman, Mateusz Kozinski, Pascal Fua.

<img width="1188" src="https://user-images.githubusercontent.com/78302409/213012884-ea6ded69-4b6c-4ef5-8ffe-aae503be789e.png">

**Fig. 1.** The intuition behind LTOPO. (a) In a perfect distance map, any path connecting pixels on the opposite sides of an annotation line (dashed, magenta) crosses a zero- valued pixel (red circle). (b) If a distance map has erroneously high-valued pixels along the annotation line, the maximin path (violet) between the same pixels crosses one of them (red circle). (c) The connectivity-oriented loss LTOPO is a sum of the smallest values crossed by maximin paths connecting pixels from different background regions. The background regions are computed by first dilating the annotation (dilated anno- tation shown in white), to accommodate possible annotation inaccuracy.

<p align="center">
  <img width="480" src="https://user-images.githubusercontent.com/78302409/213013013-63208489-fedf-4a5b-8479-2bebdc5b9d3c.png">
</p>

**Fig. 2.** Disconnections in 3D linear structures appear in at least two out of three orthogonal projections, unless the structure is occluded. 

This project was supported, in part, by the FWF Austrian Science Fund's Lise Meitner Program, project number M3374.
