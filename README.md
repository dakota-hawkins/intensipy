# intensipy
Normalize intensity values in 3D image stacks.

# Current Methods

## 1. [Intensify3D](https://github.com/nadavyayon/Intensify3D)

Python implementation of the Intensify3D algorithm originally developed by [Yoyan et al](https://www.nature.com/articles/s41598-018-22489-1). There are some minor adjustments:

  1. Pixels that are quantile normalized are subsequently smoothed using they Savitzky-Galoy method outlined in the original paper. In practice this was necessary to reduce artefact noise.
  2. Tissue detection is not currently supported.


### Results Comparison

#### Original Paper Results
![Original](https://raw.githubusercontent.com/nadavyayon/Intensify3D/master/Examples/Montage2-01.jpg)

#### Intensipy Results
![Artificial Data](images/artificial_results.png)


# Issues:
1. Produces low-contrast images that benefit from scaling each z-slice after initial normalization.

# References

1.Yayon, N. et al. Intensify3D: Normalizing signal intensity in large heterogenic image stacks. Scientific Reports 8, 4311 (2018).
