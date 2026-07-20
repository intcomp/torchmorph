# Morphological Operations

All image and volume operators on this page expect CUDA tensors shaped as
`(B, C, Spatial...)`. Add batch and channel dimensions before processing a
single image or volume.

## Structuring elements

::: torchmorph.generate_binary_structure

::: torchmorph.iterate_structure

## Binary morphology

::: torchmorph.binary_dilation

::: torchmorph.binary_erosion

::: torchmorph.binary_propagation

::: torchmorph.binary_fill_holes

::: torchmorph.binary_hit_or_miss

::: torchmorph.binary_opening

::: torchmorph.binary_closing

## Grayscale morphology

::: torchmorph.grey_dilation

::: torchmorph.grey_erosion

::: torchmorph.grey_opening

::: torchmorph.grey_closing

::: torchmorph.morphological_gradient

::: torchmorph.morphological_laplace

::: torchmorph.white_tophat

::: torchmorph.black_tophat
