# TorchMorph

TorchMorph provides tensor-native morphological image operations and distance transforms for PyTorch workloads. Its public language follows SciPy ndimage where possible while making batch and channel dimensions explicit.

## Language

**Morphological Operation**:
An operation that transforms binary or grey-valued spatial samples by comparing each location with a neighborhood defined by a structuring element.
_Avoid_: filter, convolution

**Binary Morphology**:
Morphological operations where every non-zero input value is treated as foreground and zero is treated as background.
_Avoid_: boolean convolution, mask filtering

**Grey Morphology**:
Morphological operations over numeric intensities, where the result depends on ordered values rather than only foreground/background membership.
_Avoid_: grayscale filter

**Grey Erosion**:
A grey morphology operation that selects the minimum response from a neighborhood defined by a structuring element.
_Avoid_: minimum filter

**Grey Dilation**:
A grey morphology operation that selects the maximum response from a neighborhood defined by a structuring element.
_Avoid_: maximum filter

**Grey Opening**:
A grey morphology operation formed by grey erosion followed by grey dilation with the same structuring element.
_Avoid_: erosion-dilation filter

**Grey Closing**:
A grey morphology operation formed by grey dilation followed by grey erosion with the same structuring element.
_Avoid_: dilation-erosion filter

**Structuring Element**:
The neighborhood shape or weighted neighborhood used by a morphological operation.
_Avoid_: kernel, filter

**Footprint**:
A binary structuring element that selects which neighboring locations participate in a grey morphology operation.
_Avoid_: mask, stencil

**Feature Transform**:
The nearest-background coordinate result returned alongside a distance transform.
_Avoid_: index map, nearest point map

**Spatial Dimensions**:
The trailing dimensions of an input tensor that represent the image, volume, or higher-dimensional sample being transformed.
_Avoid_: data dimensions

**Batch-Channel Tensor**:
A tensor shaped as `(B, C, Spatial...)`, where `B` and `C` are independent leading dimensions and every `(B, C)` slice is transformed independently.
_Avoid_: image tensor, BCHW-only tensor
