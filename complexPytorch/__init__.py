
"""
Library for complex valued neural networks in pytorch.
Builds on top of the native complex number support in pytorch.
Builds on top of https://github.com/wavefrontshaping/complexPyTorch, and adds some functionalities.

This is the base-file to import from.
Function- / Layer- Files have more, but redundant features.

If a function is not in here, it is probably implemented in normal pytorch and this can by used.
"""

from complexPytorch.complexLayers import (
    ComplexDropout,
    ComplexDropout2d,
    ComplexMaxPool2d,
    ComplexAvgPool2d, ComplexAdaptiveAvgPool2d,
    ComplexReLU, zReLU, modReLU,
    ComplexPReLU,
    ComplexGELU,
    ComplexLinear, ComplexLinearTwice, CombineComplexLinear,
    ComplexConv2d, ComplexConv2dTwice,
    ComplexConvTranspose2d, ComplexConvTranspose2dTwice,
    NaiveComplexLayerNorm,
    NaiveComplexBatchNorm1d, ComplexBatchNorm1d,
    NaiveComplexBatchNorm2d, ComplexBatchNorm2d,
    ComplexBasicMultiheadAttention,
    ComplexGRUCell, ComplexGRU, ComplexBNGRUCell, ComplexLSTM #except BNGRUCell all exist in normal pytorch as complex
)


from complexPytorch.complexFunctions import (
    complex_normalize,
    complex_opposite,
    complex_upsample, complex_upsample2
)

from complexPytorch.complexSoftmax import (
    softmax_real_with_abs, softmax_real_with_avg, softmax_real_with_mult, softmax_of_softmax_real_with_mult, softmax_of_softmax_real_with_avg,
    softmax_real_with_avg_polar, softmax_real_with_polar, softmax_complex, softmax_complex_max_diff, softmax_complex_split
)

from complexPytorch.complexLoss import (
    CrossEntropyComplex, CrossEntropyComplexTwice
)


    

