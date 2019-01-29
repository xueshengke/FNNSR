import sys
# caffe path
sys.path.append('/ext/xueshengke/caffe-1.0/python')
import caffe
from caffe import layers as L, params as P, to_proto

################################################################################
def weighting(bottom):
    ## pointwise product
    top = L.ElementWiseProduct(bottom, bias_term=False,
                               weight_filler=dict(type='constant', value=0.1),
                               bias_filler=dict(type='constant', value=0))
    return top

################################################################################
def weight_smooth(bottom, kernel):
    ## pointwise product
    model = weighting(bottom)
    ## smoothing convolution
    output = L.Convolution(model, num_output=1,
                           kernel_size=kernel, stride=1, pad=(kernel-1)/2,
                           dilation=1, weight_filler=dict(type='msra'),
                           bias_term=False, bias_filler=dict(type='constant'))
    return output
