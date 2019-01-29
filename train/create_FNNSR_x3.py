# This file creates FNNSR prototxt files: train and  test
from __future__ import print_function
import sys
# caffe path
sys.path.append('/ext/xueshengke/caffe-1.0/python')
sys.path.append('util')
import caffe
from caffe import layers as L, params as P, to_proto
from function import weighting, weight_smooth

################################################################################
# change filename here
train_net_path = 'train_x3.prototxt'
test_net_path = 'test_x3.prototxt'
train_data_path = 'examples/FNNSR/train_data_x3.txt'
test_data_path = 'examples/FNNSR/test_data_x3.txt'

# parameters of the network
batch_size_train = 64
batch_size_test = 2
scale = 3
depth = 4
channel = 5
kernel = 5

################################################################################
# define the network for training and validation
def train_FNNSR(train_data=train_data_path, test_data=test_data_path,
                batch_size_train=batch_size_train, batch_size_test=batch_size_test,
                depth=depth, channel=channel, kernel=kernel):
    net = caffe.NetSpec()
    net.data, net.label = L.HDF5Data(hdf5_data_param={'source': train_data,
        'batch_size': batch_size_train}, include={'phase': caffe.TRAIN}, ntop=2)
    train_data_layer = str(net.to_proto())
    net.data, net.label = L.HDF5Data(hdf5_data_param={'source': test_data,
        'batch_size': batch_size_test}, include={'phase': caffe.TEST}, ntop=2)

    ## pointwise product
    net.layer_out = weight_smooth(net.data, kernel)
    for i in range(channel-1):
        net.model = weight_smooth(net.data, kernel)
        net.layer_out = L.Eltwise(net.layer_out, net.model)

    net.S = net.layer_out

    for j in range(depth-1):
        net.layer_in = net.layer_out
        ## pointwise product
        net.layer_out = weight_smooth(net.layer_in, kernel)
        for i in range(channel - 1):
            net.model = weight_smooth(net.layer_in, kernel)
            net.layer_out = L.Eltwise(net.layer_out, net.model)
        net.S = L.Eltwise(net.S, net.layer_out)

    net.P = weighting(net.S)
    net.loss = L.EuclideanLoss(net.P, net.label)

    return train_data_layer + str(net.to_proto())

################################################################################
# deploy the network for test; no data, label, loss layers
def test_FNNSR(depth=depth, channel=channel, kernel=kernel):
    net = caffe.NetSpec()

    net.data = L.Input(shape=dict(dim=[1, 1, 2*depth+1, 2*depth+1]), ntop=1)

    ## pointwise product
    net.layer_out = weight_smooth(net.data, kernel)
    for i in range(channel - 1):
        net.model = weight_smooth(net.data, kernel)
        net.layer_out = L.Eltwise(net.layer_out, net.model)

    net.S = net.layer_out

    for j in range(depth - 1):
        net.layer_in = net.layer_out
        ## pointwise product
        net.layer_out = weight_smooth(net.layer_in, kernel)
        for i in range(channel - 1):
            net.model = weight_smooth(net.layer_in, kernel)
            net.layer_out = L.Eltwise(net.layer_out, net.model)
        net.S = L.Eltwise(net.S, net.layer_out)

    net.P = weighting(net.S)
    # net.loss = L.EuclideanLoss(net.P, net.label)

    return net.to_proto()

################################################################################
if __name__ == '__main__':
    # write train_val network
    with open(train_net_path, 'w') as f:
        print(str(train_FNNSR()), file=f)
    print('create ' + train_net_path)

    # write test network
    with open(test_net_path, 'w') as f:
        f.write('name: "FNNSR'+'_scale='+str(scale)+'_depth='+str(depth)
                +'_kernel='+str(kernel)+'_channel='+str(channel)+'"\n')
        print(str(test_FNNSR()), file=f)
    print('create ' + test_net_path)
