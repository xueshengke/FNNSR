%% extract parameters from caffe model and save them as MAT files
close all; clear; clc;
setenv('LC_ALL','C')    % remove all local configurations
setenv('GLOG_minloglevel','2')  % remove any log when loading caffe modules
addpath '/ext/xueshengke/caffe-1.0/matlab'; % change to your caffe path

%% parameters, change settings if necessary
gpu_id = 0;
% caffe.set_mode_cpu(); % for CPU
caffe.set_mode_gpu(); % for GPU
caffe.set_device(gpu_id);

model = 'test_x3.prototxt';
weights = 'model/iter_5e4_s3_d4_c5_k5.caffemodel';

scale = 3;
depth = 4;
channel = 5;
kernel = 5;
save_file = ['FNNSR_x' num2str(scale) '_d' num2str(depth) '_c' num2str(channel) ...
    '_k' num2str(kernel) '.mat'];
fnnsr_params = cell(depth, channel, 2);
net = caffe.Net(model, weights, 'test');

for i = 1 : depth
for j = 1 : channel
    id = (i-1)*channel + j;
    fnnsr_params{i,j,1} = net.layers(['ElementWiseProduct' num2str(id)]).params(1).get_data();
    if id < depth * channel
    fnnsr_params{i,j,2} = net.layers(['Convolution' num2str(id)]).params(1).get_data();
    else
    fnnsr_params{i,j,2} = net.layers('model').params(1).get_data();
    end
end
end
add_layer = net.layers('P').params(1).get_data();

save(save_file, 'fnnsr_params', 'add_layer');
caffe.reset_all();