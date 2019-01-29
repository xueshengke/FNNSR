%% test elementwise product layer in Caffe
close all; clear; clc;

setenv('LC_ALL','C')    % remove all local configurations
setenv('GLOG_minloglevel','2')  % remove any log when loading caffe modules
addpath '/ext/xueshengke/caffe-1.0/matlab'; % change to your caffe path
addpath(genpath('util'));

%% parameters, change settings if necessary
gpu_id = 0;
% caffe.set_mode_cpu(); % for CPU
caffe.set_mode_gpu(); % for GPU
caffe.set_device(gpu_id);

weights = ' ';
model = 'test_elementwise_product.prototxt';

input = ones(360, 480);
output = FNNSRNetNoWeight( model, input );

diff = input * 0.5 - output;
sum(diff(:))

%% visualize the images
figure; 
subplot(1,2,1); imshow(input); title('input image');
subplot(1,2,2); imshow(output); title('output image');
