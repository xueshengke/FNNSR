%% generate hdf5 image file for validation
clear; close all;
addpath ../util;
addpath(genpath(cd))

%% settings
folder = '/ext/xueshengke/dataset/validation';
savepath = '/ext/xueshengke/test_FNNSR';
scale = 2;  % scale factor: 2, 3, 4
patch_size = [360, 480]; % cannot be too large, it requires huge memory
stride = patch_size;
batch_size = 2;

%% initialization
data = [];
label = [];

%% generate data
file_bmp = dir(fullfile(folder, '*.bmp'));
file_png = dir(fullfile(folder, '*.png'));
filepaths = [file_bmp; file_png];
total_num = length(filepaths);

for i = 1 : total_num
    fprintf('%d / %d %s\n', i, total_num, filepaths(i).name);
    image = imread(fullfile(folder, filepaths(i).name));
    image = im2double(image);
    if size(image, 3) > 1
        image = rgb2ycbcr(image);
        image = image(:, :, 1); % only use Y channel
    end
    
    image = modcrop(image, scale);
    image_bic = imresize(imresize(image, 1/scale, 'bicubic'), scale, 'bicubic');
    image_res = image - image_bic;
    image_bic_batch = imtobatch(image_bic, patch_size);
    image_res_batch = imtobatch(image_res, patch_size);
    image_bic_batch_ht = hartleyTrans(image_bic_batch, 't');
    image_res_batch_ht = hartleyTrans(image_res_batch, 't');

    data = cat(3, data, image_bic_batch_ht);
    label = cat(3, label, image_res_batch_ht);
end
dim = size(data);
count = dim(3);
order = randperm(count);
data = data(:, :, order);
label = label(:, :, order);
data = reshape(data, [dim(1) dim(2) 1 dim(3)]);
label = reshape(label, [dim(1) dim(2) 1 dim(3)]);

%% writing to HDF5
chunksz = batch_size;

created_flag = false;
totalct = 0;
savefile = [savepath, '_x' num2str(scale) '.h5'];

for batchno = 1:floor(count/chunksz)
    last_read = (batchno-1)*chunksz;
    batchdata = data(:, :, :, last_read+1:last_read+chunksz); 
    batchlabs = label(:, :, :, last_read+1:last_read+chunksz);
    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savefile, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savefile);