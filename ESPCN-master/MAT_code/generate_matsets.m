clear;close all;
%% settings
folder = '../_ILSVRC2012_img_val_4';
typeOfimage = '*.JPEG';
outputpath = 'mat_ImageNet50000';
scale = 4;

%% Initializing
if ~exist(outputpath, 'dir')
    mkdir(outputpath)
end
filepaths = dir(fullfile(folder, '*JPEG'));
numOfmissing = 0;

%% generate data
for i = 1 : length(filepaths)
    try
        img = imread(fullfile(folder,filepaths(i).name));
    catch
        numOfmissing = numOfmissing + 1;
        continue;
    end
    %img = imread(fullfile(folder,filepaths(i).name));
    img_numel = numel(size(img));
    img_valid = modcrop(img, scale);
    img_valid = double(img_valid);
    img_low = imresize(img_valid, 1/scale, 'bicubic');
    img_bic = imresize(img_low, scale, 'bicubic');
    if img_numel == 3
        img_valid_ycbcr = rgb2ycbcr(img_valid);
        img_low_ycbcr = rgb2ycbcr(img_low);
        img_bic_ycbcr = rgb2ycbcr(img_bic);

        img_valid_y = img_valid_ycbcr(:,:,1);
        img_low_y = img_low_ycbcr(:,:,1);
        img_bic_y = img_bic_ycbcr(:,:,1);
    else
        img_valid_y = img_valid;
        img_low_y = img_low;
        img_bic_y = img_bic;
    end
    filename = sprintf('%s/%s.mat',outputpath, filepaths(i).name);
    save(filename, 'img_valid_y', 'img_low_y', 'img_bic_y');
    out = sprintf('No. %d sub-image done.', i);
    disp(out);
end
output = sprintf('Total %d sub-images done, missing %d images.', length(filepaths), numOfmissing);
disp(output);