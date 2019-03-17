clear;close all;
%% settings
scale = 4;
folder = sprintf('../_ILSVRC2012_img_val_%d', scale);
save_filea_name = sprintf('_ILSVRC2012_%d.mat', scale);

filepaths = dir(fullfile(folder, '*.JPEG'));
%% ini
len = length(filepaths);
numOfmissing = 0;
%img_valid_y = zeros(len, 17*scale, 17*scale);
%img_low_y = zeros(len, 17, 17);
%img_bic_y = zeros(len, 17*scale, 17*scale);

%% generate data
for i = 1 : len
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
    
    img_valid_y = sprintf('img_valid_y_%d', i);
    img_low_y = sprintf('img_low_y_%d', i);
    img_bic_y = sprintf('img_bic_y_%d', i);
    
    if img_numel == 3
        img_valid_ycbcr = rgb2ycbcr(img_valid);
        img_low_ycbcr = rgb2ycbcr(img_low);
        img_bic_ycbcr = rgb2ycbcr(img_bic);
        
        eval([img_valid_y, '= img_valid_ycbcr(:,:,1);']);
        eval([img_low_y, '= img_low_ycbcr(:,:,1);']);
        eval([img_bic_y, '= img_bic_ycbcr(:,:,1);']);
        %img_valid_y(i) = img_valid_ycbcr(:,:,1);
        %img_low_y(i) = img_low_ycbcr(:,:,1);
        %img_bic_y(i) = img_bic_ycbcr(:,:,1);
    else
        eval([img_valid_y, '= img_valid;']);
        eval([img_low_y, '= img_low;']);
        eval([img_bic_y, '= img_bic;']);
        %img_valid_y(i) = img_valid;
        %img_low_y(i) = img_low;
        %img_bic_y(i) = img_bic;
    end
    output = sprintf('===> No. %d prepared, total %d, remains %d', i, len, len-i);
    disp(output);
end
save(save_filea_name);