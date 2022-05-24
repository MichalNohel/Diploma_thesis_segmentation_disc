clear all
close all
clc
path_to_data='D:\Diploma_thesis_segmentation_disc\HRF\';
images_file = dir([path_to_data '\images\*.jpg']);
path_to_disc=[path_to_data 'Disc\'];
path_to_cup=[path_to_data 'Cup\'];
num_of_img=length(images_file);

%%
i=1;
disp(images_file(i).name)
image=imread([images_file(i).folder '\' images_file(i).name ]);
imshow(image)
%% disc
imageSegmenter(image)
%%
imwrite(MaskDisk01,[path_to_disc images_file(i).name(1:end-4) '_disc.png'])

%% cup
imageSegmenter(image)
%%
imwrite(MaskCup01,[path_to_disc images_file(i).name(1:end-4) '_cup.png'])


