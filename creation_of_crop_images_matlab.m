clear all
close all
clc
path_to_data='D:\Diploma_thesis_segmentation_disc';
output_image_size=[500,500];
sigma=60;
size_of_erosion=80;
percentage_number_train=0.8;


%% Dristi-GS - Expert 1
pom=52; % split to test and train dataset
images_file = dir([path_to_data '\Drishti-GS\Images\*.png']);
disc_file = dir([path_to_data '\Drishti-GS\Disc\expert1\*.png']);
cup_file = dir([path_to_data '\Drishti-GS\Cup\expert1\*.png']);
fov_file = dir([path_to_data '\Drishti-GS\Fov\*.png']);
path_to_crop_image=[path_to_data '/Data_500_500/'];


coordinates_dristi_GS=load('C:\Users\nohel\Desktop\Databaze_final\Drishti-GS\coordinates_dristi_GS.mat');
coordinates=coordinates_dristi_GS.coordinates_dristi_GS;

num_of_img=length(images_file);

for i=1:num_of_img
    %expert 1
    image=imread([images_file(i).folder '\' images_file(i).name ]);     
    mask_disc=imread([disc_file(i).folder '\' disc_file(i).name ]);  
    mask_cup=imread([cup_file(i).folder '\' cup_file(i).name ]);  
    fov=imread([fov_file(i).folder '\' fov_file(i).name ]);
    
    if i>=pom
        [center_new] = Detection_of_disc(image,fov,sigma,size_of_erosion);
        if mask_disc(center_new(2),center_new(1))~=1
            center_new(1)=coordinates(i,1);
            center_new(2)=coordinates(i,2);
        end
        [output_crop_image, output_mask_disc,output_mask_cup]=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new);
        imwrite(output_crop_image,[path_to_crop_image 'Train\Images_crop\' images_file(i).name])
        imwrite(output_mask_disc,[path_to_crop_image 'Train\Disc_crop\' disc_file(i).name])
        imwrite(output_mask_cup,[path_to_crop_image 'Train\Cup_crop\' cup_file(i).name])
    else
        imwrite(image,[path_to_crop_image 'Test\Images\' images_file(i).name])
        imwrite(mask_disc,[path_to_crop_image 'Test\Disc\' disc_file(i).name])
        imwrite(mask_cup,[path_to_crop_image 'Test\Cup\' cup_file(i).name])
    end
end

%%
figure
imshow(image)
hold on
stem(coordinates(i,1),coordinates(i,2),'g','MarkerSize',16,'LineWidth',2)
stem(center_new(1),center_new(2),'r','MarkerSize',16,'LineWidth',2)



%% Function
function[center_new] = Detection_of_disc(image,fov,sigma,velikost_erodovani)
image=rgb2xyz(im2double(image));
image=rgb2gray(image);
BW=imerode(fov,strel('disk',velikost_erodovani));
vertical_len=size(BW,1);
step=round(vertical_len/15);
BW(1:step,:)=0;
BW(vertical_len-step:vertical_len,:)=0;
image(~BW)=0;
img_filt=imgaussfilt(image,sigma);
img_filt(~BW)=0;
[r, c] = find(img_filt == max(img_filt(:)));
center_new(1)=c;
center_new(2)=r;
end

function [output_crop_image, output_mask_disc,output_mask_cup]=Crop_image(image,mask_disc,mask_cup,output_image_size,center_new)
    size_in_img=size(image);
    x_half=round(output_image_size(1)/2);
    y_half=round(output_image_size(2)/2);
    if ((center_new(2)-x_half)<0)
        x_start=1;
    elseif ((center_new(2)+x_half)>size_in_img(1))
        x_start=size_in_img(1)-output_image_size(1);
    else
        x_start=center_new(2)-x_half;
    end

    if ((center_new(1)-y_half)<0)
        y_start=1;
    elseif ((center_new(1)+y_half)>size_in_img(2))
        y_start=size_in_img(0)-output_image_size(2);
    else
        y_start=center_new(1)-y_half;
    end

    output_crop_image=image(x_start:x_start+output_image_size(1)-1,y_start:y_start+output_image_size(2)-1,:);
    output_mask_disc=mask_disc(x_start:x_start+output_image_size(1)-1,y_start:y_start+output_image_size(2)-1);
    output_mask_cup=mask_cup(x_start:x_start+output_image_size(1)-1,y_start:y_start+output_image_size(2)-1);
end



    