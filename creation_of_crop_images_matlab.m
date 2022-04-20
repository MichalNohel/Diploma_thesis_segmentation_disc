clear all
close all
clc
path_to_data='D:\Diploma_thesis_segmentation_disc';
output_image_size=[500,500];
sigma=60;
size_of_erosion=80;
percentage_number_test=0.2;
path_to_crop_image=[path_to_data '/Data_500_500/'];

%% Dristi-GS - Expert 1
pom=52; % split to test and train dataset
images_file = dir([path_to_data '\Drishti-GS\Images\*.png']);
disc_file = dir([path_to_data '\Drishti-GS\Disc\expert1\*.png']);
cup_file = dir([path_to_data '\Drishti-GS\Cup\expert1\*.png']);
fov_file = dir([path_to_data '\Drishti-GS\Fov\*.png']);

coordinates_dristi_GS=load('C:\Users\nohel\Desktop\Databaze_final\Drishti-GS\coordinates_dristi_GS.mat');
coordinates=coordinates_dristi_GS.coordinates_dristi_GS;
num_of_img=length(images_file);
pom=52; % split to test and train dataset

creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)

%% REFUGE_train

images_file = dir([path_to_data '\REFUGE\Images\Train\*.png']);
disc_file = dir([path_to_data '\REFUGE\Disc\Train\*.png']);
cup_file = dir([path_to_data '\REFUGE\Cup\Train\*.png']);
fov_file = dir([path_to_data '\REFUGE\Fov\Train\*.png']);

coordinates_REFUGE_Train=load('C:\Users\nohel\Desktop\Databaze_final\REFUGE\coordinates_REFUGE_Train.mat');
coordinates=coordinates_REFUGE_Train.coordinates_REFUGE_Train;
num_of_img=length(images_file);
pom=0; % split to test and train dataset
%%
creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)

%% REFUGE_Validation

images_file = dir([path_to_data '\REFUGE\Images\Validation\*.png']);
disc_file = dir([path_to_data '\REFUGE\Disc\Validation\*.png']);
cup_file = dir([path_to_data '\REFUGE\Cup\Validation\*.png']);
fov_file = dir([path_to_data '\REFUGE\Fov\Validation\*.png']);


coordinates_REFUGE_Validation=load('C:\Users\nohel\Desktop\Databaze_final\REFUGE\coordinates_REFUGE_Validation.mat');
coordinates=coordinates_REFUGE_Validation.coordinates_REFUGE_Validation;
num_of_img=length(images_file);
pom=0; % split to test and train dataset
%%
creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)

%% REFUGE_Test

images_file = dir([path_to_data '\REFUGE\Images\Test\*.png']);
disc_file = dir([path_to_data '\REFUGE\Disc\Test\*.png']);
cup_file = dir([path_to_data '\REFUGE\Cup\Test\*.png']);
fov_file = dir([path_to_data '\REFUGE\Fov\Test\*.png']);


coordinates_REFUGE_Test=load('C:\Users\nohel\Desktop\Databaze_final\REFUGE\coordinates_REFUGE_Test.mat');
coordinates=coordinates_REFUGE_Test.coordinates_REFUGE_Test;
num_of_img=length(images_file);
pom=401; % split to test and train dataset
%%
creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)

%% Riga - Bin Rushed

images_file = dir([path_to_data '\RIGA\Images\BinRushed\*.png']);
disc_file = dir([path_to_data '\RIGA\Disc\BinRushed\expert1\*.png']);
cup_file = dir([path_to_data '\RIGA\Cup\BinRushed\expert1\*.png']);
fov_file = dir([path_to_data '\RIGA\Fov\BinRushed\*.png']);

coordinates_RIGA_BinRushed=load('C:\Users\nohel\Desktop\Databaze_final\RIGA\coordinates_RIGA_BinRushed.mat');
coordinates=coordinates_RIGA_BinRushed.coordinates_RIGA_BinRushed;
num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
%%
creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)

%% Riga - Magrabia

images_file = dir([path_to_data '\RIGA\Images\Magrabia\*.png']);
disc_file = dir([path_to_data '\RIGA\Disc\Magrabia\expert1\*.png']);
cup_file = dir([path_to_data '\RIGA\Cup\Magrabia\expert1\*.png']);
fov_file = dir([path_to_data '\RIGA\Fov\Magrabia\*.png']);

coordinates_RIGA_Magrabia=load('C:\Users\nohel\Desktop\Databaze_final\RIGA\coordinates_RIGA_Magrabia.mat');
coordinates=coordinates_RIGA_Magrabia.coordinates_RIGA_Magrabia;
num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
%%
creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)

%% Riga - MESSIDOS

images_file = dir([path_to_data '\RIGA\Images\MESSIDOS\*.png']);
disc_file = dir([path_to_data '\RIGA\Disc\MESSIDOS\expert1\*.png']);
cup_file = dir([path_to_data '\RIGA\Cup\MESSIDOS\expert1\*.png']);
fov_file = dir([path_to_data '\RIGA\Fov\MESSIDOS\*.png']);

coordinates_RIGA_MESSIDOR=load('C:\Users\nohel\Desktop\Databaze_final\RIGA\coordinates_RIGA_MESSIDOR.mat');
coordinates=coordinates_RIGA_MESSIDOR.coordinates_RIGA_MESSIDOR;
num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
%%
creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)

%% RIM-ONE - Glaucoma

images_file = dir([path_to_data '\RIM-ONE\Images\Glaucoma\*.png']);
disc_file = dir([path_to_data '\RIM-ONE\Disc\Glaucoma\*.png']);
cup_file = dir([path_to_data '\RIM-ONE\Cup\Glaucoma\*.png']);
fov_file = dir([path_to_data '\RIM-ONE\Fov\Glaucoma\*.png']);

coordinates_RIM_ONE_Glaucoma=load('C:\Users\nohel\Desktop\Databaze_final\RIM-ONE\coordinates_RIM_ONE_Glaucoma.mat');
coordinates=coordinates_RIM_ONE_Glaucoma.coordinates_RIM_ONE_Glaucoma;
num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
%%
creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)

%% RIM-ONE - Healthy

images_file = dir([path_to_data '\RIM-ONE\Images\Healthy\*.png']);
disc_file = dir([path_to_data '\RIM-ONE\Disc\Healthy\*.png']);
cup_file = dir([path_to_data '\RIM-ONE\Cup\Healthy\*.png']);
fov_file = dir([path_to_data '\RIM-ONE\Fov\Healthy\*.png']);

coordinates_RIM_ONE_Healthy=load('C:\Users\nohel\Desktop\Databaze_final\RIM-ONE\coordinates_RIM_ONE_Healthy.mat');
coordinates=coordinates_RIM_ONE_Healthy.coordinates_RIM_ONE_Healthy;
num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
%%
creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)

%% UoA_DR - Healthy

images_file = dir([path_to_data '\UoA_DR\Images\Healthy\*.png']);
disc_file = dir([path_to_data '\UoA_DR\Disc\Healthy\*.png']);
cup_file = dir([path_to_data '\UoA_DR\Cup\Healthy\*.png']);
fov_file = dir([path_to_data '\UoA_DR\Fov\Healthy\*.png']);

coordinates_UoA_Healthy=load('C:\Users\nohel\Desktop\Databaze_final\UoA_DR\coordinates_UoA_Healthy.mat');
coordinates=coordinates_UoA_Healthy.coordinates_UoA_Healthy;
num_of_img=length(images_file);
pom=round(num_of_img*percentage_number_test); % split to test and train dataset
%%
creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)

%%
% %% UoA_DR - NDPR
% 
% images_file = dir([path_to_data '\UoA_DR\Images\NDPR\*.png']);
% disc_file = dir([path_to_data '\UoA_DR\Disc\NDPR\*.png']);
% cup_file = dir([path_to_data '\UoA_DR\Cup\NDPR\*.png']);
% fov_file = dir([path_to_data '\UoA_DR\Fov\NDPR\*.png']);
% 
% coordinates_UoA_NDPR=load('C:\Users\nohel\Desktop\Databaze_final\UoA_DR\coordinates_UoA_NDPR.mat');
% coordinates=coordinates_UoA_NDPR.coordinates_UoA_NDPR;
% num_of_img=length(images_file);
% pom=round(num_of_img*percentage_number_train); % split to test and train dataset
% %%
% creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)
% 
% %% UoA_DR - PDR
% 
% images_file = dir([path_to_data '\UoA_DR\Images\PDR\*.png']);
% disc_file = dir([path_to_data '\UoA_DR\Disc\PDR\*.png']);
% cup_file = dir([path_to_data '\UoA_DR\Cup\PDR\*.png']);
% fov_file = dir([path_to_data '\UoA_DR\Fov\PDR\*.png']);
% 
% coordinates_UoA_PDR=load('C:\Users\nohel\Desktop\Databaze_final\UoA_DR\coordinates_UoA_PDR.mat');
% coordinates=coordinates_UoA_PDR.coordinates_UoA_PDR;
% num_of_img=length(images_file);
% pom=round(num_of_img*percentage_number_train); % split to test and train dataset
% %%
% creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)

%% Detection of centres in Test datasets
clear all
close all
clc
sigma=60;
size_of_erosion=80;
test_images_file = dir('D:\Diploma_thesis_segmentation_disc\Data_500_500\Test\Images\*.png');
test_fov_file = dir('D:\Diploma_thesis_segmentation_disc\Data_500_500\Test\Fov\*.png');
test_dics_file = dir('D:\Diploma_thesis_segmentation_disc\Data_500_500\Test\Disc\*.png');
num_of_img=length(test_images_file);
Disc_centres_test=[];
Accuracy_of_detec=[];
%%
for i=1:num_of_img
    image=imread([test_images_file(i).folder '\' test_images_file(i).name ]); 
    fov=imread([test_fov_file(i).folder '\' test_fov_file(i).name ]);
    mask_disc=imread([test_dics_file(i).folder '\' test_dics_file(i).name ]); 
    [center_new] = Detection_of_disc(image,fov,sigma,size_of_erosion);
    Disc_centres_test(i,1)=center_new(1);
    Disc_centres_test(i,2)=center_new(2);
    if mask_disc(center_new(2),center_new(1))==1
        Accuracy_of_detec(i)=1;
    else
        Accuracy_of_detec(i)=0;
    end
end
accuracy=sum(Accuracy_of_detec)/length(Accuracy_of_detec)
%% save of test discs centers
% Disc_centres_test=Disc_centres_test-1
% save('Disc_centres_test.mat','Disc_centres_test')

%% Functions
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
        y_start=size_in_img(2)-output_image_size(2);
    else
        y_start=center_new(1)-y_half;
    end

    output_crop_image=image(x_start:x_start+output_image_size(1)-1,y_start:y_start+output_image_size(2)-1,:);
    output_mask_disc=mask_disc(x_start:x_start+output_image_size(1)-1,y_start:y_start+output_image_size(2)-1);
    output_mask_cup=mask_cup(x_start:x_start+output_image_size(1)-1,y_start:y_start+output_image_size(2)-1);
end

function []= creation_of_crop_images(output_image_size,images_file,disc_file,cup_file,fov_file,sigma,size_of_erosion,coordinates,path_to_crop_image,pom)
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
            imwrite(fov,[path_to_crop_image 'Test\Fov\' fov_file(i).name])
        end
    end
end

    