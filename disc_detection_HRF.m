clear all
close all
clc
sigma=60;
size_of_erosion=80;
% output_image_size=[500,500];
out_f = 'data_preprocessed_dicom_12';
% rc = 55;
path='D:\Diploma_thesis_segmentation_disc\';


%% HRF
degree = 60;
% mkdir([out_f '\HRF'])
images = dir([path 'HRF\images\*.jpg']);
center_new_HRF=[];
imname={};

for i=1:length(images)
    disp(['HRF: ' num2str(i) '/' num2str(length(images))])
    in=images(i).name(1:end-4);
    
    im=imread([path 'HRF\images\' images(i).name ]);
%     ves=logical(imread([path 'HRF\manual1\' in '.tif']));
    fov=logical(rgb2gray(imread([path 'HRF\mask\' in '_mask.tif'])));
%     va = imread([path 'HRF\clasified\' in '_Eva.png']);
%     va(va==50) = 0;
%     va(va==100) = 1;
%     va(va==150) = 2;
%     va(va==255) = 0; % ???
%     
%     chck_labels = unique(va(:));
%     if length(chck_labels)>3
%         disp(['HRF: ' in 'has incorrect labels:' strjoin(string(chck_labels))])
%     end

%     [I,V,VA,~, fov]=image_adjustment(im,rc,degree,ves,va,0, 'hrf', fov);
%     VA = VA.*uint8(V);
%     V(VA==0) = 0;
%     I = uint16(round(I.*2.^12));

%     [center_new] = Detection_of_disc(I,fov,sigma,size_of_erosion);
    [center_new] = Detection_of_disc(im,fov,sigma,size_of_erosion);
    figure
    imshow(im)
    hold on
    stem(center_new(1),center_new(2),'g','MarkerSize',16,'LineWidth',2)
    pause(1)
    answer= questdlg('Is detected optic disc?','Optic disc detection','Yes','No','Yes');

    switch answer
        case 'Yes'
            center_new_HRF(i,1)=center_new(1);
            center_new_HRF(i,2)=center_new(2);
        case 'No'
            imshow(im)
            [x,y] = ginput(1);
            center_new_HRF(i,1)=x;
            center_new_HRF(i,2)=y;
    end
    close all
    
    ind=strfind(in,'_');
    diagnose=in(ind(1)+1);
    in(ind)=[];
    if diagnose=='h'
        imname{i,1}= [ 'hrf_healthy_'  in  ];
    elseif diagnose=='g'
        imname{i,1}= [ 'hrf_glaucoma_'  in  ];
    elseif diagnose=='d'
        imname{i,1}= [ 'hrf_dr_'  in  ];
    end
    
%     dicomwrite(I(:,:,1),[out_f '\HRF\' imname '_R.dcm'])
%     dicomwrite(I(:,:,2),[out_f '\HRF\' imname '_G.dcm'])
%     dicomwrite(I(:,:,3),[out_f '\HRF\' imname '_B.dcm'])
%     dicomwrite(uint16(V),[out_f '\HRF\' imname '_ves.dcm'])
%     dicomwrite(uint16(fov),[out_f '\HRF\' imname '_fov.dcm'])
%     dicomwrite(uint16(VA),[out_f '\HRF\' imname '_va.dcm'])

end

%% ulozeni
HRF_disc_coordinate=table(imname,center_new_HRF(:,1),center_new_HRF(:,2),'VariableNames',{'name','x-coordinates','y-coordinates'})
writetable(HRF_disc_coordinate,'HRF_disc_coordinate.csv',"Delimiter",",");

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

function [output_crop_image]=Crop_image(image,output_image_size,center_new)
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
end

