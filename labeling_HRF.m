clear all
close all
clc
path_to_data='D:\Diploma_thesis_segmentation_disc\HRF\';
images_file = dir([path_to_data '\images\*.jpg']);
path_to_disc=[path_to_data 'Disc\'];
path_to_cup=[path_to_data 'Cup\'];
num_of_img=length(images_file);

%%
i=45;
disp(images_file(i).name)
image=imread([images_file(i).folder '\' images_file(i).name ]);
imshow(image)
%% disc
imageSegmenter(image)
%%
imwrite(MaskDisk45,[path_to_disc images_file(i).name(1:end-4) '_disc.png'])

%% cup
imageSegmenter(image)
%%
imwrite(MaskCup45,[path_to_cup images_file(i).name(1:end-4) '_cup.png'])
%%
clear all
close all
clc
path_to_data='D:\Diploma_thesis_segmentation_disc\HRF\';
images_file = dir([path_to_data '\images\*.jpg']);
disc_file=dir([path_to_data 'Disc\*.png']);
cup_file=dir([path_to_data 'Cup\*.png']);
i=1;
disp(images_file(i).name)
image=imread([images_file(i).folder '\' images_file(i).name ]);
disc=imread([disc_file(i).folder '\' disc_file(i).name ]);
cup=imread([cup_file(i).folder '\' cup_file(i).name ]);
%%
imshow(image)
[size_row,size_colomn]=size(image);
[y,x] = ginput(1);
x=round(x);
y=round(y)
delka=600;
if ((x+delka)>size_row)
    start_x=size_row-delka;
else
    start_x=x;
end
if ((y+delka)>size_colomn)
    start_y=size_colomn-delka;
else
    start_y=y;
end

%% Disk
DISC= repmat(disc,[1 1 3]);
imfuse5(image(start_x:start_x+delka,start_y:start_y+delka,:),DISC(start_x:start_x+delka,start_y:start_y+delka,:))

%% Cup
CUP= repmat(cup,[1 1 3]);
imfuse5(image(start_x:start_x+delka,start_y:start_y+delka,:),CUP(start_x:start_x+delka,start_y:start_y+delka,:))






