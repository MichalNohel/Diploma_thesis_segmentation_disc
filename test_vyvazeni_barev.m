path_to_data='D:\Diploma_thesis_segmentation_disc\Data_640_640\';
images_file = dir([path_to_data 'Train\Images_crop\*.png']);
num_of_img=length(images_file);
for i=1:num_of_img
    %expert 1
    image=imread([images_file(i).folder '\' images_file(i).name ]);
    figure
    subplot 131
    imshow(image)    
    subplot 132
    imshow(histeq(image))  
    illuminant=illumwhite(image);
    B = chromadapt(image,illuminant);
    subplot 133
    imshow(B)
    pause(2)
    close all

end



