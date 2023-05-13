% https://github.com/HaoZhang1018/SDNet
clc;
clear all;
I_result=double(imread(strcat('fused/139.png')));  
I_init_vi=double(imread(strcat('imgs_rgb/139.jpg')));

[Y,Cb,Cr]=RGB2YCbCr(I_init_vi);

I_final_YCbCr=cat(3,I_result(:,:,1),Cb,Cr);

I_final_RGB=YCbCr2RGB(I_final_YCbCr);

imwrite(uint8(I_final_RGB), strcat('139.png')); 


