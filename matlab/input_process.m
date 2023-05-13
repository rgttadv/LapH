% https://github.com/HaoZhang1018/SDNet
clc;
clear all;

I_ir=(imread(strcat('imgs_rgb/139.jpg')));
[Y,Cb,Cr]=RGB2YCbCr(I_ir);
imwrite(Y, strcat('imgs_gray/139.png'));