function [x,im_mean,im_std]= gcn( x )
%GCN Summary of this function goes here
%   Detailed explanation goes here
[w,l,c,n]=size(x);
x=reshape(x,w*l*c,n);
x=single(x);
im_mean=mean(x,2);
im_std=std(x,0,2);
x=(x-repmat(im_mean,1,n))./repmat(im_std,1,n);
x=reshape(x,w,l,c,n);
end

