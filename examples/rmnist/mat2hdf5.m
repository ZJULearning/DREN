clear;
prefix='examples/rmnist/';
load('data/mnist_rotation_train.amat', '-ascii')

%%
savepath='data/train.h5';
x=mnist_rotation_train(:,1:784);
y=mnist_rotation_train(:,785);
num=size(x,1);
im_mean=mean(x);
im_std=std(x);
x=(x-repmat(im_mean,num,1))./repmat(im_std,num,1);
x=reshape(x,num,1,28,28);
x=permute(x,[4,3,2,1]);
y=permute(y,[2,1]);
fid=fopen('data/train_h5.txt','wt');
fprintf(fid,'%s%s\n',prefix,savepath);
fclose(fid);
%% writing to HDF5
count=size(x,4);
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = x(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = y(1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);

%%
load('data/mnist_rotation_test.amat', '-ascii')

%%
savepath='data/test.h5';
x=mnist_rotation_test(:,1:784);
y=mnist_rotation_test(:,785);
num=size(x,1);
im_mean=mean(x);
im_std=std(x);
x=(x-repmat(im_mean,num,1))./repmat(im_std,num,1);
x=reshape(x,num,1,28,28);
x=permute(x,[4,3,2,1]);
y=permute(y,[2,1]);
fid=fopen('data/test_h5.txt','wt');
fprintf(fid,'%s%s\n',prefix,savepath);
fclose(fid);
%% writing to HDF5
count=size(x,4);
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = x(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = y(1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
