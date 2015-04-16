clear all; clc;

datadir = '~/scratch/nndata/';

net = load(strcat(datadir, 'imagenet-vgg-f.mat'));

im = imread( strcat(datadir,'clef_noise.png'));
im_ = single(im);
im_ = imresize(im_, net.normalization.imageSize(1:2));
im_ = im_ - net.normalization.averageImage;

%figure(1); clf; imagesc(im);

res = vl_simplenn(net, im_);


L = 8;

% get output and threshold
op = res(L+1).x;
maxop = max(abs(op(:)));
minop = min(abs(op(:)));
eps1 = 0.5;                     % controls dropout / regularization
thresh = min(op(:)) + eps1*(maxop-minop);
op(op<thresh) = 0;

% manually backprop the new filter outputs
imf = im_;
num_avg = 10;

for n1=1:num_avg,

    eps2 = 10;            % dz ~ 1/eps2 tests robustness of loss, i.e., generalization error
    toadd = rand(size(op))*eps2;
    %toadd(op < 1e-8) = 0;
    dzdx = op + toadd .* op;
    for i=L:-1:1,
        ll = net.layers{i};
        wtype = ll.type;
        x = res(i).x;
        if strcmp(wtype, 'pool'),
            dzdx = vl_nnpool(x, ll.pool, dzdx, 'stride', ll.stride, 'pad', ll.pad);
        elseif strcmp(wtype, 'normalize'),
            dzdx = vl_nnnormalize(x, ll.param, dzdx);
        elseif strcmp(wtype, 'relu'),
            dzdx = vl_nnrelu(x, dzdx);
        elseif strcmp(wtype, 'conv'),
            [dzdx, dzdf, dzdb] = vl_nnconv(x, ll.filters, ll.biases, dzdx, ...
                'stride', ll.stride, 'pad', ll.pad); 
        end
    end

    imf = imf + (im_ + eps2*dzdx + net.normalization.averageImage);
end

imf = imf/num_avg;
imf = imf + abs(min(imf(:)));
imf = imf/max(imf(:));

figure(1); clf; imagesc(imf);