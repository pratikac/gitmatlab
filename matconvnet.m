%net = load('imagenet-vgg-verydeep-16.mat');
net = load('~/scratch/nndata/imagenet-vgg-f.mat');

% process
im = imread('peppers.png');
im_ = single(im);
im_ = imresize(im_, net.normalization.imageSize(1:2));
im_ = im_ - net.normalization.averageImage;

% run CNN
tstart = tic;
res = vl_simplenn(net, im_);
dt = toc(tstart)

scores = squeeze(gather(res(end).x));
[best_score, best] = max(scores);
figure(1); clf; imagesc(im);
title(sprintf('%s (%d), score: %.3f', ...
    net.classes.description{best}, best, best_score));
