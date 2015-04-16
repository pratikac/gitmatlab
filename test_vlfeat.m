I = vl_impattern('roofs1');
ts = tic;
I = single(vl_imdown(rgb2gray(I)));
%I = imresize(I, 0.1);


bin_size = 8;
magnif = 3;
Is = vl_imsmooth(I, sqrt((bin_size/magnif)^2 - .25));

[f,d] = vl_dsift(Is, 'size', bin_size, 'fast');
f(3,:) = bin_size/magnif;
f(4,:) = 0;
te = toc(ts)