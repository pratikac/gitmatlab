clear all;

% read image, compute its gradient and norm

I = imread('lena.png');
I = vl_imdown(double(rgb2gray(I)));


% compute local mean via spatial smoothing;

sigma = 5;
muI = vl_imsmooth(I,sigma);

% compute local STD

thresh = 0.001; % note, changing the threshold changes the response

sigmaI = max(thresh,diag((I-muI) * (I-muI)'));

phi = inv(diag(sigmaI))*(I-muI);

% spatial pooling/blurring

sigma2 = 3; % should not be the same as sigma
y = vl_imsmooth(phi,sigma2);

% the above should be the mean of the likelihood function

figure(1); clf; hold off;
imagesc(y);
pause;

% generate a "pulse image" to probe the basis

I = 0*I;
I(100:150,100:150) = 100; % note: changing value changes the response; the filters are low-pass for small values, high-pass for high value,

% visualize the filters, computed hierarchically

eps = 0.01;
for t = 1 : 100,

    muI = vl_imsmooth(I,sigma);

    sigmaI = max(thresh,((I-muI) * (I-muI)') + eps*ones(size(I)));

    phi = inv(sigmaI)*(I-muI);

    y = vl_imsmooth(phi,sigma2);

    figure(1); clf; hold off;
    imagesc(I)
    pause;

    I = y;

end;