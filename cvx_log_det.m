rand('state',0);
randn('state',0);
n = 10;
N = 100;
Strue = sprandsym(n,0.5,0.01,1);
R = inv(full(Strue));
y_sample = sqrtm(R)*randn(n,N);
Y = cov(y_sample');
alpha = 5;

% Computing sparse estimate of R^{-1}
cvx_begin sdp
    variable S(n,n) symmetric
    maximize log_det(S) - trace(S*Y)
    sum(sum(abs(S))) <= alpha
    S >= 0
cvx_end
R_hat = inv(S);

S(find(S<1e-4)) = 0;
figure;
subplot(121);
spy(Strue);
title('Inverse of true covariance matrix')
subplot(122);
spy(S)
title('Inverse of estimated covariance matrix')