function logprob = dirichlet_logprob(a, X)
% a = parameters of Dirichlet
% X = (n, length(a)) is vector of samples
% return logprob (n, 1)

    p = length(a);
    logprob = log(X) * (a-1)';
    logprob = logprob+ gammaln(sum(a)) - sum(gammaln(a));
end
