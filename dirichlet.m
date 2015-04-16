function r = dirichlet(a, n)
% a = [a_1, \ldots, a_p]
% n = number of samples
% returns n samples drawn from Dir(a_1, \ldots, a_p)
% size(r) = (n,p)
    p = length(a);
    r = randgamma(repmat(a,n,1));
    r = r./repmat(sum(r,2),1,p);
end
