function [B, cost] = lasso(X, A, gamma)

    RELTOL = 1e-4;
    ABSTOL = 1e-4;

    c = size(X, 2);
    r = size(A, 2);

    % initialize all
    L = zeros(size(X));
    rho = 1e-4;
    max_iter = 1000;
    I = speye(r);
    max_rho = 5;
    C = randn(r,c);

    soft_thresh = @(x, th) sign(x).*max(abs(x) -th,0);

    norm2 = @(x) x(:)'*x(:);
    norm1 = @(x) sum(abs(x(:)));

    cost = [];
    for n = 1:max_iter

        % 1. solve for B
        B = (A'*A + rho*I)\(A'*X + rho*C - L);

        % solve for C
        C = soft_thresh(B + L/rho, gamma/rho);

        % update L
        L = L + rho*(B-C);

        rho = min(max_rho, rho*1.1);

        cost(n) = 0.5*norm2(X - A*B) + gamma*norm1(B);

        if n > 1
            dcost = cost(n-1) - cost(n);
            if (dcost/cost(n-1) < RELTOL) || (dcost < ABSTOL)
                break
            end
        end
    end

end