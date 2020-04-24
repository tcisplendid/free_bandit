function [y] = compute_round_2(a, b, sigma)
    % compute p(A0<B0)
    p0 = normcdf(0, a - b, sqrt(2).*sigma)
    % compute p(A1B1)
    p1 = condition_prob(a, b, sigma, 0);
    p2 = condition_prob(b, a, sigma, 0);
    p_A1B1 = (1-p0) * p1
    p_B1A1 = p0 * p2
    p_B1B2 = p0 * condition_prob(b, a, sigma, 1)
    p_A1A2 = (1-p0) * condition_prob(a, b, sigma, 1)
    sum = p_B1B2+p_B1A1+p_A1B1+p_A1A2
    y = 2.*p_B1B2+p_B1A1+p_A1B1;
end


function [res] = condition_prob(mu1,mu2,sigma,same_arm)
    less = @(y, x) normcdf(2.*y-x, mu1, sigma).*normpdf(x, mu1, sigma).*normpdf(y, mu2, sigma);
    greater = @(y, x) (1-normcdf(2.*y-x, mu1, sigma)).*normpdf(x, mu1, sigma).*normpdf(y, mu2, sigma);
    A0min = @(x) x;
    if same_arm == 1
        res = integral2(greater, -Inf, Inf, A0min, Inf);
    else
        res = integral2(less, -Inf, Inf, A0min, Inf);
    end
end
