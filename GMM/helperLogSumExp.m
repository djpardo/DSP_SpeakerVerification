function y = helperLogSumExp(x)
    % Calculate the log-sum-exponent while avoiding overflow
    a = max(x,[],1);
    y = a + sum(exp(bsxfun(@minus,x,a)),1);
end