function [N,F,S,L] = helperExpectation(features,gmm)

    post = helperGMMLogLikelihood(features,gmm);
    
    % Sum the likelihood over the frames
    L = helperLogSumExp(post);
    
    % Compute the sufficient statistics
    gamma = exp(post-L)';
    
    N = sum(gamma,1);
    F = features * gamma;
    S = (features.*features) * gamma;
    L = sum(L);
end
