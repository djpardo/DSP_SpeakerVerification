function [ubm] = helperTrainUBM(adsTrainUBM, afe, numPar, normFactors)
    % Initialize
    numComponents = 32;
    numFeatures = 13;
    alpha = ones(1,numComponents)/numComponents;
    mu = randn(numFeatures, numComponents);
    sigma = rand(numFeatures, numComponents);
    ubm = struct(ComponentProportion=alpha, mu=mu, sigma=sigma);
    
    % Define stopping criteria
    maxIter = 20;
    targetLogLikelihood = 0;
    tol = 0.5;
    pastL = -inf; % initialization of previous log-likelihood
    
    tic
    for iter = 1:maxIter
        
        % EXPECTATION
        N = zeros(1,numComponents);
        F = zeros(numFeatures,numComponents);
        S = zeros(numFeatures,numComponents);
        L = 0;
        parfor ii = 1:numPar
            adsPart = partition(adsTrainUBM, numPar, ii);
            while hasdata(adsPart)
                audioData = read(adsPart);
                audioData = audioData(:,1);
                
                % Extract features
                [features, numFeatures] = helperFeatureExtraction(audioData, afe, normFactors);
     
                % Compute a posteriori log-likelihood
                logLikelihood = helperGMMLogLikelihood(features, ubm);
    
                % Compute a posteriori normalized probability
                logLikelihoodSum = helperLogSumExp(logLikelihood);
                gamma = exp(logLikelihood - logLikelihoodSum)';
                
                % Compute Baum-Welch statistics
                n = sum(gamma,1);
                f = features * gamma;
                s = (features.*features) * gamma;
                
                % Update the sufficient statistics over utterances
                N = N + n;
                F = F + f;
                S = S + s;
                
                % Update the log-likelihood
                L = L + sum(logLikelihoodSum);
            end
        end
        
        % Print current log-likelihood and stop if it meets criteria.
        L = L/numel(adsTrainUBM.Files);
        disp("Iteration " + iter + ", Log-likelihood = " + round(L,3))
        if L > targetLogLikelihood || abs(pastL - L) < tol
            break
        else
            pastL = L;
        end
        
        % MAXIMIZATION
        N = max(N,eps);
        ubm.ComponentProportion = max(N/sum(N),eps);
        ubm.ComponentProportion = ubm.ComponentProportion/sum(ubm.ComponentProportion);
        ubm.mu = bsxfun(@rdivide,F,N);
        ubm.sigma = max(bsxfun(@rdivide,S,N) - ubm.mu.^2,eps);
    end
end

