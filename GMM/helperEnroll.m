function speakerGMM = helperEnroll(ubm,afe,normFactors,adsEnroll)
    % Initialization
    numComponents = numel(ubm.ComponentProportion);
    numFeatures = size(ubm.mu,1);
    N = zeros(1,numComponents);
    F = zeros(numFeatures,numComponents);
    S = zeros(numFeatures,numComponents);
    NumFrames = 0;
    
    while hasdata(adsEnroll)
        % Read from the enrollment datastore
        audioData = read(adsEnroll);
        audioData = audioData(:,1);
    
        % 1. Extract the features and apply feature normalization
        [features,numFrames] = helperFeatureExtraction(audioData,afe,normFactors);
        
        % 2. Calculate the a posteriori probability. Use it to determine the
        % sufficient statistics (the count, and the first and second moments)
        [n,f,s] = helperExpectation(features,ubm);
        
        % 3. Update the sufficient statistics
        N = N + n;
        F = F + f;
        S = S + s;
        NumFrames = NumFrames + numFrames;
    end
    % Create the Gaussian mixture model that maximizes the expectation
    speakerGMM = helperMaximization(N,F,S);
    
    % Adapt the UBM to create the speaker model. Use a relevance factor of 16,
    % as proposed in [2]
    relevanceFactor = 16;
    
    % Determine adaption coefficient
    alpha = N ./ (N + relevanceFactor);
    
    % Adapt the means
    speakerGMM.mu = alpha.*speakerGMM.mu + (1-alpha).*ubm.mu;
    
    % Adapt the variances
    speakerGMM.sigma = alpha.*(S./N) + (1-alpha).*(ubm.sigma + ubm.mu.^2) - speakerGMM.mu.^2;
    speakerGMM.sigma = max(speakerGMM.sigma,eps);
    
    % Adapt the weights
    speakerGMM.ComponentProportion = alpha.*(N/sum(N)) + (1-alpha).*ubm.ComponentProportion;
    speakerGMM.ComponentProportion = speakerGMM.ComponentProportion./sum(speakerGMM.ComponentProportion);
end

