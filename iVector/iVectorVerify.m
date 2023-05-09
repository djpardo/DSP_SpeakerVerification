% Manage dataset
ads = audioDatastore('/Users/aliciahyland/MATLAB/Projects/DSP_SpeakerVerification/Data/Password/');
[~, filenames] = fileparts(ads.Files);
ads.Labels = extractBetween(filenames, "_", "_");
speakers = unique(ads.Labels);
numSpeakers = numel(speakers);
[adsTrain, adsEnroll, adsTest] = splitEachLabel(ads, 0.4, 0.4);

countEachLabel(adsTrain)
countEachLabel(adsEnroll)
countEachLabel(adsTest)

% Read audio file from training dataset and plot
[audio,audioInfo] = read(adsTrain);
fs = audioInfo.SampleRate;
t = (0:size(audio,1)-1)/fs;
sound(audio,fs)

plot(t,audio)
xlabel("Time (s)")
ylabel("Amplitude")
axis([0 t(end) -1 1])
title("Sample Utterance from Training Set")

reset(adsTrain)

% DEVELOPMENT - speed up, reduces performance
speedUpExample = false;
if speedUpExample
    adsTrain = splitEachLabel(adsTrain,30);
end

% Extract 20 MFCCs, 20 delta MFCCs, and 20 delta-delta MFCCs
numCoeffs = 20;
deltaWindowLength = 9;

windowDuration = 0.025;
hopDuration = 0.01;

windowSamples = round(windowDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = windowSamples - hopSamples;

afe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    Window=hann(windowSamples,"periodic"), ...
    OverlapLength=overlapSamples, ...
    ...
    mfcc=true, ...
    mfccDelta=true, ...
    mfccDeltaDelta=true);
setExtractorParameters(afe,"mfcc",DeltaWindowLength=deltaWindowLength,NumCoeffs=numCoeffs)

% Extract the features from our training audio sample
features = extract(afe,audio);
[numHops,numFeatures] = size(features)

% Use parallel computing on multi-core systems
if ~isempty(ver("parallel")) && ~speedUpExample
    pool = gcp;
    numPar = numpartitions(adsTrain,pool);
else
    numPar = 1;
end

% Extract all features from training dataset
featuresAll = {};
tic
parfor ii = 1:numPar
    adsPart = partition(adsTrain,numPar,ii);
    featuresPart = cell(0,numel(adsPart.Files));
    for iii = 1:numel(adsPart.Files)
        audioData = read(adsPart);
        audioData = audioData(:,1);
        featuresPart{iii} = helperFeatureExtraction(audioData,afe,[]);
    end
    featuresAll = [featuresAll,featuresPart];
end
allFeatures = cat(2,featuresAll{:});
disp("Feature extraction from training set complete (" + toc + " seconds).")

% Calculate mean and standard deviation of each feature
normFactors.Mean = mean(allFeatures,2,"omitnan");
normFactors.STD = std(allFeatures,[],2,"omitnan");

% Initialize Gaussian mixture model for background model
numFeatures = 60;
numComponents = 64;
if speedUpExample
    numComponents = 32;
end
alpha = ones(1,numComponents) / numComponents;
mu = randn(numFeatures, numComponents);
vari = rand(numFeatures, numComponents) + eps;
ubm = struct(ComponentProportion=alpha, mu=mu, sigma=vari);

% Train model with expectation-maximization algorithm
maxIter = 10;
if speedUpExample
    maxIter = 2;
end
tic
for iter = 1:maxIter
    tic
    % EXPECTATION
    N = zeros(1,numComponents);
    F = zeros(numFeatures,numComponents);
    S = zeros(numFeatures,numComponents);
    L = 0;
        adsPart = partition(adsTrain,1,1);
        while hasdata(adsPart)
            audioData = read(adsPart);
            audioData = audioData(:,1);
            
            % Extract features
            Y = helperFeatureExtraction(audioData, afe, normFactors);
 
            % Compute a posteriori log-liklihood
            logLikelihood = helperGMMLogLikelihood(Y, ubm);

            % Compute a posteriori normalized probability
            amax = max(logLikelihood,[],1);
            logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
            gamma = exp(logLikelihood - logLikelihoodSum)';
            
            % Compute Baum-Welch statistics
            n = sum(gamma, 1);
            f = Y * gamma;
            s = (Y.*Y) * gamma;
            
            % Update the sufficient statistics over utterances
            N = N + n;
            F = F + f;
            S = S + s;
            
            % Update the log-likelihood
            L = L + sum(logLikelihoodSum);
        end
    
    % Print current log-likelihood
    disp("Training UBM: " + iter + "/" + maxIter + " complete (" + round(toc) + " seconds), Log-likelihood = " + round(L))
    
    % MAXIMIZATION
    N = max(N,eps);
    ubm.ComponentProportion = max(N/sum(N),eps);
    ubm.ComponentProportion = ubm.ComponentProportion/sum(ubm.ComponentProportion);
    ubm.mu = F./N;
    ubm.sigma = max(S./N - ubm.mu.^2,eps);
end

% Calculate zeroth and first-order Baum-Welsch statistics
numSpeakers = numel(adsTrain.Files);
Nc = {};
Fc = {};

tic
parfor ii = 1:numPar
    adsPart = partition(adsTrain,numPar,ii);
    numFiles = numel(adsPart.Files);
    
    Npart = cell(1,numFiles);
    Fpart = cell(1,numFiles);
    for jj = 1:numFiles
        audioData = read(adsPart);
        audioData = audioData(:,1);
        
        % Extract features
        Y = helperFeatureExtraction(audioData,afe,normFactors);
        
        % Compute a posteriori log-likelihood
        logLikelihood = helperGMMLogLikelihood(Y, ubm);
        
        % Compute a posteriori normalized probability
        amax = max(logLikelihood,[],1);
        logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
        gamma = exp(logLikelihood - logLikelihoodSum)';
        
        % Compute Baum-Welch statistics
        n = sum(gamma,1);
        f = Y * gamma;
        
        Npart{jj} = reshape(n,1,1,numComponents);
        Fpart{jj} = reshape(f,numFeatures,1,numComponents);
    end
    Nc = [Nc,Npart];
    Fc = [Fc,Fpart];
end
disp("Baum-Welch statistics completed (" + toc + " seconds).")

N = Nc;
F = Fc;
muc = reshape(ubm.mu,numFeatures,1,[]);
for s = 1:numSpeakers
    N{s} = repelem(reshape(Nc{s},1,[]),numFeatures);
    F{s} = reshape(Fc{s} - Nc{s}.*muc,[],1);
end

Sigma = ubm.sigma(:);

% Specify dimension of total variability space
numTdim = 32;
if speedUpExample
    numTdim = 16;
end

% Initialization
T = randn(numel(ubm.sigma), numTdim);
T = T/norm(T);

I = eye(numTdim);

Ey = cell(numSpeakers,1);
Eyy = cell(numSpeakers,1);
Linv = cell(numSpeakers,1);

% Number of iterations for training
numIterations = 5;

% Training loop
for iterIdx = 1:numIterations
    tic
    
    % 1. Calculate the posterior distribution of the hidden variable
    TtimesInverseSSdiag = (T./Sigma)';
    parfor s = 1:numSpeakers
        L = (I + TtimesInverseSSdiag.*N{s}*T);
        Linv{s} = pinv(L);
        Ey{s} = Linv{s}*TtimesInverseSSdiag*F{s};
        Eyy{s} = Linv{s} + Ey{s}*Ey{s}';
    end
    
    % 2. Accumlate statistics across the speakers
    Eymat = cat(2,Ey{:});
    FFmat = cat(2,F{:});
    Kt = FFmat*Eymat';
    K = mat2cell(Kt',numTdim,repelem(numFeatures,numComponents));
    
    newT = cell(numComponents,1);
    for c = 1:numComponents
        AcLocal = zeros(numTdim);
        for s = 1:numSpeakers
            AcLocal = AcLocal + Nc{s}(:,:,c)*Eyy{s};
        end
        
    % 3. Update the Total Variability Space
        newT{c} = (pinv(AcLocal)*K{c})';
    end
    T = cat(1,newT{:});

    disp("Training Total Variability Space: " + iterIdx + "/" + numIterations + " complete (" + round(toc,2) + " seconds).")
end

% i-Vector extraction
speakers = unique(adsTrain.Labels);
numSpeakers = numel(speakers);
ivectorPerSpeaker = cell(numSpeakers,1);
TS = T./Sigma;
TSi = TS';
ubmMu = ubm.mu;
tic
for speakerIdx = 1:numSpeakers
    
    % Subset the datastore to the speaker you are adapting.
    adsPart = subset(adsTrain, ismember(adsTrain.Labels, speakers(speakerIdx)));
    numFiles = numel(adsPart.Files);
    
    ivectorPerFile = zeros(numTdim,numFiles);
    for fileIdx = 1:numFiles
        audioData = read(adsPart);
        
        % Extract features
        Y = helperFeatureExtraction(audioData,afe,normFactors);
        
        % Compute a posteriori log-likelihood
        logLikelihood = helperGMMLogLikelihood(Y,ubm);
        
        % Compute a posteriori normalized probability
        amax = max(logLikelihood,[],1);
        logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
        gamma = exp(logLikelihood - logLikelihoodSum)';
        
        % Compute Baum-Welch statistics
        n = sum(gamma,1);
        f = Y * gamma - n.*(ubmMu);

        ivectorPerFile(:,fileIdx) = pinv(I + (TS.*repelem(n(:),numFeatures))' * T) * TSi * f(:);
    end
    ivectorPerSpeaker{speakerIdx} = ivectorPerFile;
end
disp("I-vectors extracted from training set (" + toc + " seconds).")

% Create a matrix of the training vectors and a map indicating which i-vector corresponds to which speaker. 
% Initialize the projection matrix as an identity matrix.

w = ivectorPerSpeaker;
utterancePerSpeaker = cellfun(@(x)size(x,2),w);

ivectorsTrain = cat(2,w{:});
projectionMatrix = eye(size(w{1},1));

performLDA = true;
if performLDA
    tic
    numEigenvectors = 16;

    Sw = zeros(size(projectionMatrix,1));
    Sb = zeros(size(projectionMatrix,1));
    wbar = mean(cat(2,w{:}),2);
    for ii = 1:numel(w)
        ws = w{ii};
        wsbar = mean(ws,2);
        Sb = Sb + (wsbar - wbar)*(wsbar - wbar)';
        Sw = Sw + cov(ws',1);
    end
    
    [A,~] = eigs(Sb,Sw,numEigenvectors);
    A = (A./vecnorm(A))';

    ivectorsTrain = A * ivectorsTrain;
    
    w = mat2cell(ivectorsTrain,size(ivectorsTrain,1),utterancePerSpeaker);
    
    projectionMatrix = A * projectionMatrix;
    
    disp("LDA projection matrix calculated (" + round(toc,2) + " seconds).")
end

performWCCN = true;
if performWCCN
    tic
    alpha = 0.9;
    
    W = zeros(size(projectionMatrix,1));
    for ii = 1:numel(w)
        W = W + cov(w{ii}',1);
    end
    W = W/numel(w);
    
    W = (1 - alpha)*W + alpha*eye(size(W,1));
    
    B = chol(pinv(W),"lower");
    
    projectionMatrix = B * projectionMatrix;
    
    disp("WCCN projection matrix calculated (" + round(toc,4) + " seconds).")
end

% Train G-PLDA model

ivectors = cellfun(@(x)projectionMatrix*x,ivectorPerSpeaker,UniformOutput=false);

numEigenVoices = 16;

K = numel(ivectors);
D = size(ivectors{1},1);
utterancePerSpeaker = cellfun(@(x)size(x,2),ivectors);

ivectorsMatrix = cat(2,ivectors{:});
N = size(ivectorsMatrix,2);
mu = mean(ivectorsMatrix,2);

ivectorsMatrix = ivectorsMatrix - mu;

whiteningType = 'ZCA';

if strcmpi(whiteningType,"ZCA")
    S = cov(ivectorsMatrix');
    [~,sD,sV] = svd(S);
    W = diag(1./(sqrt(diag(sD)) + eps))*sV';
    ivectorsMatrix = W * ivectorsMatrix;
elseif strcmpi(whiteningType,"PCA")
    S = cov(ivectorsMatrix');
    [sV,sD] = eig(S);
    W = diag(1./(sqrt(diag(sD)) + eps))*sV';
    ivectorsMatrix = W * ivectorsMatrix;
else
    W = eye(size(ivectorsMatrix,1));
end

ivectorsMatrix = ivectorsMatrix./vecnorm(ivectorsMatrix);

S = ivectorsMatrix*ivectorsMatrix';

ivectors = mat2cell(ivectorsMatrix,D,utterancePerSpeaker);

% Sort people according to number of samples

uniqueLengths = unique(utterancePerSpeaker);
numUniqueLengths = numel(uniqueLengths);

speakerIdx = 1;
f = zeros(D,K);
for uniqueLengthIdx = 1:numUniqueLengths
    idx = find(utterancePerSpeaker==uniqueLengths(uniqueLengthIdx));
    temp = {};
    for speakerIdxWithinUniqueLength = 1:numel(idx)
        rho = ivectors(idx(speakerIdxWithinUniqueLength));
        temp = [temp;rho]; %#ok<AGROW>

        f(:,speakerIdx) = sum(rho{:},2);
        speakerIdx = speakerIdx+1;
    end
    ivectorsSorted{uniqueLengthIdx} = temp; %#ok<SAGROW> 
end

V = randn(D,numEigenVoices);
Lambda = pinv(S/N);

numIter = 5;
minimumDivergence = true;

for iter = 1:numIter
    % EXPECTATION
    gamma = zeros(numEigenVoices,numEigenVoices);
    EyTotal = zeros(numEigenVoices,K);
    R = zeros(numEigenVoices,numEigenVoices);
    
    idx = 1;
    for lengthIndex = 1:numUniqueLengths
        ivectorLength = uniqueLengths(lengthIndex);
        
        % Isolate i-vectors of the same given length
        iv = ivectorsSorted{lengthIndex};
        
        % Calculate M
        M = pinv(ivectorLength*(V'*(Lambda*V)) + eye(numEigenVoices)); % Equation (A.7) in [13]
        
        % Loop over each speaker for the current i-vector length
        for speakerIndex = 1:numel(iv)
            
            % First moment of latent variable for V
            Ey = M*V'*Lambda*f(:,idx); % Equation (A.8) in [13]
            
            % Calculate second moment.
            Eyy = Ey * Ey';
            
            % Update Ryy 
            R = R + ivectorLength*(M + Eyy); % Equation (A.13) in [13]
            
            % Append EyTotal
            EyTotal(:,idx) = Ey;
            idx = idx + 1;
            
            % If using minimum divergence, update gamma.
            if minimumDivergence
                gamma = gamma + (M + Eyy); % Equation (A.18) in [13]
            end
        end
    end
    
    % Calculate T
    TT = EyTotal*f'; % Equation (A.12) in [13]
    
    % MAXIMIZATION
    V = TT'*pinv(R); % Equation (A.16) in [13]
    Lambda = pinv((S - V*TT)/N); % Equation (A.17) in [13]

    % MINIMUM DIVERGENCE
    if minimumDivergence
        gamma = gamma/K; % Equation (A.18) in [13]
        V = V*chol(gamma,'lower');% Equation (A.22) in [13]
    end
end

% Calculate score based on log-likelihood ratio

speakerIdx = 2;
utteranceIdx = 1;
w1 = ivectors{speakerIdx}(:,utteranceIdx);

speakerIdx = 1;
utteranceIdx = 2;
wt = ivectors{speakerIdx}(:,utteranceIdx);

VVt = V*V';
SigmaPlusVVt = pinv(Lambda) + VVt;

term1 = pinv([SigmaPlusVVt VVt;VVt SigmaPlusVVt]);
term2 = pinv(SigmaPlusVVt);

w1wt = [w1;wt];
score = w1wt'*term1*w1wt - w1'*term2*w1 - wt'*term2*wt

gpldaModel = struct(mu=mu, ...
                    WhiteningMatrix=W, ...
                    EigenVoices=V, ...
                    Sigma=pinv(Lambda));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Enroll new speakers who were not in the training set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

speakers = unique(adsEnroll.Labels);
numSpeakers = numel(speakers);
enrolledSpeakersByIdx = cell(numSpeakers, 1);

tic
parfor speakerIdx = 1:numSpeakers
    % Subset the datastore to the speaker you are adapting.
    adsPart = subset(adsEnroll, ismember(adsEnroll.Labels, speakers(speakerIdx)));
    numFiles = numel(adsPart.Files);
    
    ivectorMat = zeros(size(projectionMatrix,1),numFiles);
    for fileIdx = 1:numFiles
        audioData = read(adsPart);
        
        % Extract features
        Y = helperFeatureExtraction(audioData,afe,normFactors);
        
        % Compute a posteriori log-likelihood
        logLikelihood = helperGMMLogLikelihood(Y,ubm);
        
        % Compute a posteriori normalized probability
        amax = max(logLikelihood,[],1);
        logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
        gamma = exp(logLikelihood - logLikelihoodSum)';
        
        % Compute Baum-Welch statistics
        n = sum(gamma,1);
        f = Y * gamma - n.*(ubmMu);
        
        %i-vector Extraction
        w = pinv(I + (TS.*repelem(n(:),numFeatures))' * T) * TSi * f(:);

        % Intersession Compensation
        w = projectionMatrix*w;

        ivectorMat(:,fileIdx) = w;
    end
    % i-vector model
    enrolledSpeakersByIdx{speakerIdx} = mean(ivectorMat,2);
end
disp("Speakers enrolled (" + round(toc,2) + " seconds).")

enrolledSpeakers = struct;
for s = 1:numSpeakers
    enrolledSpeakers.(string(speakers(s))) = enrolledSpeakersByIdx{s};
end

% Specify CSS or G-PLDA training method

scoringMethod = 'GPLDA';

% False rejection rate

speakersToTest = unique(adsEnroll.Labels);
numSpeakers = numel(speakersToTest);
scoreFRR = cell(numSpeakers,1);
tic
parfor speakerIdx = 1:numSpeakers
    adsTestSubset = subset(adsTest, ismember(adsTest.Labels, speakersToTest(speakerIdx)));
    numFiles = numel(adsTestSubset.Files);
    
    ivectorToTest = enrolledSpeakers.(string(speakersToTest(speakerIdx))); %#ok<PFBNS> 
    
    score = zeros(numFiles,1);
    for fileIdx = 1:numFiles
        audioData = read(adsTestSubset);
        
        % Extract features
        Y = helperFeatureExtraction(audioData,afe,normFactors);
        
        % Compute a posteriori log-likelihood
        logLikelihood = helperGMMLogLikelihood(Y,ubm);
        
        % Compute a posteriori normalized probability
        amax = max(logLikelihood,[],1);
        logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
        gamma = exp(logLikelihood - logLikelihoodSum)';
        
        % Compute Baum-Welch statistics
        n = sum(gamma,1);
        f = Y * gamma - n.*(ubmMu);
        
        % Extract i-vector
        w = pinv(I + (TS.*repelem(n(:),numFeatures))' * T) * TSi * f(:);
        
        % Intersession Compensation
        w = projectionMatrix*w;
        
        % Score
        if strcmpi(scoringMethod,"CSS")
            score(fileIdx) = dot(ivectorToTest,w)/(norm(w)*norm(ivectorToTest));
        else
            score(fileIdx) = gpldaScore(gpldaModel,w,ivectorToTest);
        end
    end
    scoreFRR{speakerIdx} = score;
end
disp("FRR calculated (" + round(toc,2) + " seconds).")

% False acception rate

speakersToTest = unique(adsEnroll.Labels);
numSpeakers = numel(speakersToTest);
scoreFAR = cell(numSpeakers,1);
tic
parfor speakerIdx = 1:numSpeakers
    adsTestSubset = subset(adsTest, ~ismember(adsTest.Labels,speakersToTest(speakerIdx)));
    numFiles = numel(adsTestSubset.Files);
    
    ivectorToTest = enrolledSpeakers.(string(speakersToTest(speakerIdx))); %#ok<PFBNS> 
    score = zeros(numFiles,1);
    for fileIdx = 1:numFiles
        audioData = read(adsTestSubset);
        
        % Extract features
        Y = helperFeatureExtraction(audioData,afe,normFactors);
        
        % Compute a posteriori log-likelihood
        logLikelihood = helperGMMLogLikelihood(Y,ubm);
        
        % Compute a posteriori normalized probability
        amax = max(logLikelihood,[],1);
        logLikelihoodSum = amax + log(sum(exp(logLikelihood-amax),1));
        gamma = exp(logLikelihood - logLikelihoodSum)';
        
        % Compute Baum-Welch statistics
        n = sum(gamma,1);
        f = Y * gamma - n.*(ubmMu);
        
        % Extract i-vector
        w = pinv(I + (TS.*repelem(n(:),numFeatures))' * T) * TSi * f(:);
        
        % Intersession compensation
        w = projectionMatrix * w;
        
        % Score
        if strcmpi(scoringMethod,"CSS")
            score(fileIdx) = dot(ivectorToTest,w)/(norm(w)*norm(ivectorToTest));
        else
            score(fileIdx) = gpldaScore(gpldaModel,w,ivectorToTest);
        end
    end
    scoreFAR{speakerIdx} = score;
end
disp("FAR calculated (" + round(toc,2) + " seconds).")

% Equal error rate

amin = min(cat(1,scoreFRR{:},scoreFAR{:}));
amax = max(cat(1,scoreFRR{:},scoreFAR{:}));

thresholdsToTest = linspace(amin,amax,1000);

% Compute the FRR and FAR for each of the thresholds.
if strcmpi(scoringMethod, "CSS")
    % In CSS, a larger score indicates the enroll and test ivectors are
    % similar.
    FRR = mean(cat(1,scoreFRR{:})<thresholdsToTest);
    FAR = mean(cat(1,scoreFAR{:})>thresholdsToTest);
else
    % In G-PLDA, a smaller score indicates the enroll and test ivectors are
    % similar.
    FRR = mean(cat(1,scoreFRR{:})>thresholdsToTest);
    FAR = mean(cat(1,scoreFAR{:})<thresholdsToTest);
end

[~,EERThresholdIdx] = min(abs(FAR - FRR));
EERThreshold = thresholdsToTest(EERThresholdIdx);
EER = mean([FAR(EERThresholdIdx),FRR(EERThresholdIdx)]);

figure
plot(thresholdsToTest,FAR,"k", ...
     thresholdsToTest,FRR,"b", ...
     EERThreshold,EER,"ro",MarkerFaceColor="r")
title(["Equal Error Rate = " + round(EER,4),"Threshold = " + round(EERThreshold,4)])
xlabel("Threshold")
ylabel("Error Rate")
legend("False Acceptance Rate (FAR)","False Rejection Rate (FRR)","Equal Error Rate (EER)",Location="best")
grid on
axis tight

function [features,numFrames] = helperFeatureExtraction(audioData,afe,normFactors)
    % Input:
    % audioData   - column vector of audio data
    % afe         - audioFeatureExtractor object
    % normFactors - mean and standard deviation of the features used for normalization. 
    %               If normFactors is empty, no normalization is applied.
    %
    % Output
    % features    - matrix of features extracted
    % numFrames   - number of frames (feature vectors) returned
    
    % Convert stereo -> mono
    audioData = audioData(:,1);

    % Normalize
    audioData = audioData/max(abs(audioData(:)));
    
    % Protect against NaNs
    audioData(isnan(audioData)) = 0;
    
    % Isolate speech segment
    idx = detectSpeech(audioData,afe.SampleRate);
    features = [];
    for ii = 1:size(idx,1)
        f = extract(afe,audioData(idx(ii,1):idx(ii,2)));
        features = [features;f]; %#ok<AGROW> 
    end

    % Feature normalization
    if ~isempty(normFactors)
        features = (features-normFactors.Mean')./normFactors.STD';
    end
    features = features';
    
    % Cepstral mean subtraction (for channel noise)
    if ~isempty(normFactors)
        features = features - mean(features, "all");
    end
    
    numFrames = size(features,2);
end

function L = helperGMMLogLikelihood(x, gmm)
    permuteMu = permute(gmm.mu, [1, 3, 2]);
    xMinusMu = repmat(x, 1, 1, numel(gmm.ComponentProportion)) - permuteMu;
    permuteSigma = permute(gmm.sigma,[1,3,2]);
    
    Lunweighted = -0.5*(sum(log(permuteSigma),1) + sum(xMinusMu.*(xMinusMu./permuteSigma),1) + size(gmm.mu,1)*log(2*pi));
    
    temp = squeeze(permute(Lunweighted,[1,3,2]));
    if size(temp,1)==1
        % If there is only one frame, the trailing singleton dimension was
        % removed in the permute. This accounts for that edge case.
        temp = temp';
    end
    
    L = temp + log(gmm.ComponentProportion)';
end


function score = gpldaScore(gpldaModel,w1,wt)
% Center the data
w1 = w1 - gpldaModel.mu;
wt = wt - gpldaModel.mu;

% Whiten the data
w1 = gpldaModel.WhiteningMatrix*w1;
wt = gpldaModel.WhiteningMatrix*wt;

% Length-normalize the data
w1 = w1./vecnorm(w1);
wt = wt./vecnorm(wt);

% Score the similarity of the i-vectors based on the log-likelihood.
VVt = gpldaModel.EigenVoices * gpldaModel.EigenVoices';
SVVt = gpldaModel.Sigma + VVt;

term1 = pinv([SVVt VVt;VVt SVVt]);
term2 = pinv(SVVt);

w1wt = [w1;wt];
score = w1wt'*term1*w1wt - w1'*term2*w1 - wt'*term2*wt;
end