clear;
%% Manage dataset
% ads = audioDatastore('temp\',Includesubfolders=true,LabelSource="folderNames");
% ads = subset(ads,ads.Labels==categorical("stop"));
% [~,fileName] = cellfun(@(x)fileparts(x),ads.Files,UniformOutput=false);
% fileName = split(fileName,"_");
% speaker = strcat("a",fileName(:,1));
% ads.Labels = categorical(speaker);
% numSpeakersToEnroll = 10;
% labelCount = countEachLabel(ads);
% forEnrollAndTestSet = labelCount{:,1}(labelCount{:,2}>=3);
% forEnroll = forEnrollAndTestSet(randi([1,numel(forEnrollAndTestSet)],numSpeakersToEnroll,1));
% tf = ismember(ads.Labels,forEnroll);
% adsEnrollAndValidate = subset(ads,tf);
% adsEnroll = splitEachLabel(adsEnrollAndValidate,2);
% 
% adsTest = subset(ads,ismember(ads.Labels,forEnrollAndTestSet));
% adsTest = subset(adsTest,~ismember(adsTest.Files,adsEnroll.Files));
% 
% forUBMTraining = ~(ismember(ads.Files,adsTest.Files) | ismember(ads.Files,adsEnroll.Files));
% adsTrainUBM = subset(ads,forUBMTraining);

ads = audioDatastore('..\Data\Harvard\');
[~, filenames] = fileparts(ads.Files);
ads.Labels = extractBetween(filenames, "_", "_");
speakers = unique(ads.Labels);
numSpeakers = numel(speakers);
[adsTrainUBM, adsEnroll, adsTest] = splitEachLabel(ads, 0.4, 0.4);

%% Extract features
[audioDataUBM, audioInfo] = read(adsTrainUBM);
fs = audioInfo.SampleRate;
reset(adsTrainUBM);

windowDuration = 0.03;
overlapDuration = 0.02;

afe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    Window=hann(round(windowDuration*fs), 'periodic'), ...
    OverlapLength=round(overlapDuration*fs), ...
    ...
    mfcc=true);

%% Train Universal Background Model (UBM)
% Extract features for all adsTrainUBM files
featuresAll = {};
if ~isempty(ver("parallel"))
    numPar = 4;
else
    numPar = 1;
end
parfor ii = 1:numPar
    adsPart = partition(ads,numPar,ii);
    featuresPart = cell(0,numel(adsPart.Files));
    for iii = 1:numel(adsPart.Files)
        audioData = read(adsPart);
        audioData = audioData(:,1);
        featuresPart{iii} = helperFeatureExtraction(audioData,afe,[]);
    end
    featuresAll = [featuresAll,featuresPart];
end

allFeatures = cat(2,featuresAll{:});
normFactors.Mean = mean(allFeatures,2,"omitnan");
normFactors.STD = std(allFeatures,[],2,"omitnan");

% Initialize
numComponents = 32;
numFeatures = 13;
alpha = ones(1,numComponents)/numComponents;
mu = randn(numFeatures, numComponents);
sigma = rand(numFeatures, numComponents);
ubm = struct(ComponentProportion=alpha, mu=mu, sigma=sigma);

% Define stopping criteria
maxIter = 200;
targetLogLikelihood = 0;
tol = 0.5;
pastL = -inf; % initialization of previous log-likelihood

% Train
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
disp("UBM training completed in " + round(toc,2) + " seconds.")

%% Enroll speakers
relevanceFactor = 16;
numComponents = numel(ubm.ComponentProportion);
numFeatures = size(ubm.mu,1);

speakers = unique(adsEnroll.Labels);
numSpeakers = numel(speakers);

gmmCellArray = cell(numSpeakers,1);
tic
for ii = 1:numSpeakers
    % Subset the datastore to the speaker you are adapting.
    adsTrainSubset = subset(adsEnroll,string(adsEnroll.Labels)==speakers{ii});
    
    N = zeros(1,numComponents);
    F = zeros(numFeatures,numComponents);
    S = zeros(numFeatures,numComponents);
    while hasdata(adsTrainSubset)
        audioData = read(adsTrainSubset);
        features = helperFeatureExtraction(audioData,afe,normFactors);
        [n,f,s,l] = helperExpectation(features,ubm);
        N = N + n;
        F = F + f;
        S = S + s;
    end
    
    % Determine the maximum likelihood
    gmm = helperMaximization(N,F,S);
    
    % Determine adaption coefficient
    alpha = N./(N + relevanceFactor);
    
    % Adapt the means
    gmm.mu = alpha.*gmm.mu + (1-alpha).*ubm.mu;
    
    % Adapt the variances
    gmm.sigma = alpha.*(S./N) + (1-alpha).*(ubm.sigma + ubm.mu.^2) - gmm.mu.^2;
    gmm.sigma = max(gmm.sigma,eps);
    
    % Adapt the weights
    gmm.ComponentProportion = alpha.*(N/sum(N)) + (1-alpha).*ubm.ComponentProportion;
    gmm.ComponentProportion = gmm.ComponentProportion./sum(gmm.ComponentProportion);

    gmmCellArray{ii} = gmm;
end
disp("Enrollment completed in " + round(toc,2) + " seconds.")

for i = 1:numel(gmmCellArray)
    enrolledGMMs.(string(speakers(i))) = gmmCellArray{i};
end


% speakers = unique(adsEnroll.Labels);
% numSpeakers = numel(speakers);
% for ii = 1:numSpeakers
%     enrolledGMMs.(string(speakers{ii})) = helperEnroll(ubm, afe, normFactors, adsEnroll);
% end

%% Get FAR, FRR, ERR and plot
thresholds = -0.5:0.01:2.5;

% Calculate FRR
speakers = unique(adsEnroll.Labels);
numSpeakers = numel(speakers);
llr = cell(numSpeakers,1);
tic
for speakerIdx = 1:numSpeakers
    localGMM = enrolledGMMs.(string(speakers(speakerIdx))); 
    adsTestSubset = subset(adsTest,string(adsTest.Labels)==speakers{speakerIdx});
    llrPerSpeaker = zeros(numel(adsTestSubset.Files),1);
    for fileIdx = 1:numel(adsTestSubset.Files)
        audioData = read(adsTestSubset);
        [x,numFrames] = helperFeatureExtraction(audioData,afe,normFactors);
        
        logLikelihood = helperGMMLogLikelihood(x,localGMM);
        Lspeaker = helperLogSumExp(logLikelihood);
        
        logLikelihood = helperGMMLogLikelihood(x,ubm);
        Lubm = helperLogSumExp(logLikelihood);
        
        llrPerSpeaker(fileIdx) = mean(movmedian(Lspeaker - Lubm,3));
    end
    llr{speakerIdx} = llrPerSpeaker;
end
disp("False rejection rate computed in " + round(toc,2) + " seconds.")
llr = cat(1,llr{:});
FRR = mean(llr<thresholds);

% Calculate FAR
speakersTest = unique(adsTest.Labels);
llr = cell(numSpeakers,1);
tic
for speakerIdx = 1:numel(speakers)
    localGMM = enrolledGMMs.(string(speakers(speakerIdx)));
    adsTestSubset = subset(adsTest,string(adsTest.Labels)~=speakers{speakerIdx});
    llrPerSpeaker = zeros(numel(adsTestSubset.Files),1);
    for fileIdx = 1:numel(adsTestSubset.Files)
        audioData = read(adsTestSubset);
        [x,numFrames] = helperFeatureExtraction(audioData,afe,normFactors);
        
        logLikelihood = helperGMMLogLikelihood(x,localGMM);
        Lspeaker = helperLogSumExp(logLikelihood);
        
        logLikelihood = helperGMMLogLikelihood(x,ubm);
        Lubm = helperLogSumExp(logLikelihood);
        
        llrPerSpeaker(fileIdx) = mean(movmedian(Lspeaker - Lubm,3));
    end
    llr{speakerIdx} = llrPerSpeaker;
end
disp("False Acceptance Rate computed in " + round(toc,2) + " seconds.")
llr = cat(1,llr{:});
FAR = mean(llr>thresholds);

% Calculate EER
[~,EERThresholdIdx] = min(abs(FAR - FRR));
EERThreshold = thresholds(EERThresholdIdx);
EER = mean([FAR(EERThresholdIdx),FRR(EERThresholdIdx)]);

% Plot results
plot(thresholds,FAR,"k", ...
     thresholds,FRR,"b", ...
     EERThreshold,EER,"ro",MarkerFaceColor="r")
title(["Equal Error Rate = " + round(EER,2), "Threshold = " + round(EERThreshold,2)])
xlabel("Threshold")
ylabel("Error Rate")
legend("False Acceptance Rate (FAR)","False Rejection Rate (FRR)","Equal Error Rate (EER)")
grid on