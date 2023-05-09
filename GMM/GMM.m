clear;
%% Manage dataset
ads = audioDatastore('..\Data\Password\');
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

%% Train Universal Background Model (UBM)
ubm = helperTrainUBM(adsTrainUBM, afe, numPar, normFactors);
disp("UBM training completed in " + round(toc,2) + " seconds.")

%% Enroll speakers
for ii = 1:numSpeakers
    enrolledGMMs.(string(speakers{ii})) = helperEnroll(ubm, afe, normFactors, adsEnroll);
end

%% Get FAR, FRR, ERR and plot
thresholds = -0.5:0.01:2.5;
FRR = helperFalseRejectionRate(adsEnroll, adsTest, afe, ubm, enrolledGMMs, thresholds, normFactors);
FAR = helperFalseAcceptanceRate(adsEnroll, adsTest, afe, ubm, enrolledGMMs, thresholds, normFactors);

[~,EERThresholdIdx] = min(abs(FAR - FRR));
EERThreshold = thresholds(EERThresholdIdx);
EER = mean([FAR(EERThresholdIdx),FRR(EERThresholdIdx)]);

plot(thresholds,FAR,"k", ...
     thresholds,FRR,"b", ...
     EERThreshold,EER,"ro",MarkerFaceColor="r")
title(["Equal Error Rate = " + round(EER,2), "Threshold = " + round(EERThreshold,2)])
xlabel("Threshold")
ylabel("Error Rate")
legend("False Acceptance Rate (FAR)","False Rejection Rate (FRR)","Equal Error Rate (EER)")
grid on