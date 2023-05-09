function [FAR, FRR, EER] = helperVerify(adsEnroll, adsTest, afe, ubm, enrolledGMMs, thresholds, normFactors)

    speakers = unique(adsEnroll.Labels);
    numSpeakers = numel(speakers);
    llr = cell(numSpeakers,1);
    tic
    for speakerIdx = 1:numSpeakers

        localGMM = enrolledGMMs{speakerIdx, 1}; 
        adsTestSubset = subset(adsTest,adsTest.Labels==string(speakers(speakerIdx)));
        llrPerSpeaker = zeros(numel(adsTestSubset.Files),1);
        for fileIdx = 1:numel(adsTestSubset.Files)
            audioData = read(adsTestSubset);
            audioData = audioData(:,1);
            
            features = helperFeatureExtraction(audioData,afe, normFactors);
            
            logLikelihood = helperGMMLogLikelihood(features,localGMM);
            Lspeaker = helperLogSumExp(logLikelihood);
            
            logLikelihood = helperGMMLogLikelihood(features,ubm);
            Lubm = helperLogSumExp(logLikelihood);
            
            llrPerSpeaker(fileIdx) = mean(movmedian(Lspeaker - Lubm,3));
        end
        llr{speakerIdx} = llrPerSpeaker;

    end

    llr = cat(1,llr{:});
    FRR = mean(llr<thresholds);
    FAR = mean(llr>thresholds);

    [~,EERThresholdIdx] = min(abs(FAR - FRR));
    EERx = thresholds(EERThresholdIdx);
    EERy = mean([FAR(EERThresholdIdx),FRR(EERThresholdIdx)]);
    EER = [EERx, EERy];

end

