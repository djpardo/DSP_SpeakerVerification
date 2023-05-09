function [features, numFrames] = helperFeatureExtraction(audioData, afe, normFactors)
    % Convert stereo -> mono
    audioData = audioData(:,1);

    % Normalize
    audioData = audioData/max(abs(audioData(:)));
    
    % Protect against NaNs
    audioData(isnan(audioData)) = 0;
    
    % Isolate speech segment
    idx = detectSpeech(audioData, afe.SampleRate);
    audioData = audioData(idx(1, 1):idx(1, 2));
    
    % Feature extraction
    features = extract(afe,audioData);

    % Feature normalization
    if ~isempty(normFactors)
        features = (features-normFactors.Mean')./normFactors.STD';
    end
    features = features';
    
    % Cepstral mean subtraction (for channel noise)
    if ~isempty(normFactors)
        features = features - mean(features,"all");
    end

    numFrames = size(features, 2);
end