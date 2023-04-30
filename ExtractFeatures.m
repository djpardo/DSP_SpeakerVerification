function [features] = ExtractFeatures(audioData, fs)

    % Setup audio extractor
    windowDuration = 0.04;
    hopDuration = 0.01;
    windowSamples = round(windowDuration*fs);
    hopSamples = round(hopDuration*fs);
    overlapSamples = windowSamples - hopSamples;

    extractor = audioFeatureExtractor( ...
        SampleRate = fs, ...
        Window = hann(windowSamples, "periodic"), ...
        OverlapLength = overlapSamples, ...
        ...
        mfcc = true);

    % Normalize
    audioData = audioData ./ max(abs(audioData));

    % Remove non-speech components
    audioData(isnan(audioData)) = 0;                % remove NaN's
    index = detectSpeech(audioData, fs);            % get indeces of speech
    audioData = audioData(index(1,1):index(1,2));   % trim data

    % Extract features
    features = extract(extractor, audioData);

end