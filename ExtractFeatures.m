function [features] = ExtractFeatures(audioData, fs)

    % Normalize
    audioData = audioData ./ max(abs(audioData));

    % Isolate speech segment
    index = detectSpeech(audioData, fs);
    audioData = audioData(index(1,1):index(1,2));

    % Create audioFeatureExtractor object
    extractor = audioFeatureExtractor( ...
        SampleRate = fs, ...
        Window = window, ...
        OverlapLength = overlap_length, ...
        ...
        mfcc = true);

    % Extract features
    features = extract(extractor, audioData);

end