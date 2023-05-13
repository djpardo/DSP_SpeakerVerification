%% Read file
filename = "..\Data\Harvard\Harvard_Daniel_1.wav";
[audioIn,fs] = audioread(filename);
audioIn = audioIn(:, 1);
indeces = detectSpeech(audioIn, fs);
audioIn = audioIn(indeces(1):indeces(2));

%% Plot energy
aFE = audioFeatureExtractor("SampleRate",fs, ...
    shortTimeEnergy=true);

features = extract(aFE,audioIn);
e = abs(audioIn).^2;
% features = (features - mean(features,1))./std(features,[],1);