%% Load file
[audioIn, fs] = audioread("Data\Harvard\Harvard_Daniel_5.wav");
audioIn = audioIn(:, 1);

%% Detect speech and trim
indeces = detectSpeech(audioIn, fs);
audioIn = audioIn(indeces(1):indeces(2));
sound(audioIn, fs);