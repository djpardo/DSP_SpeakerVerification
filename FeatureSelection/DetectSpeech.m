%% Load file
[audioIn, fs] = audioread("..\Data\Password\Password_Daniel_1.wav");
audioIn = audioIn(:, 1);

%% Detect speech and trim
% Plot detection
detectSpeech(audioIn, fs);

% Trim file and play
indeces = detectSpeech(audioIn, fs);
audioIn = audioIn(indeces(1):indeces(2));
sound(audioIn, fs);