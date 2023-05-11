%% Read file
filename = "..\Data\Harvard\Harvard_Daniel_1.wav";
[x,fs] = audioread(filename);
x = x(:, 1);
indeces = detectSpeech(x, fs);
x = x(indeces(1):indeces(2));

%% Plot MFCCs
mfcc(x,fs)