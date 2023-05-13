%% Read file
[sample1,fs] = audioread("..\Data\Hello\Hello_Daniel_1.wav");
sample1 = sample1(:, 1);
indeces = detectSpeech(sample1, fs);
sample1 = sample1(indeces(1):indeces(2));

[sample2,fs] = audioread("..\Data\Hello\Hello_Bridget_1.wav");
sample2 = sample2(:, 1);
indeces = detectSpeech(sample2, fs);
sample2 = sample2(indeces(1):indeces(2));

%% Plot signal, f0
tiledlayout(2,2)

nexttile
t = (0:length(sample1)-1)/fs;
plot(t,sample1)
title("Speaker 1 (Male)")
xlabel("Time (s)")
ylabel("Amplitude")
grid minor
axis tight

nexttile
t = (0:length(sample2)-1)/fs;
plot(t,sample2)
title("Speaker 2 (Female)")
xlabel("Time (s)")
ylabel("Amplitude")
grid minor
axis tight

nexttile
pitch(sample1,fs)

nexttile
pitch(sample2, fs)