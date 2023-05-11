%% Read file
filename = "..\Data\Harvard\Harvard_Daniel_1.wav";
[x,fs] = audioread(filename);
x = x(:, 1);
indeces = detectSpeech(x, fs);
x = x(indeces(1):indeces(2));

%% Plot zero-crossings
tiledlayout(2,1);

nexttile
win = fs*0.01;
rate = zerocrossrate(x, WindowLength=win, Method="comparison");
plot(rate)
title("Zero-Crossing Rate");
ylabel("Rate")
xlim([0, 200]);

nexttile
h = 0.1;
idu = find(rate > h);
idu(1:2) = [];
vi = [(idu-1) idu]*win;
m = sigroi2binmask(vi,length(x));
mask = signalMask([m ~m],Categories=["Unvoiced" "Voiced"],SampleRate=fs);
plotsigroi(mask,x)
ylabel("Amplitude")
