%% Manage dataset
ads = audioDatastore('Data\Password\');
[~, filenames] = fileparts(ads.Files);
ads.Labels = extractBetween(filenames, "_", "_");
[adsTrain, adsTest] = splitEachLabel(ads, 0.8);

%% Read sample
[x, xinfo] = read(adsTrain);
x = x(:, 1);                    % Sample is in stereo - convert to mono
fs = xinfo.SampleRate;

% signalDuration = audioinfo(string(adsTrain.Files(1))).Duration;
% t = 0:seconds(1/fs):seconds(signalDuration);
% t = t(1:end-1);

% figure
% plot(t,x)
% title("Label: " + xinfo.Label)
% grid on
% axis tight
% ylabel("Amplitude")
% xlabel("Time (s)")
% sound(x, fs)          % Play sample

%% Create audioFeatureExtractor to extract features
windowDuration = 0.03;
overlapDuration = 0.02;
% helperVisualizeBuffer(x, fs, WindowDuration=windowDuration, OverlapDuration=overlapDuration);

afe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    Window=hann(round(windowDuration*fs), 'periodic'), ...
    OverlapLength=round(overlapDuration*fs));

% Enable chosen features
afe.pitch = true;
afe.zerocrossrate = true;
afe.shortTimeEnergy = true;
afe.mfcc = true;
afe.mfccDelta = true;
afe.mfccDeltaDelta = true;

outputMap = info(afe);

%% Use Transform to extract features for entire datastore
adsTrainTransform = transform(adsTrain, @(x){extract(afe,x)});
features = readall(adsTrainTransform, UseParallel=true);        % Set UseParallel=true to use Parallel Computing Toolbox

N = cellfun(@(x)size(x, 1), features);
T = repelem(adsTrain.Labels, N);            % labels
X = cat(1, features{:});
X = X(:,:,1);

%% Perform feature selection
rng('default');
[featureSelectionIndex, featureSelectionScores] = fscmrmr(X, T);
featurenames = uniqueFeatureName(outputMap);
featurenamesSorted = featurenames(featureSelectionIndex);

% Plot scores of all features
% figure
% bar(featureSelectionScores)
% ylabel("Feature Score")
% xlabel("Feature Matrix Column")

% Plot top 20 features
figure
bar(reordercats(categorical(featurenames),featurenamesSorted),featureSelectionScores)
ylabel("Feature Score")
xlabel("Feature Matrix Column")
xlim([featurenamesSorted(1),featurenamesSorted(20)])