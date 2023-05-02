function varargout = helperVisualizeBuffer(x,fs,nvargs)
arguments
    x (:,:) {mustBeReal,mustBeA(x,["single","double"])}
    fs (1,1) {mustBeReal,mustBeA(fs,["single","double"]),mustBePositive}

    nvargs.Window {mustBeReal,mustBeA(nvargs.Window,["single","double"])} = []
    nvargs.WindowLength {mustBeScalarOrEmpty,mustBeReal,mustBePositive,mustBeA(nvargs.WindowLength,["single","double"])} = []
    nvargs.WindowDuration {mustBeScalarOrEmpty,mustBeReal,mustBePositive,mustBeA(nvargs.WindowDuration,["single","double"])} = []

    nvargs.OverlapLength {mustBeScalarOrEmpty,mustBeInteger} = []
    nvargs.OverlapDuration {mustBeScalarOrEmpty,mustBeReal,mustBePositive,mustBeA(nvargs.OverlapDuration,["single","double"])} = []
    nvargs.OverlapPercent {mustBeScalarOrEmpty,mustBeReal,mustBeA(nvargs.OverlapPercent,["single","double"]),mustBeInRange(nvargs.OverlapPercent,0,99)} = []
    nvargs.HopLength {mustBeScalarOrEmpty,mustBeInteger,mustBeReal,mustBeA(nvargs.HopLength,["single","double"]),mustBePositive} = []
    nvargs.HopDuration {mustBeScalarOrEmpty,mustBeReal,mustBeA(nvargs.HopDuration,["single","double"]),mustBePositive} = []

    nvargs.PlotInTime = true;
end
MustBeExclusive(nvargs.Window,nvargs.WindowLength,nvargs.WindowDuration)
MustBeExclusive(nvargs.OverlapLength,nvargs.OverlapDuration,nvargs.OverlapPercent,nvargs.HopLength,nvargs.HopDuration)

% Get the window length in samples.
windowSpecifier = find(~cellfun("isempty",{nvargs.Window,nvargs.WindowLength,nvargs.WindowDuration}),1);
switch windowSpecifier
    case 1
        windowLength = numel(nvargs.Window);
    case 2
        windowLength = nvargs.WindowLength;
    case 3
        windowLength = round(nvargs.WindowDuration*fs);
    otherwise
        error("Window, WindowLength, or WindowDuration must be specified.")
end

% Get the overlap length in samples
rateSpecifier = find(~cellfun("isempty",{nvargs.OverlapLength,nvargs.OverlapDuration,nvargs.OverlapPercent,nvargs.HopLength,nvargs.HopDuration}),1);
switch rateSpecifier
    case 1
        overlapLength = nvargs.OverlapLength;
    case 2
        overlapLength = round(nvargs.OverlapDuration*fs);
    case 3
        overlapLength = round(windowLength*nvargs.OverlapPercent/100);
    case 4
        overlapLength = windowLength - nvargs.HopLength;
    case 5
        overlapLength = windowLength - round(nvargs.HopDuration*fs);
    otherwise
        error("OverlapLength, OverlapPercent, OverlapDuration, HopLength, or HopDuration must be specified.")
end

if windowLength<=overlapLength
    error("Overlap length must be less than window length.")
end

% Buffer the signal.
hopLength = windowLength - overlapLength;
xb = audio.internal.buffer(x,windowLength,hopLength);
xb = reshape(xb,size(xb,1),[],size(x,2));
numHops = size(xb,2);

switch nargout
    case 1
        % If user requested buffered signal, output it.
        varargout{1} = xb;
    case 0
        % Plot the buffered signal along with the windows.

        inputStamps = 1:size(x,1);
        frameStamps = round(windowLength/2) + hopLength*(0:numHops-1);

        figure(Position=[200 200 900 600]);
        ymin = min(x,[],"all");
        ymax = max(x,[],"all");
        xmin = 0;%frameStamps(1) - ceil(windowLength/2);
        xmax = size(x,1);%frameStamps(end) + floor(windowLength/2);

        if nvargs.PlotInTime
            frameStamps = frameStamps/fs;
            xmin = xmin/fs;
            xmax = xmax/fs;
            inputStamps = (0:size(x,1)-1)/fs;
        end

        % Plot input
        tiledlayout(2,1)

        nexttile
        plot(inputStamps,x)
        axis([xmin xmax ymin ymax])
        ylabel("Amplitude")
        if nvargs.PlotInTime
            xlabel("Time (s)")
        else
            xlabel("Sample")
        end
        title("Signal")

        nexttile
        numFramesOverlapped = ceil(windowLength/hopLength);
        p = linspace(ymin,ymax,numFramesOverlapped+1);

        if nvargs.PlotInTime
            frontEdge = frameStamps - windowLength/2/fs;
            backEdge = frameStamps + windowLength/2/fs;
        else
            frontEdge = frameStamps - floor(windowLength/2);
            backEdge = frameStamps + ceil(windowLength/2);
        end
        color = {[0 0.4470 0.7410],[0.8500 0.3250 0.0980],[0.9290 0.6940 0.1250],[0.4940 0.1840 0.5560],[0.4660 0.6740 0.1880],[0.3010 0.7450 0.9330],[0.6350 0.0780 0.1840]};
        color = color(1:5);
        idx = 1;
        cidx = 1;
        pp = {-2 -1 0 1 2};
        ppIdx=1;
        for ii = 1:numHops
            plot(linspace(frontEdge(ii),backEdge(ii),windowLength),pp{ppIdx} + xb(:,ii),Color=color{cidx}),hold on
            ppIdx = ppIdx+1;
            if ppIdx==6
                ppIdx=1;
            end
            if idx+1 == numel(p)
                idx = 1;
            else
                idx = idx+1;
            end
            if cidx==5
                cidx=1;
            else
                cidx = cidx+1;
            end
        end
        hold off
        axis([xmin xmax -2.5 2.5])
        title("Analysis Windows of Signal")
        xlabel("Time (s)")
        set(gca,ytick=[])
end
end

function MustBeExclusive(varargin)
if sum(~cellfun("isempty",varargin)) ~= 1
    eid = "XY:NotExclusive";
    msg = "X and Y must be exclusive.";
    throwAsCaller(MException(eid,msg))
end
end