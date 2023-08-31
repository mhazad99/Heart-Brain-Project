clc
clear
close all
%% Load Data
load Res.mat % remember to save Res.mat in another folder then delete this
%folderPath = ['C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Brainstorm\' ...
%   'brainstorm_db\FirstSubject\data\RET_0002\NIGHT']; % Update with your folder path
%%% folder path in laptop: 
folderPath = ['D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\Brainstorm\' ...
    'brainstorm_db\FirstSubject\data\RET_0002\NIGHT'];
[CleanedEEGinfo] = EEGArtifactRemoval(folderPath);
F3 = CleanedEEGinfo.CleanedF3'; C3 = CleanedEEGinfo.CleanedC3';
O1 = CleanedEEGinfo.CleanedO1';
fs = 256;
%% Upsampling the signal without distortion
% Load the original signal
RRI = Res.RRs;
HR = 60./RRI;
Fs = length(HR)/(length(F3)/256); % As an example for F3
% Define the target integer sampling rate
Fs_target = 256;

% Calculate the upsampling factor
upsampling_factor = Fs_target / Fs;
interpolatedHR = interp(HR, floor(upsampling_factor));
t1 = 0:29593-1; %seconds % change it for general form!
t2 = 0:1/186:29593-1/186; % change it for general form!
% figure
% plot(t1, HR)
% hold on
% plot(t2, interpolatedHR)
CleanedTime = (length(F3)-(2*256*30))/256; % change it for general form!
interpolatedHR = interpolatedHR(1:5498880,1); % change it for general form!
Filter1 = designfilt('bandpassiir', 'FilterOrder', 6,...
    'HalfPowerFrequency1', 0.15,'HalfPowerFrequency2', ...
    0.4,'SampleRate', 256);
HFHR = filtfilt(Filter1, interpolatedHR);
%% Calculating the SI between two signals in each 30 second epoch
bounderies = [0.5, 4, 8, 12, 16, 3.5, 7.5, 11.5, 15.5, 19.5];
N = CleanedTime/30; % in this case we decreased two epochs from the EEG signal to align the timing with the HR (change it for general form)
for iter = 1:length(bounderies)/2
    Filter = designfilt('bandpassiir', 'FilterOrder', 6,...
        'HalfPowerFrequency1', bounderies(iter),'HalfPowerFrequency2', ...
        bounderies(iter+length(bounderies)/2),'SampleRate', 256);
    FreqBand = filtfilt(Filter, F3);
    for item = 1:N
        EEGwindow = FreqBand(1+((item-1)*fs*30):item*fs*30);
        EEGwindow = EEGwindow.*hamming(length(EEGwindow));
        EEGHilbert = hilbert(EEGwindow);
        EEGPhaseStorage(:, item, iter) = angle(EEGHilbert);
    end
    for item = 1:N
        HRwindow = HFHR(1+((item-1)*fs*30):item*fs*30);
        HRwindow = HRwindow.*hamming(length(HRwindow));
        HRHilbert = hilbert(HRwindow);
        HRPhaseStorage(:, item, iter) = angle(HRHilbert);
    end
    for Numberofepochs = 1:N
        SIValue = 0;
        n = length(EEGHilbert);
        for item2 = 1:n
            SIValue = SIValue + exp(1i*(HRPhaseStorage(item2,Numberofepochs, iter) ...
                - EEGPhaseStorage(item2,Numberofepochs, iter)));
        end
        SI = SIValue/n;
        SIm = abs(SI);
        SIp = atan(imag(SI)/real(SI));
        SImStorage(iter, Numberofepochs) = SIm;
        SIpStorage(iter, Numberofepochs) = SIp;
        SI = 0;
    end
end

% Calculating phase difference
for i = 1:5
    PhaseDifference(:,:,i) = mod((HRPhaseStorage(:,:,i) - EEGPhaseStorage(:,:,i)), 2*pi);
    MeanPhaseDifference(:, i) = mean(PhaseDifference(:,:,i), 2);
    % Polar Histogram plot
    %figure
    %polarhistogram(MeanPhaseDifference(:, i),20)
end

% Calculating phase difference for one epoch
figure
subplot(3,1,1)
plot(0:1/fs:30-1/fs, HRPhaseStorage(:,1,1), 'b')
title('Phase of High frequency of HR in 30 seconds')
xlabel('time','FontSize',12)
ylabel('phase','FontSize',12)
subplot(3,1,2)
plot(0:1/fs:30-1/fs, EEGPhaseStorage(:,1,1), 'r')
title('Delta Frequency band (0.5-3.5 Hz) phase in 30 seconds')
xlabel('time','FontSize',12)
ylabel('phase','FontSize',12)
subplot(3,1,3)
polarhistogram(PhaseDifference(:,1,1),20)

% finding the frequencies of oscillation for each phase time series
N1 = length(HRPhaseStorage(:,1,1));
DTFT = fft(HRPhaseStorage(:,1,1));
DTFT = DTFT(1:(N1/2)+1);
DTFT = abs(DTFT);
freq = 0:fs/N1:fs/2;
figure
plot(freq, 10*log10(DTFT))
DTFT2 = fft(EEGPhaseStorage(:,1,1));
DTFT2 = DTFT2(1:(N1/2)+1);
DTFT2 = abs(DTFT2);
Filter0 = designfilt('lowpassiir', 'FilterOrder', 6,...
'PassbandFrequency', 6,'PassbandRipple', 0.2,'SampleRate', 256);
DTFT2 = filtfilt(Filter0, DTFT2);
hold on
plot(freq, 10*log10(DTFT2))
legend('fft of HR phase', 'fft of Delta frequency band phase')
