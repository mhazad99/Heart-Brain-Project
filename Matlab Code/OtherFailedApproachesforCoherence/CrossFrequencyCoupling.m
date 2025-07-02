clc
clear
close all
tic
%% Load signals and Initials
Filelocation = "C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Depression\RET_0002.edf";
TextfileName = "C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Depression\RET_0002.txt";
[rawdata] = ReadDataset(Filelocation);
[newy, Adjust_num] = Hypnogram(TextfileName);
fs = rawdata.fs;
%% Things need to be changed based on subjects
PostWakeNum = 25;
CleanedTime = 4980*5;
load Res.mat
%% Refrencing ECG & EEG
%ECG = ECG1 - ECG2; %This referencing does not good results
EEG = rawdata.F3 - ((rawdata.M1 + rawdata.M2)/2);
% SleepEEG and SleepECG are the signals with removed pre-post wake signals
%% EEG filtering and artifact removal
FilteredEEG = EEGFiltering(EEG, fs, Adjust_num, PostWakeNum);
%% ECG filtering and artifact removal
FilteredECG = ECGFiltering(rawdata.ECG1, fs, Adjust_num, PostWakeNum);
%% Computing HR from ECG
RRI = Res.RRs;
HR = 60./RRI;
ResampledHR = resample(HR, floor(length(FilteredECG)/length(HR)), 1);

%% Computing PSD of ... band
bounderies = [0.5, 4, 8, 12, 16, 3.5, 7.5, 11.5, 15.5, 19.5];
N = CleanedTime/30;
for iter = 1:length(bounderies)/2
    Filter = designfilt('bandpassiir', 'FilterOrder', 6,...
        'HalfPowerFrequency1', bounderies(iter),'HalfPowerFrequency2', ...
        bounderies(iter+length(bounderies)/2),'SampleRate', 256);
    FreqBand = filtfilt(Filter, FilteredEEG);
    for item = 1:N
        window = FreqBand(1+((item-1)*fs*30):item*fs*30);
        window = window.*hamming(length(window));
        % Calculate Hilbert Transform to find the power time series for each window
        Hilbertfun = hilbert(window);
        WindowPower = abs(Hilbertfun).^2;
        N1 = length(WindowPower);
        DTFT = fft(WindowPower);
        DTFT = DTFT(1:(N1/2)+1);
        DTFT = abs(DTFT);
        freq = 0:fs/N1:fs/2;
        Dfreq = max(findpeaks(DTFT));
        idx = find(DTFT == Dfreq);
        freqval = freq(idx);
        freqvalstorage(iter, item) = freqval;
        %         figure
        %         plot(freq, DTFT)
        %         figure
        %         loglog(freq, 10*log10(abs(DTFT)))
        % Calculating the phase of the power time series
        SecondHilbert = hilbert(WindowPower);
        WindowPhaseStorage(:, item) = angle(SecondHilbert);
    end
    for item = 1:N
        Filter1 = designfilt('bandpassiir', 'FilterOrder', 6,...
            'HalfPowerFrequency1', freqvalstorage(iter, item)-freqvalstorage(iter, item)/2, ...
            'HalfPowerFrequency2', freqvalstorage(iter, item)+freqvalstorage(iter, item)/2,'SampleRate', fs);
        ECGLowfreq = filtfilt(Filter1, FilteredECG);
        ECGwindow = ECGLowfreq(1+((item-1)*fs*30):item*fs*30);
        ECGwindow = ECGwindow.*hamming(length(ECGwindow));
        ECGHilbert = hilbert(ECGwindow);
        LowerECGPhaseStorage(:, item) = angle(ECGHilbert);
    end
    for Numberofepochs = 1:N
        SIValue = 0;
        n = length(SecondHilbert);
        for item2 = 1:n
            SIValue = SIValue + exp(1i*(LowerECGPhaseStorage(item2,Numberofepochs) ...
                - WindowPhaseStorage(item2,Numberofepochs)));
        end
        SI = SIValue/n;
        SIm = abs(SI);
        SIp = atan(imag(SI)/real(SI));
        SImStorage(iter, Numberofepochs) = SIm;
        SIpStorage(iter, Numberofepochs) = SIp;
        SI = 0;
    end
end
toc
%save SImStorage SImStorage
%save SIpStorage SIpStorage
figure
labels = {'Delta','Theta', 'Alpha', 'Sigma', 'Beta'};
for hist = 1:5
    subplot(5, 1, hist)
    h = histogram(SImStorage(hist, :));
    title(labels(hist))
end
xlabel('Magnitude of SI')
figure
labels = {'Delta','Theta', 'Alpha', 'Sigma', 'Beta'};
for hist = 1:5
    subplot(5, 1, hist)
    h = histogram(SIpStorage(hist, :));
    title(labels(hist))
end
xlabel('Phase of SI')
% x = [1 N];
% y = [1 5];
% imagesc(SImStorage)
% colorbar
% labels = {'Delta','Theta', 'Alpha', 'Sigma', 'Beta'};
% yticks(1:5)
% yticklabels(labels)
% xlabel('Epochs','FontSize',12)
% ylabel('EEG Frequency Bands','FontSize',12)
% cb = colorbar;
% cb.Title.String = 'Magnitude of SI';
% figure
% x = [1 N];
% y = [1 5];
% imagesc(SIpStorage)
% colorbar
% labels = {'Delta','Theta', 'Alpha', 'Sigma', 'Beta'};
% yticks(1:5)
% yticklabels(labels)
% xlabel('Epochs','FontSize',12)
% ylabel('EEG Frequency Bands','FontSize',12)
% cb = colorbar;
% cb.Title.String = 'Phase of SI';
% cb.Limits = [-pi/2, pi/2];
% figure
% imagesc(freqvalstorage)
% colorbar
% labels = {'Delta','Theta', 'Alpha', 'Sigma', 'Beta'};
% yticks(1:5)
% yticklabels(labels)
% xlabel('Epochs','FontSize',12)
% ylabel('Dominant frequency of each power time series','FontSize',12)
% cb = colorbar;
% cb.Title.String = 'lower synch. freq.';
