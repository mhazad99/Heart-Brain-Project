clc
clear
close all

load ECG1.mat
load ECG2.mat
load F3.mat
load M1.mat
load M2.mat
load Adjust_num.mat

fs = 256;
T = 0:1/fs:(4980*5)-1/fs; % 4980*5 is the sleep time in second              %% Things to be changed based on subjects
EEG = F3 - ((M1 + M2)/2);
SleepEEG = EEG(Adjust_num*fs*30:end-25*fs-1);                            %% Things to be changed based on subjects
SleepLengthSig = length(SleepEEG);
SleepECG = ECG1(Adjust_num*fs*30:end-25*fs-1);                            %% Things to be changed based on subjects

Filter11 = designfilt('lowpassiir', 'FilterOrder', 3,...
    'PassbandFrequency', 50, 'SampleRate', fs);
FilteredECG11 = filtfilt(Filter11, SleepECG);
ECGDetrended  = detrend(FilteredECG11, 12);
%% Computing HR from ECG
[qrs_amp_raw, qrs_i_raw, delay] = pan_tompkin(ECGDetrended, fs, 0);
t = T(qrs_i_raw);
for item = 1: length(qrs_i_raw)-1
    RRI(item) = T(qrs_i_raw(item+1)) - T(qrs_i_raw(item));
end
fsRRI = length(RRI)/(SleepLengthSig/fs);
% figure
% plot(t(1:end-1), RRI)
DownsampleRRI = resample(RRI, (SleepLengthSig/fs), length(RRI));
HR = 60./DownsampleRRI;
% figure
% plot(T, HR)

%% Filtering EEG
Filter1 = designfilt('bandpassiir', 'FilterOrder', 6,...
    'HalfPowerFrequency1', 0.4,'HalfPowerFrequency2', 30,...
    'SampleRate', 256);
FilteredEEG = filtfilt(Filter1, SleepEEG);
%% Computing PSD of ... band
% bounderies = [0.5, 4, 8, 12, 16, 3.5, 7.5, 11.5, 15.5, 19.5];
Filter2 = designfilt('bandpassiir', 'FilterOrder', 6,...
    'HalfPowerFrequency1', 16,'HalfPowerFrequency2', 19.5,...
    'SampleRate', 256);
Theta = filtfilt(Filter2, FilteredEEG);

% Filter2 = designfilt('bandpassiir', 'FilterOrder', 6,...
%     'HalfPowerFrequency1', 12,'HalfPowerFrequency2', 15.5,...
%     'SampleRate', 256);
% Sigma = filtfilt(Filter2, FilteredEEG);
% figure
% plot(T, Delta, 'b')
% xlabel('t (s)','FontSize',20)
% ylabel('Delta','FontSize',20)

%% PSD Analysis of EEG freq bands
% N = length (FilteredEEG);
% EEGdft = fft(FilteredEEG);
% EEGdft = EEGdft(1:(N/2)+1);
% PSDEEG = (1/(fs*N)) * abs(EEGdft).^2;
% PSDEEG(2:end - 1) = 2 * PSDEEG(2:end - 1);
% freq = 0:fs/N:fs/2;
% figure
% plot(freq, PSDEEG, 'b')
% xlabel('frequency (Hz)','FontSize',20)
% ylabel('PSD of EEG','FontSize',20)

for item = 1:4980*5                                                         %% Things to be changed based on subjects
    window = Theta(1+((item-1)*256):item*256);
%     N = length(window);
%     EEGdft = fft(window);
%     EEGdft = EEGdft(1:(N/2)+1);
%     PSDEEG = (1/(fs*N)) * abs(EEGdft).^2;
%     freq = 0:fs/N:fs/2;
%     Power1 = trapz(PSDEEG);
    pRMS1 = rms(window)^2;
    PSTimeSeries1(item) = pRMS1;
end
% for item = 1:24930
%     window = Sigma(1+((item-1)*256):item*256);
%     pRMS2 = rms(window)^2;
%     PSTimeSeries2(item) = pRMS2;
% end
t1 = 0:4980*5-1;                                                            %% Things to be changed based on subjects
figure
subplot(3,1,1)
plot(t1, HR)
xlabel('t (s)','FontSize',20)
% plot(t1, PSTimeSeries2, 'k')
% ylabel('PS of sigma band','FontSize',12)
%ylabel('1/RR','FontSize',12)
subplot(3,1,2)
plot(t1, PSTimeSeries1, 'r')
%xlabel('time (s)','FontSize',20)
ylabel('PS of Theta band','FontSize',12)

for item = 1:(4980*5)/30 %(24930/30) number of sleep epochs                 %% Things to be changed based on subjects
    window1 = HR(1+(item-1)*30:item*30);
    NormalizedWindow1 = normalize(window1);
    window2 = PSTimeSeries1(1+(item-1)*30:item*30);
    NormalizedWindow2 = normalize(window2);
    %window3 = PSTimeSeries2(1+(item-1)*30:item*30);
    %NormalizedWindow3 = normalize(window3);
    [xcf,lags] = crosscorr(NormalizedWindow1,NormalizedWindow2,NumLags=29);
    %[RHO,PVAL] = corr(NEWHR(1+(7680*(item-1)):item*7680)',PSDTimeSeries(1+(7680*(item-1)):item*7680)','Type','Spearman');
    CorrStorage(1, :) = lags;
    CorrStorage(item+1, :) = xcf;
end

for item = 2:(4980*5)/30 + 1                                                %% Things to be changed based on subjects
    [Maxval, idx1] = max(CorrStorage(item, :));
    [minval, idx2] = min(CorrStorage(item, :));
    if abs(Maxval) > abs(minval)
        TimeDelay(item-1, 1) = CorrStorage(1, idx1);
    elseif abs(Maxval) < abs(minval)
        TimeDelay(item-1, 1) = CorrStorage(1, idx2);
    end
end
%[MaxCorr, TimeDelay1] = max(CorrStorage,[],2);

tcor = 0:30:4980*5-30;                                                      %% Things to be changed based on subjects
%figure
subplot(3,1,3)
plot(tcor, TimeDelay, 'o-b');
xlabel('time (s)','FontSize',12)
ylabel('Maximum Time Delay','FontSize',12)
%figure
% plot(freq, 10*log10(PSDEEG), 'b')
% xlabel('frequency (Hz)','FontSize',20)
% ylabel('PSD of ECG in logaritmic order','FontSize',20)
%figure
%plot(freq, abs(ECGdft), 'b')
% figure
% plot(lags, xcf)
% xlabel('lags','FontSize',20)
% ylabel('cross correlation','FontSize',20)
%% Calculating %TDS
stable = 0;
for item = 1:length(TimeDelay)
    if item > 3
        if abs(TimeDelay(item) - TimeDelay(item-1)) < 3 && (abs(TimeDelay(item) - TimeDelay(item-2)) < 3) && ...
                (abs(TimeDelay(item) - TimeDelay(item-3)) < 3) && (abs(TimeDelay(item-1) - TimeDelay(item-2)) < 3) &&...
                (abs(TimeDelay(item-1) - TimeDelay(item-3)) < 3) && (abs(TimeDelay(item-2) - TimeDelay(item-3)) < 3)
            stable  = stable + 1;
        end
    end
end
PercentTDS = (stable/length(TimeDelay))*100;

