clc
clear
close all

load ECG1.mat
load ECG2.mat
load F3.mat
load M1.mat
load M2.mat
load newy.mat
load LAT1.mat
load RAT1.mat
load RAT2.mat

fs = 256;
T = 0:1/fs:(5393*5)-1/fs;

%% Refrencing ECG & EEG
%ECG = ECG1 - ECG2;
EEG = F3 - ((M1 + M2)/2);
ECG = ECG1 - mean(ECG1);

%% Filtering EEG
Filter1 = designfilt('bandpassiir', 'FilterOrder', 6,...
    'HalfPowerFrequency1', 0.4,'HalfPowerFrequency2', 30,...
    'SampleRate', fs);
FilteredEEG = filtfilt(Filter1, EEG);
%% PSD Analysis of EEG
for item = 1:floor((length(FilteredEEG) - 514560)/(fs*30))-5 % (length(EEG) - 514560 (times when the subject was not asleep))/256
    window = FilteredEEG(514560+((item-1)*fs*30):514560+(item*fs*30));
    for iter = 1:6
        window2 = window(1+(iter-1)*fs*5:iter*fs*5);
        n = length (window2);
        window2dft = fft(window2);
        window2dft = window2dft(1:(n/2)+1);
        PSDwindow2 = (1/(fs*n)) * abs(window2dft).^2;
        PSDwindow2(2:end - 1) = 2 * PSDwindow2(2:end - 1);
        PSDwindow2Storage(:, iter) = PSDwindow2;
        freqwindow = 0:fs/n:fs/2;
    end
    windowavg(:, item) = mean(PSDwindow2Storage, 2);
end
N1 = length(FilteredEEG);
EEGdft = fft(FilteredEEG);
EEGdft = EEGdft(1:(N1/2)+1);
PSDEEG = (1/(fs*N1)) * abs(EEGdft).^2;
PSDEEG(2:end - 1) = 2 * PSDEEG(2:end - 1);
freqEEG = 0:fs/N1:fs/2;
EEGPSOvernight = [trapz(freqEEG(find(freqEEG>0.5, 1):find(freqEEG>3.5, 1)), PSDEEG(find(freqEEG>0.5, 1):find(freqEEG>3.5, 1))),...
                  trapz(freqEEG(find(freqEEG>4, 1):find(freqEEG>7.5, 1)), PSDEEG(find(freqEEG>4, 1):find(freqEEG>7.5, 1)))...
                  trapz(freqEEG(find(freqEEG>8, 1):find(freqEEG>11.5, 1)), PSDEEG(find(freqEEG>8, 1):find(freqEEG>11.5, 1))),...
                  trapz(freqEEG(find(freqEEG>12, 1):find(freqEEG>15.5, 1)), PSDEEG(find(freqEEG>12, 1):find(freqEEG>15.5, 1)))...
                  trapz(freqEEG(find(freqEEG>16, 1):find(freqEEG>19.5, 1)), PSDEEG(find(freqEEG>16, 1):find(freqEEG>19.5, 1)))]; % [Delta,Theta,Alpha,Sigma,Beta] for normalization

for iter2 = 1:length(windowavg)
    WinowPSD = windowavg(:, iter2);
    NormalizedPSWindows(iter2, :) = [(trapz(freqwindow(find(freqwindow>0.5, 1):find(freqwindow>3.5, 1)), WinowPSD(find(freqwindow>0.5, 1):find(freqwindow>3.5, 1))))/EEGPSOvernight(1,1),...
                          (trapz(freqwindow(find(freqwindow>4, 1):find(freqwindow>7.5, 1)), WinowPSD(find(freqwindow>4, 1):find(freqwindow>7.5, 1))))/EEGPSOvernight(1,2)...
                          (trapz(freqwindow(find(freqwindow>8, 1):find(freqwindow>11.5, 1)), WinowPSD(find(freqwindow>8, 1):find(freqwindow>11.5, 1))))/EEGPSOvernight(1,3),...
                          (trapz(freqwindow(find(freqwindow>12, 1):find(freqwindow>15.5, 1)), WinowPSD(find(freqwindow>12, 1):find(freqwindow>15.5, 1))))/EEGPSOvernight(1,4)...
                          (trapz(freqwindow(find(freqwindow>16, 1):find(freqwindow>19.5, 1)), WinowPSD(find(freqwindow>16, 1):find(freqwindow>19.5, 1))))/EEGPSOvernight(1,5)];
end

% figure
% plot(freq, PSDEEG, 'b')
%% Low pass Filtering and Detrened ECG
Filter2 = designfilt('bandpassiir', 'FilterOrder', 4,...
    'HalfPowerFrequency1', 0.5,'HalfPowerFrequency2', 20,...
    'SampleRate', fs);
FilteredECG = filtfilt(Filter2, ECG1);
ECGDetrended  = detrend(FilteredECG, 5);
% figure
% plot(T, ECG1,'b', T, ECGDetrended, 'r')
% xlabel('t (s)','FontSize',20)
% %ylabel('Referenced ECG','FontSize',20)
% legend('ECGDetrended', ' ECG1')
% % hold on
% t = 0:30:26940-30;
% plot(t, newy*100, 'k')
% legend('Detrended ECG', ' Sleep Stages')

%% Computing HR from ECG
windowlength = 180;
for item = 1:floor((length(ECGDetrended)- 514560)/(30*fs))-5 % 514560 is the time when the subject starts to sleep (before this time are not sleep data!)
    [qrs_amp_raw, qrs_i_raw, delay] = pan_tompkin(ECGDetrended(514560+(30*fs*(item-1)):514560+(windowlength*fs)+(30*fs*(item-1))), fs, 0); % 2010th second is the start of the first wake stage: 2010*256
    t = T(qrs_i_raw);
    for item2 = 1: length(qrs_i_raw)-1
        RRI(item2) = T(qrs_i_raw(item2+1)) - T(qrs_i_raw(item2));
    end
    fsRRI = length(RRI)/windowlength;
    N = length (RRI);
    RRIdft = fft(RRI);
    RRIdft = RRIdft(1:(N/2)+1);
    PSDRRI = (1/(fsRRI*N)) * abs(RRIdft).^2;
    PSDRRI(2:end - 1) = 2 * PSDRRI(2:end - 1);
    freqRRI = 0:fsRRI/N:fsRRI/2;
    %figure
    %plot(freq, (PSDRRI), 'b')
    LFidx1 = find(freqRRI>0.035, 1);
    LFidx2 = find(freqRRI>0.14, 1);
    HFidx1 = find(freqRRI>=0.15, 1);
    HFidx2 = find(freqRRI>0.4, 1);
    LF = trapz(freqRRI(LFidx1:LFidx2), PSDRRI(LFidx1:LFidx2));
    HF = trapz(freqRRI(HFidx1:HFidx2), PSDRRI(HFidx1:HFidx2));
    LFStorage_nu(item) = LF/(HF+LF); %normalizing and storing LF: LF/HF+LF
    HFStorage_nu(item) = HF/(HF+LF); %normalizing and storing HF: HF/HF+LF
end
% T = 0:30:826*30-30;
% plot(T, HFStorage_nu, T, NormalizedPSWindows(:, 1))
% legend('HF_nu', 'DeltaPS')
fsnew = 1/30;
[Cxy, f] = mscohere(HFStorage_nu, NormalizedPSWindows(:, 1),[],[],1652,fsnew);
figure
plot(f, Cxy)
%cpsd(HFStorage_nu, NormalizedPSWindows(:, 1));

