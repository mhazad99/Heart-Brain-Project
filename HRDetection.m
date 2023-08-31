clc
clear
close all

load ECG1.mat
load ECG2.mat

load Adjust_num.mat

fs = 256;
T = 0:1/fs:(3525*6)-1/fs; % 4980*5 is the sleep time in second                 %% Things to be changed based on subjects

SleepECG = ECG1(Adjust_num*fs*30:end-(102*fs)-1);                           %% Things to be changed based on subjects
SleepLengthSig = length(SleepECG);

ECG = SleepECG - mean(SleepECG);
%% Low pass Filtering and Detrened ECG
Filter2 = designfilt('lowpassiir', 'FilterOrder', 3,...
    'PassbandFrequency', 25, 'SampleRate', fs);
FilteredECG = filtfilt(Filter2, ECG);
ECGDetrended  = detrend(FilteredECG, 6);
figure
plot(T, ECGDetrended)
xlabel('t (s)','FontSize',20)

[pks,locs] = findpeaks(ECGDetrended(1:30*fs),'MINPEAKDISTANCE',250);
hold on
plot(T(locs), pks,'r*')

for item = 1:length(locs)-1
    RRICalculated(item) = T(locs(item+1)) - T(locs(item));
end

tCalculated = T(locs);
figure
plot(tCalculated(1:end-1), 60./RRICalculated)
%hold on
%% RR interval detection
[qrs_amp_raw, qrs_i_raw, delay] = pan_tompkin(ECGDetrended, fs, 0);
t = T(qrs_i_raw);
for item = 1: length(qrs_i_raw)-1
    RRI(item) = T(qrs_i_raw(item+1)) - T(qrs_i_raw(item));
end
fsRRI = length(RRI)/(SleepLengthSig/fs);
%figure
%plot(t(1:end-1), 60./RRI)
%DownsampleRRI = resample(RRI, (SleepLengthSig/fs), length(RRI));
%RRI_resampled = resample(DownsampleRRI, fs, 1);
%HR = 60./RRI_resampled;
