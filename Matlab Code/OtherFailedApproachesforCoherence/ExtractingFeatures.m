clc
clear
close all

load ECG1.mat
load ECG2.mat
load F3.mat
load M1.mat
load M2.mat
load newy.mat
load C3.mat
load O1.mat
load Adjust_num.mat

fs = 256;
T = 0:1/fs:(3525*6)-1/fs; % 4986*5 is the sleep time in second - The begining and the end of the signal should be synced!
% 21150/6 = 3525
%% Refrencing ECG & EEG
%ECG = ECG1 - ECG2;
F3 = F3 - ((M1 + M2)/2);
C3 = C3 - ((M1 + M2)/2);
O1 = O1 - ((M1 + M2)/2);
ECG = ECG1 - mean(ECG1);
SleepF3 = F3(Adjust_num*fs*30:end-(102*fs)-1); % 102 second from end should be removed
SleepC3 = C3(Adjust_num*fs*30:end-(102*fs)-1);
SleepO1 = O1(Adjust_num*fs*30:end-(102*fs)-1);
SleepLengthSig = length(SleepF3);
SleepECG = ECG1(Adjust_num*fs*30:end-(102*fs)-1);
%% Filtering EEG
Filter1 = designfilt('bandpassiir', 'FilterOrder', 6,...
    'HalfPowerFrequency1', 0.4,'HalfPowerFrequency2', 30,...
    'SampleRate', fs);
FilteredF3 = filtfilt(Filter1, SleepF3);
FilteredC3 = filtfilt(Filter1, SleepC3);
FilteredO1 = filtfilt(Filter1, SleepO1);
%% Low pass Filtering and Detrened ECG
Filter2 = designfilt('lowpassiir', 'FilterOrder', 3,...
    'PassbandFrequency', 50, 'SampleRate', fs);
FilteredECG = filtfilt(Filter2, SleepECG);
ECGDetrended  = detrend(FilteredECG, 5);
% figure
% plot(T, ECGDetrended, 'r')
% xlabel('t (s)','FontSize',20)
%ylabel('Referenced ECG','FontSize',20)
%legend('Filtered ECG', ' Detrended ECG')
% hold on
%t = 0:30:24930-30;
%newy = newy(68:end);
% plot(t, newy*100, 'k')
% legend('Detrended ECG', ' Sleep Stages')

% Computing HR from ECG
[qrs_amp_raw, qrs_i_raw, delay] = pan_tompkin(ECGDetrended, fs, 0);
t = T(qrs_i_raw);
for item = 1: length(qrs_i_raw)-1
    RRI(item) = T(qrs_i_raw(item+1)) - T(qrs_i_raw(item));
end
fsRRI = length(RRI)/(SleepLengthSig/fs);
% figure
% plot(t(1:end-1), RRI)
DownsampleRRI = resample(RRI, (SleepLengthSig/fs), length(RRI));
RRI_resampled = resample(DownsampleRRI, fs, 1);
HR = 60./RRI_resampled;
HRTDS = 60./DownsampleRRI;% resampled HR without upsampling for TDS

%% CPSD Method
m = 0;
n = 0;
for item = 1: length(newy)
    if newy(item) == 0
        n = n + 1;
        N1Idx(1, n) = item;
    end
end
newy(N1Idx) = [];
%% Finding Sleep Stages
N2 = [];
N3 = [];
REM = [];
Wake = [];
i = 0;
j = 0;
k = 0;
l = 0;
for item = 1:length(newy) %(24930/30) number of sleep epochs
    if newy(item) == 2
        i = i+1;
        N2(1, i) = (item - 1)*30;
    elseif newy(item) == 3
        j = j+1;
        N3(1, j) = (item - 1)*30;
    elseif newy(item) == 4
        k = k+1;
        REM(1, k) = (item - 1)*30;
    elseif newy(item) == 5
        l = l+1;
        Wake(1, l) = (item - 1)*30;
    end
end

Features = zeros(678,31); % 678 is the number of sleep stages without N1
n = 0;
for item = 1: length(newy)
    if newy(item) == 1
        n = n + 1;
        N1Idx2(1, n) = item;
    else
        m = m + 1;
        [CxyF,f] = mscohere(HR(1 + 256*30*(item-1):256*30*item),FilteredF3(1 + 256*30*(item-1):256*30*item),[],[],[],fs);
        C_StorageF(:, m) = CxyF;
        [CxyC,f] = mscohere(HR(1 + 256*30*(item-1):256*30*item),FilteredC3(1 + 256*30*(item-1):256*30*item),[],[],[],fs);
        C_StorageC(:, m) = CxyC;
        [CxyO,f] = mscohere(HR(1 + 256*30*(item-1):256*30*item),FilteredO1(1 + 256*30*(item-1):256*30*item),[],[],[],fs);
        C_StorageO(:, m) = CxyO;
        Features(m, 1) = newy(item);
    end
end
%plot(f, C_StorageF(:, 1))
lengthfeatures = length(Features);
%% Max coherence of freq bands
Features(:, 2:11) = CPSDfeatures(C_StorageF, lengthfeatures, f);
Features(:, 12:21) = CPSDfeatures(C_StorageC, lengthfeatures, f);
Features(:, 22:31) = CPSDfeatures(C_StorageO, lengthfeatures, f);

bounderies = [0.5, 4, 8, 12, 16, 3.5, 7.5, 11.5, 15.5, 19.5];
EEGChannels = [FilteredF3, FilteredC3, FilteredO1];
counter = 0;
cols = [32, 42, 52];
for item = EEGChannels
    counter = counter + 1;
    for iter = 1: length(bounderies)/2
        [maxcor, TimeDelay, NormalizedWindow1, NormalizedWindow2, xcf, lags] = TDSFeature(bounderies(iter), bounderies(iter+5), EEGChannels, ECGDetrended, fs, T, N1Idx2);
        maxcorrstorage(:, iter) = maxcor;
        TimeDelayStorage(:, iter) = TimeDelay;
    end
    Features(:, cols(counter): cols(counter) + 4) = maxcorrstorage;
    Features(:, cols(counter)+5: cols(counter)+9) = TimeDelayStorage;
end
headers = ["labels","MF_Delta","MF_Theta","MF_Alpha","MF_Sigma","MF_Beta", "MFf_Delta","MFf_Theta"...
    ,"MFf_Alpha","MFf_Sigma","MFf_Beta","MC_Delta","MC_Theta","MC_Alpha","MC_Sigma","MC_Beta"...
    ,"MCf_Delta","MCf_Theta","MCf_Alpha","MCf_Sigma","MCf_Beta","MO_Delta","MO_Theta","MO_Alpha"...
    ,"MO_Sigma","MO_Beta","MOf_Delta","MOf_Theta","MOf_Alpha","MOf_Sigma","MOf_Beta", "MCF_Delta"...
    ,"MCF_Theta","MCF_Alpha","MCF_Sigma","MCF_Beta","TDF_Delta","TDF_Theta","TDF_Alpha"...
    ,"TDF_Sigma","TDF_Beta","MCC_Delta","MCC_Theta","MCC_Alpha","MCC_Sigma","MCC_Beta","TDC_Delta"...
    ,"TDC_Theta","TDC_Alpha","TDc_Sigma","TDC_Beta","MCO_Delta","MCO_Theta","MCO_Alpha","MCO_Sigma"...
    ,"MCO_Beta","TDO_Delta","TDO_Theta","TDO_Alpha","TDO_Sigma","TDO_Beta"];
Features = array2table(Features, "VariableNames",headers);
writetable(Features,'Features.csv')
%% plotting
t1 = 1:30;
subplot(3,1,1);
plot(t1, NormalizedWindow1, 'b');
xlim([1 30])
xlabel('t (s)','FontSize',15)
ylabel('HR','FontSize',15)
subplot(3,1,2);
plot(t1, NormalizedWindow2,'r');
xlim([1 30])
xlabel('t (s)','FontSize',15)
ylabel('PS of Beta','FontSize',15)
subplot(3,1,3);
plot(lags, xcf, 'k');
xlim([-29 29])
xlabel('time lags','FontSize',15)
ylabel('CC in 30s epoch','FontSize',15)
%% TDS method
% Computing PSD of ... band
% Delta freq band
function[features] = CPSDfeatures(C_Storage, lengthfeatures, f)
for item = 1:lengthfeatures
    CXY = C_Storage(:, item);
    [maxval, idx] = max(CXY(4:28));
    [maxval1, idx1] = max(CXY(32:60));
    [maxval2, idx2] = max(CXY(64:92));
    [maxval3, idx3] = max(CXY(96:124));
    [maxval4, idx4] = max(CXY(128:156));
    features(item, :) = [maxval, maxval1, maxval2, maxval3, maxval4, f(idx+4-1), ...
        f(idx1+32-1), f(idx2+64-1), f(idx3+96-1), f(idx4+128-1)];
end
end

function [maxcor, TimeDelay, NormalizedWindow1, NormalizedWindow2, xcf, lags] = TDSFeature(lowerband, upperband, EEGChannels, ECGDetrended, fs, T, N1Idx2)
Filter2 = designfilt('bandpassiir', 'FilterOrder', 6,...
    'HalfPowerFrequency1', lowerband,'HalfPowerFrequency2', upperband,...
    'SampleRate', 256);
OneFreqBand = filtfilt(Filter2, EEGChannels);
for item = 1:(3525*6) % it should be changed based on each subject! (Total sleep time for analysis)
    window = OneFreqBand(1+((item-1)*fs):item*fs);
    pRMS1 = rms(window)^2;
    PSTimeSeries1(item) = pRMS1;
end
[qrs_amp_raw, qrs_i_raw, delay] = pan_tompkin(ECGDetrended, fs, 0);
t = T(qrs_i_raw);
for item = 1: length(qrs_i_raw)-1
    RRI(item) = T(qrs_i_raw(item+1)) - T(qrs_i_raw(item));
end
SleepLengthSig = length(EEGChannels);
fsRRI = length(RRI)/(SleepLengthSig/fs);
% figure
% plot(t(1:end-1), RRI)
DownsampleRRI = resample(RRI, (SleepLengthSig/fs), length(RRI));
HRTDS = 60./DownsampleRRI;% resampled HR without upsampling for TDS
for item = 1:(3525*6)/30 %(Total sleep time for each subject/30) number of sleep epochs
    window1 = HRTDS(1+(item-1)*30:item*30);
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

for item = 2:((3525*6)/30)+1 %(Total sleep time for each subject/30) number of sleep epochs
    [Maxval, idx1] = max(CorrStorage(item, :));
    [minval, idx2] = min(CorrStorage(item, :));
    if abs(Maxval) > abs(minval)
        TimeDelay(item-1, 1) = CorrStorage(1, idx1);
        maxcor(item-1, 1) = abs(Maxval);
    elseif abs(Maxval) < abs(minval)
        TimeDelay(item-1, 1) = CorrStorage(1, idx2);
        maxcor(item-1, 1) = abs(minval);
    end
end
maxcor(N1Idx2) = []; % remove where N1 stage correlation happend
TimeDelay(N1Idx2) = [];
end
%% Time Delay Method with 7 segments each  30 seconds
% for item = 1:831 %(24930/30) number of sleep epochs
%     window1 = HRTDS(1+(item-1)*30:item*30);
%     NormalizedWindow1 = normalize(window1);
%     window2 = PSTimeSeries1(1+(item-1)*30:item*30);
%     NormalizedWindow2 = normalize(window2);
%     for iter = 1:7
%         [xcf,lags] = crosscorr(NormalizedWindow1(1+(iter-1)*4: 6+(iter-1)*4),NormalizedWindow2(1+(iter-1)*4: 6+(iter-1)*4),NumLags=5);
%         %CorrStorage(1, :) = lags;
%         CorrStorage(iter, :) = xcf;
%     end
%     %[xcf,lags] = crosscorr(NormalizedWindow1,NormalizedWindow2,NumLags=29);
%     %[RHO,PVAL] = corr(NEWHR(1+(7680*(item-1)):item*7680)',PSDTimeSeries(1+(7680*(item-1)):item*7680)','Type','Spearman');
%     TotalCorrStorage(1+(item-1)*7:7+(item-1)*7, :) = CorrStorage;
% end
% TotalCorrStorage(end+1, :) = lags;
% for item = 1:length(TotalCorrStorage)-1
%     [Maxval, idx1] = max(TotalCorrStorage(item, :));
%     [minval, idx2] = min(TotalCorrStorage(item, :));
%     if abs(Maxval) > abs(minval)
%         TimeDelay(item, 1) = TotalCorrStorage(end, idx1);
%         maxcor(item, 1) = abs(Maxval);
%     elseif abs(Maxval) < abs(minval)
%         TimeDelay(item, 1) = TotalCorrStorage(end, idx2);
%         maxcor(item, 1) = abs(minval);
%     end
% end




