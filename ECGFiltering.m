function [NormalizedECG] = ECGFiltering(rawECG, fs)

%% Cancelation of DC part from ECG
ECG = rawECG - mean(rawECG);
%% Passband filter for noise cancelation
f1=1;    %%0.5? Rebecca vals:5                                             % cuttoff low frequency to get rid of baseline wander
f2=25;   %%20?  Rebecca vals:15                                            % cuttoff frequency to discard high frequency noise
Wn=[f1 f2]*2/fs;                                                           % cutt off based on fs
N = 3;                                                                     % order of 3 less processing
[a,b] = butter(N,Wn);                                                      % bandpass filtering
PassbandECG = filtfilt(a,b,ECG);
%% Normalization of the filtered signal
NormalizedECG = PassbandECG/max(abs(PassbandECG));

end