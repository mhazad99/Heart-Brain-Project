function [FilteredEEG] = EEGFiltering(EEG, fs, Adjust_num, PostWakeNum)
%% Deleting the pre-wake and post-wake parts of the signal to ignor the effect of motion in the final signal
SleepEEG = EEG(Adjust_num*fs*30:end-(PostWakeNum*fs)-1);
%% Passband filter for noise cancelation
Filter = designfilt('bandpassiir', 'FilterOrder', 6,...
    'HalfPowerFrequency1', 0.2,'HalfPowerFrequency2', 30,...
    'SampleRate', fs);
FilteredEEG = filtfilt(Filter, SleepEEG);

end