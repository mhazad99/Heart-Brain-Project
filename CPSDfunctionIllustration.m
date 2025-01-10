% Create a new figure
figure;
% First subplot (top)
subplot(3, 1, 1); % 3 rows, 1 column, 1st plot
x = linspace(0, 128, 1025);
plot(x, 10*log10(pwelch(ECG.ECGCleaned(1:30*256),[],[],[],256)));
title('ECG power spectral density');
xlabel('Frequency');
ylabel('...');
% Second subplot (middle)
subplot(3, 1, 2); % 3 rows, 1 column, 2nd plot
plot(x, 10*log10(pwelch(CleanedEEGinfo.CleanedC3(1:30*256*1),[],[],[],256)));
title('EEG power spectral density');
xlabel('Frequency');
ylabel('...');
% Third subplot (bottom)
subplot(3, 1, 3); % 3 rows, 1 column, 3rd plot
plot(x, 10*log10(cpsd(ECG.ECGCleaned(1:30*256),CleanedEEGinfo.CleanedC3(1:30*256*1),[],[],[],256)));
title('Cross power spectral density of ECG and EEG');
xlabel('Frequency');
ylabel('...');
% Adjusting the layout
sgtitle('Illustrating Cross power spectral density procedure');
cpsd(ECG.ECGCleaned(1:30*256), ...
CleanedEEGinfo.CleanedC3(1:30*256*1),[],[],[],256);