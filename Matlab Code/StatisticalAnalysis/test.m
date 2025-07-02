clc
clear
close all

% %% Histogram
% load SImStorage.mat
% load SIpstorage.mat
% load SImbStorage.mat
% load SIpbStorage.mat
% figure
% labels = {'Delta','Theta', 'Alpha', 'Sigma', 'Beta'};
% for hist = 1:5
%     subplot(5, 1, hist)
%     h = histogram(SImbStorage(hist, :));
%     title(labels(hist))
% end
% xlabel('Magnitude of SI')
% %subtitle('Magnitude of SI for actual data')
% figure
% labels = {'Delta','Theta', 'Alpha', 'Sigma', 'Beta'};
% for hist = 1:5
%     subplot(5, 1, hist)
%     h = histogram(SIpbStorage(hist, :));
%     title(labels(hist))
% end
% xlabel('Phase of SI')
% %subtitle('Phase of SI for actual data')
%% Read and Edit text file
TextfileName = "C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Depression\RET_0002.txt";
T = readtable(TextfileName,'VariableNamingRule','preserve', 'ReadRowNames',true);
%T(5,:) = [];
Removingrows = T.Event;
ValidElements = {'Wake','NREM 1','NREM 2','NREM 3','REM'};
list = ismember(Removingrows, ValidElements);
T(~list, :) = [];
T.Duration(:) = 30;
% n = T.Properties.RowNames;
% count = 0;
% for item = 1:length(n)
%     if contains(n(item), '_')
%         count = count + 1;
%         list(:, count) = item;
%     end
% end
% T(list, :) = [];
% % nan_index = isnan(T.Duration);
% T.Duration = fillmissing(T.Duration,'constant',30);
writetable(T, 'RET_0002.txt', 'Delimiter','\t')
StartTime = timeofday(T.("Start Time")(1));
EndTime = timeofday(T.("Start Time")(end));
Filelocation = "C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Depression\RET_0002.edf";
[data, annotations] = edfread(Filelocation,'DataRecordOutputType','timetable','TimeOutputType','datetime');
info = edfinfo(Filelocation); %Dataset infromation
% ECG1 = data.EKG1; ECG2 = data.EKG2; F3 = data.F3; M1 = data.M1;
% M2 = data.M2; C3 = data.C3; O1 = data.O1;

Timedata = data.("Record Time");
timeOnly = timeofday(data.("Record Time"));
Startindx = find(timeOnly == StartTime);
Endindx = find(timeOnly == EndTime);
data(Endindx:end, :) = [];
data(1:Startindx-1, :) = [];
% structheader = table2struct(data,"ToScalar",true);
% edfwrite('RET_0002.edf', structheader, data);
%[rawdata] = ReadDataset(Filelocation);
%% Upsampling the signal without distortion
% % Load the original signal
% load Res.mat
% RRI = Res.RRs;
% HR = 60./RRI;
% Fs = length(HR)/24900;
% % Define the target integer sampling rate
% Fs_target = 256;
% 
% % Calculate the upsampling factor
% upsampling_factor = Fs_target / Fs;
% 
% % Define the interpolation filter
% interp_filter = fir1(63, 1/upsampling_factor);
% 
% % Upsample the signal using the interpolation filter
% y_upsampled = upsample(HR, upsampling_factor);
% y_upsampled_filtered = filter(interp_filter, 1, y_upsampled);
% 
% % Resample the upsampled signal at the target sampling rate
% y_resampled = resample(y_upsampled_filtered, Fs_target, Fs);
% 
% % Plot the original and upsampled signals
% t_orig = (0:length(HR)-1) / Fs;
% t_resampled = (0:length(y_resampled)-1) / Fs_target;
% figure;
% subplot(2,1,1);
% plot(t_orig, HR);
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Original Signal');
% subplot(2,1,2);
% plot(t_resampled, y_resampled);
% xlabel('Time (s)');
% ylabel('Amplitude');
% title('Upsampled Signal');
%%
% load F3.mat
% load M1.mat
% load M2.mat
% load Adjust_num.mat
% load HR.mat
% load newy.mat
% load ECG1.mat
%
% fs = 256;
% T = 0:1/fs:(4610*6)-1/fs; % 4980*5 is the sleep time in second                 %% Things to be changed based on subjects
%
% %% Refrencing ECG & EEG
% EEG = F3 - ((M1 + M2)/2);
% SleepEEG = EEG(Adjust_num*fs*30:end-(12*fs)-1);                                   %% Things to be changed based on subjects
% ECG = ECG1 - mean(ECG1);
% SleepECG = ECG(Adjust_num*fs*30:end-(12*fs)-1);                             %% Things to be changed based on subjects
% newy = newy(Adjust_num + 1:end);
%
% for item=1:length(newy)-1
%     [psd_EEG,f_EEG] = pwelch(SleepEEG(1+30*256*(item-1):(item*30*256)),hamming(4500),2500,[],fs);
%     %[psd_HR,f_HR] = pwelch(HR(256*100:(256*100)+(30*256)-1),hamming(5120),2560,[],fs);
%     builtinphase = angle(hilbert(psd_EEG));
%     %builtinphase2 = angle(hilbert(psd_HR));
%     psd_storage(:, item) = psd_EEG;
%     phase_storage(:, item) = builtinphase;
% end
% Average_all = mean(psd_storage, 2);
% Average_Phase = mean(phase_storage, 2);
%
% figure
% plot(f_EEG, 10*log10(Average_all), 'b')
% hold on
% %
% for item = 1:length(newy)-1
%     N = length(SleepEEG((1+30*256*(item-1):(item*30*256))));
%     EEGdft = fft(SleepEEG((1+30*256*(item-1):(item*30*256))));
%     EEGdft = EEGdft(1:(N/2)+1);
%     PSDEEG = (1/(fs*N)) * abs(EEGdft).^2;
%     PSDEEG(2:end - 1) = 2 * PSDEEG(2:end - 1);
%     h = hilbert(PSDEEG);
%     HandPhase = angle(h);
%     psd_storage_hand(:, item) = PSDEEG;
%     phase_storage_hand(:, item) = HandPhase;
% end
% Average_all_hand = mean(psd_storage_hand, 2);
% Average_Phase_hand = mean(phase_storage_hand, 2);
%
% freq = 0:fs/N:fs/2;
% plot(freq, 10*log10(Average_all_hand), 'r')
% legend('EEG PSD Magnitude builtin', 'EEG PSD Magnitude from scratch')
% figure
% plot(f_EEG, Average_Phase, freq, Average_Phase_hand)
% legend('EEG PSD Phase builtin', 'EEG PSD Phase from scratch')
%
%
% %% CPSD Calculation
% for item = 1:length(newy)-1
%     [pxy,f_0] = cpsd(SleepEEG((1+30*256*(item-1):(item*30*256))), ...
%         SleepECG((1+30*256*(item-1):(item*30*256))),hamming(5120),2560,[],fs);
%     cpsd_storage(:, item) = pxy;
%     cpsd_phase_storage(:, item) = angle(pxy);
% end
% cpsd_average = mean(cpsd_storage, 2);
% cpsd_phase_average = mean(cpsd_phase_storage, 2);
% % figure
% % plot(f_0, abs(cpsd_average))
%
% %% CPSD and coherence from the scratch
% for item = 1:length(newy)-1
%     [xcf,lags] = crosscorr(SleepEEG((1+30*256*(item-1):(item*30*256))), ...
%         SleepECG((1+30*256*(item-1):(item*30*256))),NumLags=7679);
%     CPSDHand = fft(xcf, 8194);
%     N1 = length(CPSDHand);
%     freq1 = 0:fs/N1:fs/2-fs/N1;
%     CPSDHand = CPSDHand(1:(N1/2));
%     CPSDHandPhase_storage(:, item) = angle(CPSDHand);
%     %cpsd_storage_hand(:, item) = CPSDHand;
%     [psd_EEG,~] = pwelch(SleepEEG(1+30*256*(item-1):(item*30*256)), ...
%         hamming(5120),2560,[],fs);
%     [psd_ECG,frequency] = pwelch(SleepECG(1+30*256*(item-1):(item*30*256)), ...
%         hamming(5120),2560,[],fs);
%     Coherence = ((abs(CPSDHand).^2)./(abs(hilbert(psd_ECG)).*abs(hilbert(psd_EEG)))).^0.5;
%     coherence_storage_hand(:, item) = Coherence;
% end
% % average_cpsd = mean(cpsd_storage_hand, 2);
% % hold on
% % plot(freq1, 40*abs(average_cpsd))
% % legend('builtin', 'hand')
% average_CPSDHandPhase = mean(CPSDHandPhase_storage, 2);
% coherence_average_hand = mean(coherence_storage_hand, 2);
% figure
% plot(frequency, coherence_average_hand);
% legend('Magnitude of Coherence from scratch')
% %hold on
% % figure
% % cpsdphase = angle(pxy);
% % plot(f_0, cpsdphase)
% % hold on
% % CPSDHandphase = angle(CPSDHand);
% % plot(freq1, CPSDHandphase)
%
% %% Coherence Calculation
% for item = 1:length(newy)-1
%     [cxy,f_1] = mscohere(SleepEEG((1+30*256*(item-1):(item*30*256))), ...
%         SleepECG((1+30*256*(item-1):(item*30*256))),hamming(5120),2560,[],fs);
%     coherence_storage(:, item) = cxy;
% end
% coherence_average = mean(coherence_storage, 2);
% figure
% plot(f_1, coherence_average)
% legend('Magnitude of Coherence builtin')
%
% figure
% plot(frequency, average_CPSDHandPhase, frequency, cpsd_phase_average)
% legend('Phase Coherence from scratch', 'Phase Coherence builtin')
% Denum = (hilbert(psd_HR).*hilbert(psd_EEG));
% num = abs(pxy_REM).^2;
% Coherence_REM = num./Denum;
% [Cxy_REM,f_REM] = mscohere(HR(256*100:(256*100)+(30*256)-1),SleepEEG(256*100:(256*100)+(30*256)-1),hamming(5120),2560,[],fs);

%rawData = ReadEcgInputFile("D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\M-Hassan\Control\RET_1505_RD (23102013).edf");
% Filelocation = "D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\M-Hassan\Depression\RET_0002.edf";
% [rawdata] = ReadDataset(Filelocation);

%% Hilbert Transform from scratch
% % Calculate Hilbert Transform to find the power time series for each window
% N = length(window);
% Windowfft = fft(window);
% complexf = 1i*Windowfft;
% %Find the positive and negative indices
% posf = 2:floor(N/2) + mod(N,2);
% negf = ceil(N/2) + 1 + ~mod(N,2):N;
% %Rotate the Fourier coeficients
% Windowfft(posf) = Windowfft(posf) + -1i*complexf(posf);
% Windowfft(negf) = Windowfft(negf) + 1i*complexf(negf);
% Hilbertfun = ifft(Windowfft);
% WindowPower = abs(Hilbertfun).^2;
% %plot(WindowPower)
% %figure