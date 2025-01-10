clc
clear
close all

%% Load signals and Initials
% Specify the folder path where your .mat files are located
addpath('C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\EEG Artifact Removal Pipeline\Hassan_preprocessing\')
addpath('C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\EEG Artifact Removal Pipeline\Hassan_preprocessing\PREP_BS\')
addpath('C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\EEG Artifact Removal Pipeline\Hassan_preprocessing\PREP_BS_2\')
addpath('C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\EEG Artifact Removal Pipeline\Hassan_preprocessing\RL2BV_V5\')
%Filelocation = "C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Depression\RET_0002.edf";
%TextfileName = "C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Depression\RET_0002.txt"; %It can be changed to editted version I think

%folderpath = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Depression Cases'; % Depression cases
folderpath = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Insomniacs Cases'; % Control cases
% My laptop directory:: D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets\M-Hassan\Depression

txtfileList = dir(fullfile(folderpath, '*.txt'));
edffileList = dir(fullfile(folderpath, '*.edf'));
flag = 1;
for i = 1:numel(txtfileList) % !!BE CAREFUL ABOUT THE STARTING POINT OF THE LOOP!!
    txtfileName = txtfileList(i).name; % Get the file name
    txtfilePath = fullfile(folderpath, txtfileName); % Get the full file path
    edffileName = edffileList(i).name;
    edffilePath = fullfile(folderpath, edffileName);
    %%% adding the EEGArtifact removal pipeline
    %%% Comment it when you have done the artifact removal part!
    %PreProcess_Coppieters_2015_faster(string(txtfileName(1:8))) %%%Uncomment it when you want to export brainstorm results
%end %%% uncomment it when you want to export Brainsotrm results
    subfolderPath = ['C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Brainstorm\' ...
        'brainstorm_db']; % Update with your folder path
    folderPath = fullfile(subfolderPath, txtfileName(1:8),'\data\', txtfileName(1:8), 'NIGHT');
    [StartTime, EndTime, y, Stageidx, FinalLen] = Hypnogram(txtfilePath, folderPath);
    [CleanedEEGinfo] = EEGArtifactRemoval(folderPath, Stageidx, y, FinalLen, flag); % if flag == 1 it won't consider the sleep stages!
    [rawdata] = ReadDataset(edffilePath, StartTime, EndTime);
    fs = rawdata.fs;
    % Each index in CleanedEEGinfo.(.)artifactidx corresponds to 30 second of
    % the data starting from the first 30 second of the recorded signal!

    %%% Since the artifacts for F3, C3, and O1 are in different positions, you
    %%% should remove the noisy epochs for ECG corresponding to each EEG
    %%% channel. Hence, in this case you will have three different ECG that you
    %%% can analyse each of them with a specific EEG channel!

    %%%% filter ECG signal before continueing the procedure
    rawdata.ECG1 = ECGFiltering(rawdata.ECG1, fs);
    if flag % if flag == 1 it won't consider the sleep stages for ECG
        ECG.ECGCleaned = rawdata.ECG1;
        for item = numel(CleanedEEGinfo.MergedIndexes):-1:1
            idx = CleanedEEGinfo.MergedIndexes(item);
            ECG.ECGCleaned(1+fs*((idx-1)*30):fs*(idx*30)) = [];
        end
          %%% Remove F3 noisy epochs from ECG
%         ECG.ECGF3 = rawdata.ECG1;
%         for item = numel(CleanedEEGinfo.F3artifactidx):-1:1
%             idx = CleanedEEGinfo.F3artifactidx(item);
%             ECG.ECGF3(1+fs*((idx-1)*30):fs*(idx*30)) = [];
%         end
%         %%% Remove C3 noisy epochs from ECG
%         ECG.ECGC3 = rawdata.ECG1;
%         for item = numel(CleanedEEGinfo.C3artifactidx):-1:1
%             idx = CleanedEEGinfo.C3artifactidx(item);
%             ECG.ECGC3(1+fs*((idx-1)*30):fs*(idx*30)) = [];
%         end
%         %%% Remove O1 noisy epochs from ECG
%         ECG.ECGO1 = rawdata.ECG1;
%         for item = numel(CleanedEEGinfo.O1artifactidx):-1:1
%             idx = CleanedEEGinfo.O1artifactidx(item);
%             ECG.ECGO1(1+fs*((idx-1)*30):fs*(idx*30)) = [];
%         end
        folderName = txtfileName(1:8);
        newFolderPath = fullfile(folderpath, folderName);
        % Use the mkdir function to create the new folder
        mkdir(newFolderPath);
        file1Path = fullfile(newFolderPath, 'ECG.mat');
        file2Path = fullfile(newFolderPath, 'CleanedEEGinfo.mat');
        save(file1Path, 'ECG');
        save(file2Path, 'CleanedEEGinfo');
    else
        % REM F3
        %%% Remove F3 noisy epochs and sleep stages from ECG
        ECG.ECGF3REM = rawdata.ECG1;
        % Merge the lists and create a combined list of indexes
        combinedList1 = [CleanedEEGinfo.F3artifactidx', Stageidx.NREM3idx, ...
            Stageidx.NREM2idx, Stageidx.NREM1idx, Stageidx.Wake];
        % Remove duplicate indexes and sort the combined list
        REMF3Removeidx = unique(combinedList1);
        for item = numel(REMF3Removeidx):-1:1
            idx = REMF3Removeidx(item);
            ECG.ECGF3REM(1+fs*((idx-1)*30):fs*(idx*30)) = []; % remove artifacts
        end
        % NREM F3
        %%% Remove F3 noisy epochs and sleep stages from ECG
        ECG.ECGF3NREM = rawdata.ECG1;
        % Merge the lists and create a combined list of indexes
        combinedList2 = [CleanedEEGinfo.F3artifactidx', Stageidx.REMidx, ...
            Stageidx.Wake];
        % Remove duplicate indexes and sort the combined list
        NREMF3Removeidx = unique(combinedList2);
        for item = numel(NREMF3Removeidx):-1:1
            idx = NREMF3Removeidx(item);
            ECG.ECGF3NREM(1+fs*((idx-1)*30):fs*(idx*30)) = []; % remove artifacts and opposite sleep stages
        end
        % REM C3
        %%% Remove C3 noisy epochs and sleep stages from ECG
        ECG.ECGC3REM = rawdata.ECG1;
        % Merge the lists and create a combined list of indexes
        combinedList3 = [CleanedEEGinfo.C3artifactidx', Stageidx.NREM3idx, ...
            Stageidx.NREM2idx, Stageidx.NREM1idx, Stageidx.Wake];
        % Remove duplicate indexes and sort the combined list
        REMC3Removeidx = unique(combinedList3);
        for item = numel(REMC3Removeidx):-1:1
            idx = REMC3Removeidx(item);
            ECG.ECGC3REM(1+fs*((idx-1)*30):fs*(idx*30)) = []; % remove artifacts and opposite sleep stages
        end
        % NREM C3
        %%% Remove C3 noisy epochs and sleep stages from ECG
        ECG.ECGC3NREM = rawdata.ECG1;
        % Merge the lists and create a combined list of indexes
        combinedList4 = [CleanedEEGinfo.C3artifactidx', Stageidx.REMidx, ...
            Stageidx.Wake];
        % Remove duplicate indexes and sort the combined list
        NREMC3Removeidx = unique(combinedList4);
        for item = numel(NREMC3Removeidx):-1:1
            idx = NREMC3Removeidx(item);
            ECG.ECGC3NREM(1+fs*((idx-1)*30):fs*(idx*30)) = []; % remove artifacts and opposite sleep stages
        end
        % REM O1
        %%% Remove O1 noisy epochs and sleep stages from ECG
        ECG.ECGO1REM = rawdata.ECG1;
        % Merge the lists and create a combined list of indexes
        combinedList5 = [CleanedEEGinfo.O1artifactidx', Stageidx.NREM3idx, ...
            Stageidx.NREM2idx, Stageidx.NREM1idx, Stageidx.Wake];
        % Remove duplicate indexes and sort the combined list
        REMO1Removeidx = unique(combinedList5);
        for item = numel(REMO1Removeidx):-1:1
            idx = REMO1Removeidx(item);
            ECG.ECGO1REM(1+fs*((idx-1)*30):fs*(idx*30)) = []; % remove artifacts and opposite sleep stages
        end
        % NREM O1
        %%% Remove O1 noisy epochs and sleep stages from ECG
        ECG.ECGO1NREM = rawdata.ECG1;
        % Merge the lists and create a combined list of indexes
        combinedList6 = [CleanedEEGinfo.O1artifactidx', Stageidx.REMidx, ...
            Stageidx.Wake];
        % Remove duplicate indexes and sort the combined list
        NREMO1Removeidx = unique(combinedList6);
        for item = numel(NREMO1Removeidx):-1:1
            idx = NREMO1Removeidx(item);
            ECG.ECGO1NREM(1+fs*((idx-1)*30):fs*(idx*30)) = []; % remove artifacts and opposite sleep stages
        end
        folderName = txtfileName(1:8);
        stagesfolderpath = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Insomniacs Cases'; %%% Change this directory
        % for Depression and Control cases!
        newFolderPath = fullfile(stagesfolderpath, folderName);
        % Use the mkdir function to create the new folder
        mkdir(newFolderPath);
        file1Path = fullfile(newFolderPath, 'ECG.mat');
        file2Path = fullfile(newFolderPath, 'CleanedEEGinfo.mat');
        save(file1Path, 'ECG');
        save(file2Path, 'CleanedEEGinfo');
        clear ECG CleanedEEGinfo;
    end
end

%% Things need to be changed based on subjects
PostWakeNum = 25;
%% Refrencing ECG & EEG
%ECG = ECG1 - ECG2; %This referencing does not good results
EEG = rawdata.F3 - ((rawdata.M1 + rawdata.M2)/2);
% SleepEEG and SleepECG are the signals with removed pre-post wake signals
%% EEG filtering and artifact removal
FilteredEEG = EEGFiltering(EEG, fs, Adjust_num, PostWakeNum);
%% ECG filtering and artifact removal
FilteredECG = ECGFiltering(rawdata.ECG1, fs, Adjust_num, PostWakeNum);
%% Things need to be changed based on subjects
T = 0:1/fs:(length(FilteredEEG)/fs)-1/fs;
figure
plot(T, FilteredECG, 'b')
%xlabel('t (s)','FontSize',20)
%ylabel('Referenced ECG','FontSize',20)
%legend('Filtered ECG', ' Detrended ECG')
% hold on
%% Read Cleaned EEG and ECG data from directory
rawfilefolderpath = ['D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets' ...
    '\M-Hassan\Control Cases']; % Control cases
% rawfilefolderpath = ['D:\MASc @ ETS\OneDrive - ETS\Thesis\Datasets' ...
%     '\M-Hassan\Depression Cases']; % Depression cases
%fileList = dir(fullfile(rawfilefolderpath));
readNestedFolders(rawfilefolderpath)


%% To plot the Hypnogram uncomment the following lines
% t = 0:30:4722*5;          %% Things to be changd based on subjects
newy = newy(Adjust_num + 1:end);
% plot(t, newy*100, 'k')
% legend('ECG Detrended', ' Sleep Stages')

%% QRS Detection
%qrsPreProcessing = QrsPreProcessingPT(fs, T, ECG1(Adjust_num*fs*30:end-(130*fs)-1), newy, Adjust_num);
%qrsDetection = QrsDetectionPT(fs, qrsPreProcessing, newy, Adjust_num);

%% Computing HR from ECG with pan tompkin algorithm (Should be replaced with the python algorithms)
% [qrs_amp_raw, qrs_i_raw, delay] = pan_tompkin(ECGwithoutArtifact, fs, 0);
% t = T(qrs_i_raw);
% for item = 1: length(qrs_i_raw)-1
%     RRI(item) = T(qrs_i_raw(item+1)) - T(qrs_i_raw(item));
% end
% fsRRI = length(RRI)/(SleepLengthSig/fs);
% % figure
% % plot(t(1:end-1), RRI)
% DownsampleRRI = resample(RRI, (SleepLengthSig/fs), length(RRI));  % Check it again!
% RRI_resampled = resample(DownsampleRRI, fs, 1);
% HR = 60./RRI_resampled;
% figure
% plot(T, HR)

%% Finding Sleep Stages
i = 0;
j = 0;
k = 0;
l = 0;
for item = 1:length(newy) - 1 %(24930/30) number of sleep epochs
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

%% CPSD Method
for item = 1: length(REM)
    %[psd_HR,f_HR] = pwelch(HR(fs*REM(item):fs*(REM(item))+(30*fs)),hamming(5120),2560,[],fs);
    %[psd_EEG,f_EEG] = pwelch(SleepEEG(fs*REM(item):fs*(REM(item))+(30*fs)),hamming(5120),2560,[],fs);
    [pxy_REM,f_REM0] = cpsd(FilteredECG(fs*REM(item):fs*(REM(item))+(30*fs)), ...
        FilteredEEG(fs*REM(item):fs*(REM(item))+(30*fs)),hamming(5120),2560,[],fs);
    %Coherence_REM = (abs(pxy_REM).^2)./(hilbert(psd_HR).*hilbert(psd_EEG));
    [Cxy_REM,f_REM] = mscohere(FilteredECG(fs*REM(item):fs*(REM(item))+(30*fs)), ...
        FilteredEEG(fs*REM(item):fs*(REM(item))+(30*fs)),hamming(5120),2560,[],fs);
    C_REM_Storage(:, item) = Cxy_REM;
    CoherencePhaseREM(:, item) = angle(pxy_REM);
end
for item = 1: length(N2)
    [Cxy_N2,f_N2] = mscohere(FilteredECG(fs*N2(item):fs*(N2(item))+(30*fs)), ...
        FilteredEEG(fs*N2(item):fs*(N2(item))+(30*fs)),hamming(5120),2560,[],fs);
    [pxy_N2,f_N20] = cpsd(FilteredECG(fs*N2(item):fs*(N2(item))+(30*fs)), ...
        FilteredEEG(fs*N2(item):fs*(N2(item))+(30*fs)),hamming(5120),2560,[],fs);
    C_N2_Storage(:, item) = Cxy_N2;
    CoherencePhaseN2(:, item) = angle(pxy_N2);
end
for item = 1: length(N3)
    [Cxy_N3,f_N3] = mscohere(FilteredECG(fs*N3(item):fs*(N3(item))+(30*fs)), ...
        FilteredEEG(fs*N3(item):fs*(N3(item))+(30*fs)),hamming(5120),2560,[],fs);
    [pxy_N3,f_N30] = cpsd(FilteredECG(fs*N3(item):fs*(N3(item))+(30*fs)), ...
        FilteredEEG(fs*N3(item):fs*(N3(item))+(30*fs)),hamming(5120),2560,[],fs);
    C_N3_Storage(:, item) = Cxy_N3;
    CoherencePhaseN3(:, item) = angle(pxy_N3);
end
for item = 2: length(Wake)
    [Cxy_Wake,f_Wake] = mscohere(FilteredECG(fs*Wake(item):fs*(Wake(item))+(30*fs)), ...
        FilteredEEG(fs*Wake(item):fs*(Wake(item))+(30*fs)),hamming(5120),2560,[],fs);
    [pxy_Wake,f_Wake0] = cpsd(FilteredECG(fs*Wake(item):fs*(Wake(item))+(30*fs)), ...
        FilteredEEG(fs*Wake(item):fs*(Wake(item))+(30*fs)),hamming(5120),2560,[],fs);
    C_Wake_Storage(:, item) = Cxy_Wake;
    CoherencePhaseWake(:, item) = angle(pxy_Wake);
end
for item = 1:length(newy)-1 % just consider you should change the newy info in the Hypnogram file to make it work!
    [Cxy_TOT,f_TOT] = mscohere(FilteredECG(1+30*fs*(item-1):30*fs*item), ...
        FilteredEEG(1+30*fs*(item-1):30*fs*item),hamming(5120),2560,[],fs);
    [pxy_TOT,f_TOT] = cpsd(FilteredECG(1+30*fs*(item-1):30*fs*item), ...
        FilteredEEG(1+30*fs*(item-1):30*fs*item),hamming(5120),2560,[],fs);
    C_overall(:, item) = Cxy_TOT;
    CoherencePhase(:, item) = angle(pxy_TOT);
end

Average_REM = mean(C_REM_Storage, 2); % average between all Cxy
Average_N3 = mean(C_N3_Storage, 2);
Average_N2 = mean(C_N2_Storage, 2);
Average_Wake = mean(C_Wake_Storage, 2);
Average_Coverall = mean(C_overall, 2);
%phase calculation
Average_PhaseREM = mean(CoherencePhaseREM, 2);
Average_PhaseN2 = mean(CoherencePhaseN2, 2);
Average_PhaseN3 = mean(CoherencePhaseN3, 2);
Average_PhaseWake = mean(CoherencePhaseWake, 2);
Average_CoherencePhase = mean(CoherencePhase, 2);

%[Cxy,f_REM] = mscohere(HR,SleepEEG,hamming(5120),2560,[],fs);

figure
plot(f_REM, Average_REM, 'b-', LineWidth=2)
hold on
plot(f_N3, Average_N3, 'r.-', LineWidth=2)
hold on
plot(f_N2, Average_N2, 'k', LineWidth=1.5)
%hold on
%plot(f_Wake, Average_Wake,'-', LineWidth=1.5)
hold on
xline(0.5, '--');xline(3.5, '--')
xline(4, '--');xline(7.5, '--')
xline(8, '--');xline(11.5, '--')
xline(12, '--');xline(15.5, '--')
xline(16, '--');xline(19.5, '--')
legend('REM','N3', 'N2', 'FontSize',20)
xlabel('frequency (Hz)','FontSize',20)
ylabel('Coherence','FontSize',20)

figure
plot(f_TOT, Average_Coverall, 'b')
xlabel('frequency (Hz)','FontSize',20)
ylabel('Magnitude of Coherence','FontSize',20)
hold on
xline(0.5, '--');xline(3.5, '--');xline(4, '--');xline(7.5, '--')
xline(8, '--');xline(11.5, '--');xline(12, '--');xline(15.5, '--')
xline(16, '--');xline(19.5, '--')
figure
plot(f_TOT, Average_CoherencePhase, 'r')
xlabel('frequency (Hz)','FontSize',20)
ylabel('Phase of Coherence','FontSize',20)
hold on
xline(0.5, '--');xline(3.5, '--');xline(4, '--');xline(7.5, '--')
xline(8, '--');xline(11.5, '--');xline(12, '--');xline(15.5, '--')
xline(16, '--');xline(19.5, '--')

%% MEAN and STD of freq bands
% Mean_Delta_N3 = mean(Average_N3(4:28));
% std_Delta_N3 = std(Average_N3(4:28));
% Mean_Theta_N3 = mean(Average_N3(32:60));
% std_Theta_N3 = std(Average_N3(32:60));
% Mean_alpha_N3 = mean(Average_N3(64:92));
% std_alpha_N3 = std(Average_N3(64:92));
% Mean_sigma_N3 = mean(Average_N3(96:124));
% std_sigma_N3 = std(Average_N3(96:124));
% Mean_beta_N3 = mean(Average_N3(128:156));
% std_beta_N3 = std(Average_N3(128:156));

%% Bootstrapping to make sure the results are reliable
l = length(N2) + length(N3) + length(REM) + length(Wake);
BootstrapArray1 = randsample([N2, N3, REM, Wake(2:end)], l-1);
BootstrapArray2 = randsample([N2, N3, REM, Wake(2:end)], l-1);
for item = 1: l-1
    [CxyB,fB] = mscohere(FilteredECG(fs*BootstrapArray1(item):fs*(BootstrapArray1(item)) ...
        +(30*fs)),FilteredEEG(fs*BootstrapArray2(item):fs*(BootstrapArray2(item))+(30*fs)),hamming(5120),2560,[],fs);
    CxyBStorage(:, item) = CxyB;
end
%figure
% hold on
% CxyBAverage = mean(CxyBStorage, 2);
% plot(fB, CxyBAverage)
% legend('Identical epochs', 'Bootstrapped epochs','FontSize',20)
% xlabel('frequency (Hz)','FontSize',20)
% ylabel('Coherence','FontSize',20)

% figure
% plot(f_TOT, Cxy_TOT, fB, CxyB)

lN2 = length(N2);
BAN21 = randsample(N2, lN2);
BAN22 = randsample(N2, lN2);
for item = 1: length(N2)
    [Cxy_N2B,f_N2B] = mscohere(FilteredECG(fs*BAN21(item):fs*(BAN21(item))+(30*fs)), ...
        FilteredEEG(fs*BAN22(item):fs*(BAN22(item))+(30*fs)),hamming(5120),2560,[],fs);
    [pxy_N2B,f_N20B] = cpsd(FilteredECG(fs*BAN21(item):fs*(BAN21(item))+(30*fs)), ...
        FilteredEEG(fs*BAN22(item):fs*(BAN22(item))+(30*fs)),hamming(5120),2560,[],fs);
    C_N2B_Storage(:, item) = Cxy_N2B;
    CoherencePhaseN2B(:, item) = angle(pxy_N2B);
end
lN3 = length(N3);
BAN31 = randsample(N3, lN3);
BAN32 = randsample(N3, lN3);
for item = 1: length(N3)
    [Cxy_N3B,f_N3B] = mscohere(FilteredECG(fs*BAN31(item):fs*(BAN31(item))+(30*fs)), ...
        FilteredEEG(fs*BAN32(item):fs*(BAN32(item))+(30*fs)),hamming(5120),2560,[],fs);
    [pxy_N3B,f_N30B] = cpsd(FilteredECG(fs*BAN31(item):fs*(BAN31(item))+(30*fs)), ...
        FilteredEEG(fs*BAN32(item):fs*(BAN32(item))+(30*fs)),hamming(5120),2560,[],fs);
    C_N3B_Storage(:, item) = Cxy_N3B;
    CoherencePhaseN3B(:, item) = angle(pxy_N3B);
end
Average_N2B = mean(C_N2B_Storage, 2);
Average_N3B = mean(C_N3B_Storage, 2);

CoherenceRET_0013 = Average_Coverall;                                      %% Things to be changed based on subjects
PhaseCoherenceRET_0013 = Average_CoherencePhase;                           %% Things to be changed based on subjects
targetFolder = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets';
fullFilePath = fullfile(targetFolder, CPSDCoherences);
% Full file path of coherence for depression cases
DFullfilepath = fullfile(fullFilePath, DepressionCoherences);
% Full file path of coherence for control cases
CFullfilepath = fullfile(fullFilePath, ControlCoherences);

save(CFullfilepath, CoherenceRET_0013);
save(CFullfilepath, PhaseCoherenceRET_0013);
%%% save PhaseCoherenceRET_0013 PhaseCoherenceRET_0013                         %% Things to be changed based on subjects
%%% save CoherenceRET_0013 CoherenceRET_0013                                   %% Things to be changed based on subjects
%save f_TOT f_TOT
% figure
% plot(f_N20B, Average_N2B)
% hold on
% plot(f_N2, Average_N2, 'r', LineWidth=1.5)
% legend('N2 Bootstrapping','N2 Identical', 'FontSize',20)
% xlabel('frequency (Hz)','FontSize',20)
% ylabel('Coherence','FontSize',20)
%
% figure
% plot(f_N30B, Average_N3B)
% hold on
% plot(f_N3, Average_N3, 'r', LineWidth=1.5)
% legend('N3 Bootstrapping','N3 Identical', 'FontSize',20)
% xlabel('frequency (Hz)','FontSize',20)
% ylabel('Coherence','FontSize',20)

%% PSD Analysis of ECG
% N = length (ECGDetrended(1:100000));
% ECGdft = fft(ECGDetrended(1:100000));
% ECGdft = ECGdft(1:(N/2)+1);
% PSDECG = (1/(fs*N)) * abs(ECGdft).^2;
% PSDECG(2:end - 1) = 2 * PSDECG(2:end - 1);
% freq = 0:fs/N:fs/2;
% figure
% plot(freq, 10*log10(PSDECG), 'b')
% xlabel('frequency (Hz)','FontSize',20)
% ylabel('fft of ECG','FontSize',20)
% figure
% plot(freq, 10*log10(PSDECG), 'b')
% xlabel('frequency (Hz)','FontSize',20)
% ylabel('PSD of ECG in logaritmic order','FontSize',20)
%figure
%plot(freq, abs(ECGdft), 'b')
%figure
%[pxx,f] = pwelch(HR,[],[],[],fs);
%plot(f, 10*log10(pxx))

%% Phase plots
% figure
% plot(f_REM0, Average_PhaseREM)
% hold on
% plot(f_N30, Average_PhaseN3)
% hold on
% plot(f_N20, Average_PhaseN2)
% hold on
% plot(f_Wake0, Average_PhaseWake)
% legend('REM Phase', 'N3Phase', 'N2Phase', 'WakePhase')

%save HR HR
