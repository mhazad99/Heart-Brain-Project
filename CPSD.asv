clc
clear
close all

%% load data
folderPath = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Insomniacs Cases'; % Change based on cases
ChannelNameList = ["C3REM", "C3NREM", "O1REM", "O1NREM", "F3REM", "F3NREM"];
ChannelNames = ["C3", "O1", "F3"];
for iter = ChannelNames % Change it when you are using sleepstages or the entire sleep together!!!
    [signal] = ReadNestedFolders(folderPath, iter);
end

function [signal] = ReadNestedFolders(folderPath, iter)
EEGVar = sprintf('Cleaned%s', iter);
            %%% Change the next two lines based on the application, whether you are using sleep stages or the entire night!!!
%ECGVar = sprintf('ECG%s', iter); % for sleep stages
ECGVar = 'ECGCleaned'; % for entire night
count = 0;
% List files and folders in the given folder
contents = dir(folderPath);
% Loop through each item in the folder
for i = 1:numel(contents)
    item = contents(i);
    % Exclude "." and ".." folders
    if strcmp(item.name, '.') || strcmp(item.name, '..')
        continue;
    end

    % Check if the item is a folder
    if item.isdir
        % Recursively call the function for nested folders
        subFolderPath = fullfile(folderPath, item.name);
        [signal] = ReadNestedFolders(subFolderPath, iter);
        if length(signal{1,1}.CleanedEEGinfo.(EEGVar)) == length(signal{1,2}.ECG.(ECGVar))
            EEG = signal{1,1}.CleanedEEGinfo.(EEGVar);
            ECG = signal{1,2}.ECG.(ECGVar);
            CalculateCoherence(EEG, ECG, item.name, char(iter))
        else
            disp('length difference')
            newECGlength = length(signal{1,1}.CleanedEEGinfo.(EEGVar)); % cut the end of ECG signal to be aligned with EEG
            EditedECG = signal{1,2}.ECG.(ECGVar)(1:newECGlength);
            EEG = signal{1,1}.CleanedEEGinfo.(EEGVar);
            CalculateCoherence(EEG, EditedECG, item.name, char(iter))
        end
    else
        % Process the file
        filePath = fullfile(folderPath, item.name);
        %filePath = fullfile(item.folder, item.name);
        if contains(filePath, '.edf') || contains(filePath, '.txt')
            continue
        else
            count = count + 1;
            signal{count} = load(string(filePath));
        end

    end

end
end
%%
function [] = CalculateCoherence(a, b, subjectname, iter)
EEG = a;
ECG = b;
fs = 256;
loop = length(EEG)/(fs*30);
for item = 1:loop-1
    [Cxy_TOT,f_TOT] = mscohere(ECG(1+30*fs*(item-1):30*fs*item), ...
        EEG(1+30*fs*(item-1):30*fs*item),hamming(5120),2560,[],fs);
    [pxy_TOT,f_TOT] = cpsd(ECG(1+30*fs*(item-1):30*fs*item), ...
        EEG(1+30*fs*(item-1):30*fs*item),hamming(5120),2560,[],fs);
    C_overall(:, item) = Cxy_TOT;
    CoherencePhase(:, item) = angle(pxy_TOT);
end

%%%%%% Comment or Uncomment the following lines if you want to calculate
%%%%%% the avg of the coherences!
Average_Coverall = mean(C_overall, 2);  % Uncomment it if you want to calculate the avg of the coherences!
MagCoherence = Average_Coverall;  % Uncomment it if you want to calculate the avg of the coherences!
%MagCoherence = C_overall; Uncomment it if you want to have all of the
%coherences for all of epochs
%Average_CoherencePhase = mean(CoherencePhase, 2);
%PhaseCoherence = Average_CoherencePhase;
targetFolder = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences';
% Full file path of coherence for depression cases
% Fullfilepath = fullfile(targetFolder, 'SleepStagesBased', ['DepressionCoherences', iter]);
% mkdir(Fullfilepath)

% Full file path of coherence for control cases
%Fullfilepath = fullfile(targetFolder, 'SleepStagesBased', ['ControlCoherences', iter]); % Uncomment it when you want to compute the sleep stages based coherences

Fullfilepath = fullfile(targetFolder, 'Insomniac Cases', ['Coherences', iter]);
mkdir(Fullfilepath)
file1Path = fullfile(Fullfilepath, ['MagCoherence' subjectname '.mat']);   % Change the directory for depressed or controls
%file2Path = fullfile(Fullfilepath, ['PhaseCoherence' subjectname '.mat']);% Change the directory for depressed or controls
save(file1Path, "MagCoherence", "f_TOT");
%save(file2Path, "PhaseCoherence", "f_TOT");

end
%%
% figure
% plot(f_TOT, Average_Coverall, 'b')
% xlabel('frequency (Hz)','FontSize',20)
% ylabel('Magnitude of Coherence','FontSize',20)
% hold on
% xline(0.5, '--');xline(3.5, '--');xline(4, '--');xline(7.5, '--')
% xline(8, '--');xline(11.5, '--');xline(12, '--');xline(15.5, '--')
% xline(16, '--');xline(19.5, '--')
% figure
% plot(f_TOT, Average_CoherencePhase, 'r')
% xlabel('frequency (Hz)','FontSize',20)
% ylabel('Phase of Coherence','FontSize',20)
% hold on
% xline(0.5, '--');xline(3.5, '--');xline(4, '--');xline(7.5, '--')
% xline(8, '--');xline(11.5, '--');xline(12, '--');xline(15.5, '--')
% xline(16, '--');xline(19.5, '--')
