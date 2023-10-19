clc
clear
close all

%% Calling gpu
gpu = gpuDevice();
%% load data
folderPath = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Depression Cases'; % Change based on cases
[signal] = ReadNestedFolders(folderPath);
function [signal] = ReadNestedFolders(folderPath)
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
        [signal] = ReadNestedFolders(subFolderPath);
        if length(signal{1,1}.CleanedEEGinfo.CleanedO1) == length(signal{1,2}.ECG.ECGO1)
            EEG = signal{1,1}.CleanedEEGinfo.CleanedO1; %gpuEEG = gpuArray(EEG);
            ECG = signal{1,2}.ECG.ECGO1; %gpuECG = gpuArray(ECG);
            CalculateSpectrogram(EEG, ECG, item.name)
            clear CalculateSpectrogram
        else
            newECGlength = length(signal{1,1}.CleanedEEGinfo.CleanedO1); % cut the end of ECG signal to be aligned with EEG
            EditedECG = signal{1,2}.ECG.ECGO1(1:newECGlength); %gpuECG = gpuArray(EditedECG);
            EEG = signal{1,1}.CleanedEEGinfo.CleanedO1; %gpuEEG = gpuArray(EEG);
            CalculateSpectrogram(EEG, EditedECG, item.name)
            clear CalculateSpectrogram
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

%% Calculate spectrograms
function [] = CalculateSpectrogram(EEG, ECG, subjectname)
fs = 256;
loop = length(EEG)/(fs*30);
targetFolder = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets';
% Full file path of coherence for depression cases
DFullfilepath = fullfile(targetFolder, 'Spectrograms', 'Arrays', 'O1', 'Depressed', subjectname);
mkdir(DFullfilepath)
for item = 1:loop-1
    if item == 1 || item == loop-1
        continue
    else
        %%%%%%%%%%%%% EEG
        [S1, f, t] = spectrogram(EEG(1+30*fs*(item-1)-(11*fs):30*fs*item), ... 
            hamming(2560),2304,[],fs); %%% It will calculate the STFT from 10s window before
            % the start of the main window (it is 9 because when the window comes to the calculation period it is passed one step)
            % and it does't need the 10s after because it will be
            % calculated in the next window ---> I changed the 9 to 11 to
            % match the dimension
            
        % Calculate the magnitude spectrogram
        magnitude_S1 = abs(S1(1:512, :));
        % Rescale the magnitude values to a suitable range
        min_val1 = min(magnitude_S1(:));
        max_val1 = max(magnitude_S1(:));
        scaled_magnitude_S1 = (magnitude_S1 - min_val1) / (max_val1 - min_val1);
        cpuarray = gather(scaled_magnitude_S1);
        %%% [numFreqBins, numTimeSegments] = size(s);
        %axis off
        %%% Capture the current figure
        %fig = gcf; frame = getframe(fig);
        %%% Convert the captured frame to an image
        %image1 = frame2im(frame);
        %cropped_image1 = image1(78:327, 66:190, :);
        %%% Specify the file path and format for saving the image
        EEGfilename = 'EEG_' + string(subjectname) + '_' + string(item);
        fileformat = 'mat';
        file1Path = fullfile(DFullfilepath, [char(EEGfilename) '.' fileformat]);
        %         filename = sprintf('data_%d.mat', i);
        save(file1Path, 'cpuarray');
        %imwrite(cropped_image1, file1Path);
        %close
        %%%%%%%%%%%%% ECG
%         [S2, f, t] = spectrogram(ECG(1+30*fs*(item-1)-(11*fs):30*fs*item), ...
%             hamming(2560),2304,[],fs);
%         magnitude_S2 = abs(S2(1:512, :));
%         % Rescale the magnitude values to a suitable range
%         min_val2 = min(magnitude_S2(:));
%         max_val2 = max(magnitude_S2(:));
%         scaled_magnitude_S2 = (magnitude_S2 - min_val2) / (max_val2 - min_val2);
%         cpuarray = gather(scaled_magnitude_S2);  
%         %axis off
%         %fig = gcf; frame = getframe(fig);
%         % Convert the captured frame to an image
%         %image2 = frame2im(frame);
%         %cropped_image2 = image2(78:327, 66:190, :); % Image margins for create spectrograms (30 second upto 30 Hz)
%         ECGfilename = 'ECG_' + string(subjectname) + '_' + string(item);
%         file2Path = fullfile(DFullfilepath, [char(ECGfilename) '.' fileformat]);
%         save(file2Path, 'cpuarray');
        %imwrite(cropped_image2, file2Path);
        %close
    end
end
% cropped_image = image1(78:327, 66:177, :); % fixed 30 second and 35 Hz of the spectrogram
% imshow(cropped_image)
end