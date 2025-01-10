function [CleanedEEGinfo] = EEGArtifactRemoval(folderPath, Stageidx, y, FinalLen, flag)

addpath(folderPath)

% Get a list of all .mat files in the folder
fileList = dir(fullfile(folderPath, '*.mat'));
% if flag == 1 it won't consider the sleep stages!
if flag
    C3count = 0; O1count = 0; F3count = 0;
    % Loop through each file and load its contents
    for i = 1:numel(fileList)-2
        fileName = fileList(i+2).name; % Get the file name
        filePath = fullfile(folderPath, fileName); % Get the full file path

        % Load the .mat file
        loadedData = load(filePath); % The loaded data will be stored in a struct
        C3list(i, 1) = loadedData.ChannelFlag(9, :);
        O1list(i, 1) = loadedData.ChannelFlag(10, :);
        F3list(i, 1) = loadedData.ChannelFlag(25, :);
        if C3list(i, 1) == 1 && O1list(i, 1) == 1 && F3list(i, 1) == 1
            C3count = C3count + 1;
            C3withoutArtifact(C3count, 1) = {loadedData.EEG(1, :)}; % Store the parts of signal that don't have artifact
            O1count = O1count + 1;
            O1withoutArtifact(O1count, 1) = {loadedData.EEG(2, :)};
            F3count = F3count + 1;
            F3withoutArtifact(F3count, 1) = {loadedData.EEG(3, :)};
        end
%         if O1list(i, 1) == 1
%             O1count = O1count + 1;
%             O1withoutArtifact(O1count, 1) = {loadedData.EEG(2, :)};
%         end
%         if F3list(i, 1) == 1
%             F3count = F3count + 1;
%             F3withoutArtifact(F3count, 1) = {loadedData.EEG(3, :)};
%         end
        ST = loadedData.History{3,3,1};
        ST1(i, 1) = str2double(ST(6:10));
    end
    CleanedEEGinfo.CleanedC3 = [C3withoutArtifact{:}];
    CleanedEEGinfo.CleanedO1 = [O1withoutArtifact{:}]; % Concatenate the cleaned segments of the signals
    CleanedEEGinfo.CleanedF3 = [F3withoutArtifact{:}];
    %artifactcount = sum(F3list == -1); % Counts the number of epochs that has artifact
    CleanedEEGinfo.C3artifactidx = find(C3list == -1);
    CleanedEEGinfo.O1artifactidx = find(O1list == -1);
    CleanedEEGinfo.F3artifactidx = find(F3list == -1); % Finds the index of epochs with artifact
    % finding the start time in second
    CleanedEEGinfo.StartTime = ST1(1,1);
    MergedIndexes = [CleanedEEGinfo.C3artifactidx; CleanedEEGinfo.O1artifactidx; CleanedEEGinfo.F3artifactidx];
    CleanedEEGinfo.MergedIndexes = unique(MergedIndexes);
else
    REMC3count = 0; NREMC3count = 0; REMO1count = 0; NREMO1count = 0;
    REMF3count = 0; NREMF3count = 0;
    % Loop through each file and load its contents
    for i = 1:FinalLen %numel(fileList)-2
        fileName = fileList(i+2).name; % Get the file name
        filePath = fullfile(folderPath, fileName); % Get the full file path
        
        % Load the .mat file
        loadedData = load(filePath); % The loaded data will be stored in a struct
        C3list(i, 1) = loadedData.ChannelFlag(9, :);
        O1list(i, 1) = loadedData.ChannelFlag(10, :);
        F3list(i, 1) = loadedData.ChannelFlag(25, :);
        if C3list(i, 1) == 1 && y(1, i) == 4
            REMC3count = REMC3count + 1;
            REMC3withoutArtifact(REMC3count, 1) = {loadedData.EEG(1, :)}; % Store the parts of signal that doesn't have artifact
        elseif C3list(i, 1) == 1 && (y(1, i) == 3 || y(1, i) == 2 || y(1, i) == 1)
            NREMC3count = NREMC3count + 1;
            NREMC3withoutArtifact(NREMC3count, 1) = {loadedData.EEG(1, :)}; % Store the parts of signal that doesn't have artifact
        end
        if O1list(i, 1) == 1 && y(1, i) == 4
            REMO1count = REMO1count + 1;
            REMO1withoutArtifact(REMO1count, 1) = {loadedData.EEG(2, :)};
        elseif O1list(i, 1) == 1 && (y(1, i) == 3 || y(1, i) == 2 || y(1, i) == 1)
            NREMO1count = NREMO1count + 1;
            NREMO1withoutArtifact(NREMO1count, 1) = {loadedData.EEG(2, :)};
        end
        if F3list(i, 1) == 1 && y(1, i) == 4
            REMF3count = REMF3count + 1;
            REMF3withoutArtifact(REMF3count, 1) = {loadedData.EEG(3, :)};
        elseif F3list(i, 1) == 1 && (y(1, i) == 3 || y(1, i) == 2 || y(1, i) == 1)
            NREMF3count = NREMF3count + 1;
            NREMF3withoutArtifact(NREMF3count, 1) = {loadedData.EEG(3, :)};
        end
        ST = loadedData.History{3,3,1};
        ST1(i, 1) = str2double(ST(6:10));
    end
    CleanedEEGinfo.CleanedC3REM = [REMC3withoutArtifact{:}];
    CleanedEEGinfo.CleanedC3NREM = [NREMC3withoutArtifact{:}];
    CleanedEEGinfo.CleanedO1REM = [REMO1withoutArtifact{:}]; % Concatenate the cleaned segments of the signals
    CleanedEEGinfo.CleanedO1NREM = [NREMO1withoutArtifact{:}];
    CleanedEEGinfo.CleanedF3REM = [REMF3withoutArtifact{:}];
    CleanedEEGinfo.CleanedF3NREM = [NREMF3withoutArtifact{:}];
    %artifactcount = sum(F3list == -1); % Counts the number of epochs that has artifact
    CleanedEEGinfo.C3artifactidx = find(C3list == -1);
    CleanedEEGinfo.O1artifactidx = find(O1list == -1);
    CleanedEEGinfo.F3artifactidx = find(F3list == -1); % Finds the index of epochs with artifact
    MergedIndexes = [CleanedEEGinfo.C3artifactidx; CleanedEEGinfo.O1artifactidx; CleanedEEGinfo.F3artifactidx];
    CleanedEEGinfo.MergedIndexes = unique(MergedIndexes);
    % finding the start time in second
    CleanedEEGinfo.StartTime = ST1(1,1);
end
end