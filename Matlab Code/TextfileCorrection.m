
clc
clear
close all

% To rewrite the text files for EEG artifact removal!
%% Read and Edit text file
% Get a list of all .mat files in the folder
folderpath = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\M-Hassan\Insomniacs Cases';
fileList = dir(fullfile(folderpath, '*.txt'));
for i = 1:numel(fileList)
    fileName = fileList(i).name; % Get the file name
    filePath = fullfile(folderpath, fileName); % Get the full file path

    % Load the .mat file
    T = readtable(filePath,'VariableNamingRule','preserve', 'ReadRowNames',true); % The loaded data will be stored in a struct
    Removingrows = T.Event;
    ValidElements = {'Wake','NREM 1','NREM 2','NREM 3','REM'};
    list = ismember(Removingrows, ValidElements);
    T(~list, :) = [];
    T.Duration(:) = 30;
    writetable(T, filePath, 'Delimiter','\t')
end
