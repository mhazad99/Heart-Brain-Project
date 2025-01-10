
clc
clear
close all

matlabDir = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\Brainstorm\brainstorm_db';
dirList = dir(matlabDir);
isDir = [dirList(:).isdir];
for i=1:numel(dirList)
    % Specify the directory containing MATLAB files
    if isDir(i) && ~ismember(dirList(i).name, {'.', '..'})
        subjectname = dirList(i).name; % Get the file name
        % List MATLAB files in the directory
        InsideMatlabDir = fullfile(matlabDir, subjectname,'\data\', subjectname, 'NIGHT');
        matFiles = dir(fullfile(InsideMatlabDir, '*.mat'));


        % Process and modify each MATLAB file in the directory
        for item = 3:numel(matFiles)
            matFilePath = fullfile(InsideMatlabDir, matFiles(item).name);
            modifyAndSaveMatFile(matFilePath);
        end

        disp('Modifications and saving completed.');

    end
end

% Define a function to process and modify MATLAB files
function modifyAndSaveMatFile(filePath)
% Load the MATLAB file
matData = load(filePath);

% Perform modifications on the loaded data (example: add 10 to a variable)
matData.EEG(1, :) = matData.F(9, :);
matData.EEG(2, :) = matData.F(10, :);
matData.EEG(3, :) = matData.F(25, :);
matData.F = [];

% Save the modified data back to the same file
save(filePath, '-struct', 'matData');
end