clc
clear
close all
% Path to the directory containing the folders with .mat files
directoryPath = 'C:\Users\AS79560\OneDrive - ETS\Thesis\Datasets\CPSDCoherences\Control Cases(for insm)';

% List of folder names
folders = {'CoherencesC3', 'CoherencesF3', 'CoherencesO1'};

for i = 1:numel(folders)
    % Path to the current folder
    folderPath = fullfile(directoryPath, folders{i});

    % List all .mat files in the current folder
    matFiles = dir(fullfile(folderPath, '*.mat'));

    for j = 1:numel(matFiles)
        % Load the .mat file
        mat = load(fullfile(folderPath, matFiles(j).name));
        CoherencesMatrix(i, :, j) = mat.MagCoherence;
    end
end
CoherenceC3 = CoherencesMatrix(1, :, :);
CoherenceC3 = reshape(CoherenceC3, 4097, 19);
CoherenceF3 = CoherencesMatrix(2, :, :);
CoherenceF3 = reshape(CoherenceF3, 4097, 19);
CoherenceO1 = CoherencesMatrix(3, :, :);
CoherenceO1 = reshape(CoherenceO1, 4097, 19);
for i = 1:3
    % Path to the current folder
    if i == 1
        folderPath = fullfile(directoryPath, folders{i});
        % List all .mat files in the current folder
        matFiles = dir(fullfile(folderPath, '*.mat'));
        for j = 1:1%numel(matFiles)
            % Load the .mat file
            mat = load(fullfile(folderPath, matFiles(j).name));

            % Convert the data to a table (assuming variable is a matrix)
            T = array2table(CoherenceC3);

            % Save the table to an Excel file
            excelFileName = fullfile(folderPath, strrep(matFiles(j).name, '.mat', '.xlsx'));
            writetable(T, excelFileName);
        end
    elseif i == 2
        folderPath = fullfile(directoryPath, folders{i});
        % List all .mat files in the current folder
        matFiles = dir(fullfile(folderPath, '*.mat'));
        for j = 1:1%numel(matFiles)
            % Load the .mat file
            mat = load(fullfile(folderPath, matFiles(j).name));

            % Convert the data to a table (assuming variable is a matrix)
            T = array2table(CoherenceF3);

            % Save the table to an Excel file
            excelFileName = fullfile(folderPath, strrep(matFiles(j).name, '.mat', '.xlsx'));
            writetable(T, excelFileName);
        end
    elseif i == 3
        folderPath = fullfile(directoryPath, folders{i});
        % List all .mat files in the current folder
        matFiles = dir(fullfile(folderPath, '*.mat'));
        for j = 1:1%numel(matFiles)
            % Load the .mat file
            mat = load(fullfile(folderPath, matFiles(j).name));

            % Convert the data to a table (assuming variable is a matrix)
            T = array2table(CoherenceO1);

            % Save the table to an Excel file
            excelFileName = fullfile(folderPath, strrep(matFiles(j).name, '.mat', '.xlsx'));
            writetable(T, excelFileName);
        end
    end
end
