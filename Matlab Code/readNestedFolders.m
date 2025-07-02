function readNestedFolders(folderPath)
% List files and folders in the given folder
contents = dir(folderPath);
disp(contents)
% Loop through each item in the folder
for i = 1:numel(contents)
    item = contents(i);
    disp(item)
    % Exclude "." and ".." folders
    if strcmp(item.name, '.') || strcmp(item.name, '..')
        continue;
    end

    % Check if the item is a folder
    if item.isdir
        % Recursively call the function for nested folders
        subFolderPath = fullfile(folderPath, item.name);
        readNestedFolders(subFolderPath);
    else
        % Process the file
        filePath = fullfile(folderPath, item.name);
        %EEG = load(item(1)); ECG = load(item(2));
        % Add your code to read and process the file here
        % For example, you can display the file name
        disp(filePath);
        loadedData = load('filePath.mat');
    end
    break
end
end