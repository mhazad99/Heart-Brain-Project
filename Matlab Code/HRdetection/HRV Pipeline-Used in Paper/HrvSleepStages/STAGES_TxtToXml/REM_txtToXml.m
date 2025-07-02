 % RET_txtToXml.m
%
clear;
clc; 

% Set the input data files directorry
sourceDirectory = uigetdir(pwd, 'Select the input data files directory');
fprintf('Input folder: %s\n',sourceDirectory);

% Set the output data files directorry
destinationDirectory = uigetdir(pwd, 'Select the desination data files directory');
fprintf('Output folder: %s\n',destinationDirectory);

% Generate the name of the CSV information file.
dateTimeStr = string(datetime,'yyyy-MM-dd-HH-mm-ss');
infoCSVFileName = strcat(destinationDirectory,filesep,'Info_',dateTimeStr,'.csv');
fprintf('Creating information file: %s\n',infoCSVFileName);
CreateCSVFileTitles(infoCSVFileName);

% Retrieve the input data file names
[inputFileNames,  baseFileNames] = GetSourceFiles(sourceDirectory);

nbInputFiles = length(inputFileNames);
if (nbInputFiles > 0)
   
   for i=1:nbInputFiles
      [header,data] = ReadInputFile(char(inputFileNames(1,i)));
      WriteOutputFile(header, baseFileNames(i), data, destinationDirectory);
      UpdateCSVInfoFile(header,infoCSVFileName);
   end   
end   