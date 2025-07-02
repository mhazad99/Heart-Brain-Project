 % START_XmlToTxt.m
%
clear;
clc; 

% Get the folder with input data files
sourceDirectory = uigetdir(pwd, 'Select the input data files directory');
sourceDirectory = string(sourceDirectory);
fprintf('Input folder: %s\n',sourceDirectory);

% Fet teh list of all XML files to convert
allXmlFiles = dir(strcat(sourceDirectory,filesep,'*.xml'));
nbXmlFiles = length(allXmlFiles);

for i=1:nbXmlFiles
    xmlFileName = string(allXmlFiles(i).name);
    fprintf("\tProcessing XML file %s (%d/%d)\n",xmlFileName,i,nbXmlFiles);
    
    % Read the Xml data file and extract the sleep statge scores.
    fullXmlFileName = strcat(sourceDirectory,filesep,xmlFileName);
    stagesNum = GetStagesFromXml(fullXmlFileName);
    
    % Create and write corresponding text file.
    temp = strsplit(xmlFileName,".edf.xml");
    baseFileName  = string(temp(1));
    fprintf("\tWriting data to %s TEXT file...", strcat(baseFileName,".txt"));
    WriteTextFile(baseFileName, sourceDirectory, stagesNum)
    fprintf("\n");
end    
fprintf("DONE!\n");