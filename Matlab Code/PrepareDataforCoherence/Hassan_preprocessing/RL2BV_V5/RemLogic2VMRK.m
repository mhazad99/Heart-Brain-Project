% function RemLogic2VMRK()
% SYNOPSIS: RemLogic2VMRK()
%
% Required files:
%       RemLogic event file(s) in a text format.
%           - File names must begin with the subject ID. Length of the subject ID is define by the
%               variable SUBJ_ID_LENGTH.
%           - File content must include 2 OR 3 columns following one of the headers:
%               + Epoch & Event
%               + Time, Event & Duration
%       BrainVision event files (.vmrk)
%
% EXAMPLES:
%
% REMARKS: If some marker files (.vmrk) have already been merged with their RemLogic event
%       file and it is desired to re-run this script for other subjects, you must create a new folder
%       for the new subjects, else the markers will be append twice to the old subjects marker files
%       You could also most the converted RemLogic file to another location, this script won't
%       modify any file that is not paired with a RemLogic event file.
%
% See also 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Created with:
%   MATLAB ver.: 9.6.0.1099231 (R2019a) Update 1 on
%    Microsoft Windows 10 Home Version 10.0 (Build 17763)
%
% Author:     Tomy Aumont
% Work:       Center for Advance Research in Sleep Medicine
% Email:      tomy.aumont@umontreal.ca
% Website:    
% Created on: 16-May-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear; clc;

SUBJ_ID_LENGTH = 8; % length to look at when verifying correspondance in file name

%% SELECT REMLOGIC DIRECTORY
%%%%%%%%%%%%%%%%
disp('RL2BV>     Select Directory with RemLogic event files...')
srcDir = uigetdir(pwd,'Select directory with RemLogic Sleep Scoring files');
%srcDir = uigetdir(pwd,'Select Directory with RemLogic Sleep scoring files');
if srcDir == 0; disp('RL2BV>     User has exit'); return; end

%% SELECT RECORDING DIRECTORY
%%%%%%%%%%%%%%%%
disp('RL2BV>     Select Recordings Directory...')
destDir = uigetdir(pwd, 'Select Recordings Directory');
if destDir == 0; disp('RL2BV>     User has exit'); return; end

%% LIST SOURCE FILES TO CONVERT
%%%%%%%%%%%%%%%%
srcFileList = dir([srcDir filesep '*.txt']);
% Get subject ID from source file name
[~,srcSubjName,~] = cellfun(@(x) fileparts(x(1:SUBJ_ID_LENGTH)), ...
                                                {srcFileList.name},'UniformOutput',0);
% Sort file by name length (position file with suffix at the end)
[~,idx] = sort(cellfun(@(c) length(c),{srcFileList.name}));
srcFileList = srcFileList(idx);
% Generate full filepaths
srcFileList = fullfile({srcFileList.folder},{srcFileList.name});

%% LIST MARKER FILES TO APPEND
%%%%%%%%%%%%%%%%
switch questdlg('Append existing marker files or create new ones?','Choose','Append','Create','Append')
    case 'Append'
        destFileList = dir([destDir filesep '*.vmrk']);
        % Skip file containing '_bckup' or '_orig' in their name
        keep = ~contains({destFileList.name},{'_bckup','_orig'});
        destFileList = destFileList(keep);
        
        % Get subject ID from destination file name
        [~,destSubjName,~] = cellfun(@(x) fileparts(x(1:SUBJ_ID_LENGTH)), ...
                                                        {destFileList.name},'UniformOutput',0);
        
        % Keep only files corresponding to the source subjects
        % matchedIdx = find(ismember(destFileSubjName,srcFileSubjName)); % matching indexes
%         matchedIdx = ismember(destSubjName,srcSubjName);
        matchedIdx = contains(destSubjName,srcSubjName);
        if isempty(matchedIdx)
            fprintf(2,'RL2BV>     ERROR: No .vmrk file corresponds to input files.\n')
            return
        end
        destFileList = destFileList(matchedIdx);
            
        if length(srcFileList) ~= length(destFileList)
            % Find subjects with different number of input and output files
            nSubjSrcFiles = cellfun(@(c) sum(contains(srcFileList,c)),srcSubjName);
            nSubjDestFiles = cellfun(@(c) sum(contains({destFileList.name},c)),srcSubjName);
            nFileDiff = nSubjSrcFiles - nSubjDestFiles;
            errSubj = find(nSubjSrcFiles ~= nSubjDestFiles);

            for iSubj = 1:length(errSubj)
                if nFileDiff(iSubj) > 0
                    % Some input files do not have corresponding output files
                    iFile = contains(srcFileList,srcSubjName{iSubj});
                    % Get file names without extension
                    [~,fname,~] = cellfun(@(c) fileparts(c),srcFileList(iFile),'UniformOutput',false);
                    % Remove RemLogic-added suffix '-Events'
                    fname = strrep(fname,'-Events','');
                    % Find index of the file matching the output file
                    isInput = cellfun(@(c) find(contains({destFileList.name},c)),fname,'UniformOutput',false);
                    isInput = ~cellfun(@isempty,isInput);
                    % Mask to remove unmatching file
                    iFile(isInput) = 0;
                    % Update output file list
                    srcFileList = srcFileList(~iFile);
                    
                elseif nFileDiff(iSubj) < 0
                    % Some output files do not have corresponding input files
                    iFile = contains({destFileList.name},srcSubjName{iSubj});
                    % Get file name without extension
                    [~,fname,~] = cellfun(@(c) fileparts(c),{destFileList(iFile).name},'UniformOutput',false);
                    % Find index of the file matching the input file
                    isInput = cellfun(@(c) find(contains(srcFileList,c)),fname,'UniformOutput',false);
                    isInput = ~cellfun(@isempty,isInput);
                    % Mask to remove unmatching file
                    iFile(isInput) = 0;
                    % Update output file list
                    destFileList = destFileList(~iFile);
                end
            end
            
%             if any(nSubjSrcFiles > nSubjDestFiles) %length(srcFileList) > length(destFileList)
%                 % Skip input files without corresponding output file
%                 errFile = nSubjSrcFiles > nSubjDestFiles;
%                 fprintf('WARNING: No recording files corresponds to following input files. Skipping...\n');
%                 iErrFile = find(errFile);
%                 for iFile = 1:length(iErrFile)
%                     fprintf('\t%s\n',srcFileList{iErrFile(iFile)});
%                 end
%                 srcFileList = srcFileList(~errFile);
%             end
%             
%             if any(nSubjDestFiles > nSubjSrcFiles) %length(srcFileList) < length(destFileList)
%                 % Skip output files without corresponding input file
%                 nWrongFile = nSubjDestFiles > nSubjSrcFiles;
%                 for iSubj = 1:length(nWrongFile)
%                     fnames = cellfun(@(c) c(1:end-5),{destFileList(iSubj).name})
%                     contains(srcFileList,fnames)
%                 end
%                 destFileList = destFileList(~errFile);
%             end
        end
        % Generate full filepaths
        destFileList = fullfile({destFileList.folder},{destFileList.name});
        
        
    case 'Create'
        % Marker file with same name as scoring file but .vmrk in destination folder
        % More than one file can have the same name. Will append these files
        % instead of creating them.
        if contains(destDir,'EGI',"IgnoreCase",true)
            destDirList = dir(destDir);
            % Keep only directories
            destDirList = destDirList([destDirList.isdir]);
            
            % ===== SAM CRITERIA ONLY =====
            destDirList = destDirList(5:end);
            % =============================
            
            % Get file list from all recording directories as 1 cell array
            destFileList = arrayfun(@GetFiles ,destDirList,'UniformOutput',false);
            destFileList = cat(1,destFileList{:});
            % Remove empty file names (caused by GetFiles when an hidden file is found)
            destFileList = destFileList(~cellfun(@isempty, destFileList));
            % Keep only recordings that explicitly shows the .mff extension
            destFileList = destFileList(cellfun(@(c) endsWith(c,'.mff','IgnoreCase',true) , destFileList));
            
            % Keep only files that correspond to RemLogic marker files
            % Based on the subject name and assume that there is only one
            % recording per visit folder
            srcSubj = split(srcFileList',filesep);
            srcSubj  = split(srcSubj(:,end),'-Events'); srcSubj(:,1)
            destFileList = destFileList(contains(destFileList,srcSubj));

        elseif contains(destDir, 'EDF', "IgnoreCase", true)

            destFileList = f_GetPath(destDir);

            % Keep only recordings that explicitly shows the .mff extension
            destFileList = destFileList(cellfun(@(c) endsWith(c,{'.rec','.edf'},'IgnoreCase',true) , destFileList));
            
            % Keep only files that correspond to RemLogic marker files
            % Based on the subject name and assume that there is only one
            % recording per visit folder
            srcSubj = split(srcFileList',filesep);
            srcSubj  = split(srcSubj(6,end),'.txt'); srcSubj(:,1)
            destFileList = destFileList(contains(destFileList,srcSubj));
        else
            destFileList = srcFileList;
            for iRow=1 : length(srcFileList)
    %             destFileList(iRow).folder = destDir;
                [fPath, fName, fExt] = fileparts(srcFileList{iRow});
                % Remove tailing tag from retrospective database files
                fExt = strrep(lower(fExt),'txt','vmrk');
%                 fName = strrep(upper(fName),'-TAG','');
%                 fName = strrep(upper(fName),'_TAG','');
                fName = regexprep(fName, '-TAG', '', 'ignorecase');
                fName = regexprep(fName, '_TAG', '', 'ignorecase');
                fName = regexprep(fName, '-EVENTS', '', 'ignorecase');
                destFileList{iRow} = strrep(fullfile(fPath,[fName,fExt]),' ','');
            end
        end
    otherwise
        disp('Exit.')
        return
end

%% READ & APPEND MARKER FILES
%%%%%%%%%%%%%%%%
for i=1:length(srcFileList)
    fprintf('\n\nRL2BV>_____PROCESSING FILE (%d/%d)\n',i,length(srcFileList))
    [~,evt] = readRemLogicEvtFile(srcFileList{i},false);
%     f_ImportRemLogicEvt2Bst(destFileList{i},evt);
    AddRemLogicEvt2VMRK(destFileList{i}, evt);
end

disp('RL2BV>     Complete')

% end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       HELPER FUNCTIONS
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cFileList = GetFiles(sDir)
    dirPath = fullfile(sDir.folder,sDir.name);
    fileList = dir(dirPath); fileList = fileList(3:end);
    
    % ===== SAM CRITERIA ONLY =====
    % Get biggest MFF file (whole night)
    [~, dSize] = system(['du -sk ' dirPath filesep '*.mff']);
    dSize = strsplit(dSize,{dirPath,newline});
    dSize = dSize(~contains(dSize,filesep));
    [~,bigMFF] = max(str2double(dSize));
    fileList = fileList(contains({fileList.name},'.mff','IgnoreCase',true));
    % ==============================
    
    cFileList = arrayfun(@GetPath,fileList(bigMFF),'UniformOutput',false);
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function strPath = GetPath(sFile)
    if startsWith(sFile.name,'.')
        strPath = '';
    else
        strPath = fullfile(sFile.folder,sFile.name);
    end
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
