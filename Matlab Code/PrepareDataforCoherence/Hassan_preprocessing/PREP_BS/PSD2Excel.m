function PSD2Excel(subjectNames)
%PSD2EXCEL - Convert Brainstorm PSD files for a list of subjects to a single Excel files.
%   The excel format is:
%       Subject_1_ID | SleepStage1_FreqBand1 | SleepStage1_FreqBand2 | ... | SleepStage2_FreqBand1 | ...
%       EEG_Sensors1 | ... | ...
%       EEG_Sensors2 | ... | ...
%       ...
%       (empty line)
%       (empty line)
%       Subject_2_ID | SleepStage1_FreqBand1 | SleepStage1_FreqBand2 | ... | SleepStage2_FreqBand1 | ...
%       ...
%       _______
% 
% SYNOPSIS: PSD2Excel()
%
%       subjectNames:   Cell array of subject names.
% 
% Required files:
%
% EXAMPLES:
%
% REMARKS:
%   TODO:
%       - ADD OPTION TO READ PSD FILES FROM THE PARAM STRUCTURE (see PreProcess.m)
% See also 
%
% Copyright Tomy Aumont

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Created with:
%   MATLAB ver.: 9.6.0.1135713 (R2019a) Update 3 on
%    Microsoft Windows 10 Home Version 10.0 (Build 17763)
%
% Author:     Tomy Aumont
% Work:       Center for Advance Research in Sleep Medicine
% Email:      tomy.aumont@umontreal.ca
% Website:    
% Created on: 24-Jun-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get base directory to look for PSD files
disp('Select Brainstorm protocol directory')
protocolDir = uigetdir(pwd,'Select Brainstorm protocol directory');
if ~protocolDir
    disp('User cancelled. Exit')
    return
end

% Get subject to process
if ~nargin
    subject_list            = f_GetPath(fullfile(protocolDir,'data'),true);
    subject_list            = subject_list(~endsWith(subject_list,{'@default_study','inter','.mat','.DS_Store'}));
    [~,subject_name_list]   = cellfun(@fileparts,subject_list,'UniformOutput',false);
    [iSelect,isOk]          = listdlg(  'PromptString','Select subjects to process:', ...
                                        'SelectionMode','multiple', ...
                                        'ListString',subject_name_list);
  
    if isOk
        subjectNames    = subject_name_list(iSelect);
        disp(['Subject to process: ' strjoin(subjectNames,', ')])
    else
        disp('No subject provided. Exit')
        return
    end
    
%     windowTitle     = 'Enter comma separated subject names';
%     disp(windowTitle)
%     subjectNames = split(inputdlg(windowTitle,'Subject names',[1 50]),',');
%     if isempty(subjectNames)
%         disp('No subject provided. Exit')
%         return
%     else
%         disp(['Subject to process: ' strjoin(subjectNames,', ')])
%     end
end

% Get output directory
disp('Select output directory')
resDir = uigetdir(pwd,'Select output directory');
if ~resDir
    disp('User cancelled. Exit')
    return
end

% Prepare output file name
dtNow = char(datetime('now','Format','dd-MM-yyyy_HH_mm_ss'));
outputFileName = [resDir filesep 'PSD_' dtNow '.xlsx'];
    
% Get input files (partial path from Brainstorm subject dir)
fileNames = GetPSDFiles(protocolDir,subjectNames);

% Gather data
data = cell(length(subjectNames),size(fileNames,1));
colNames = cell(length(subjectNames),size(fileNames,1));
TF = cell(length(subjectNames),size(fileNames,1));

for iSubj = 1:size(fileNames,2)
    for iFile = 1:size(fileNames,1)
        if ~isempty(fileNames{iFile,iSubj})
            % Read PSD file
           % data{iSubj,iFile} = load(fileNames{iFile,iSubj},'TF','RowNames','Freqs','Comment');
            data{iSubj,iFile} = in_bst_timefreq(fileNames{iFile,iSubj},1,'TF','RowNames','Freqs','Comment');
            % Get sleep stage
            tmp = split(data{iSubj,iFile}.Comment,',');
            stage = tmp{end};
            % Frequency bands per sleep stage as column names
            colNames{iSubj,iFile} = strcat(stage,'_',data{iSubj,iFile}.Freqs(:,1)','_',data{iSubj,iFile}.Freqs(:,2)','_Hz');
            colNames{iSubj,iFile} = strrep(colNames{iSubj,iFile},'.','_');
            colNames{iSubj,iFile} = strrep(colNames{iSubj,iFile},', ','_to_');
            % Format data to fit a table
            TF{iSubj,iFile} = cell(1,size(data{iSubj,iFile}.TF,3));
            for i= 1:length(colNames{iSubj,iFile})
                TF{iSubj,iFile}{i} = data{iSubj,iFile}.TF(:,1,i);
            end
        end
    end
end

% Format data to fit exel file
nbColumnPerSubj = sum(cellfun(@(c) length(c),TF),2);
padPerSubject = max(nbColumnPerSubj) - nbColumnPerSubj;
dataOut = [];
for iSubj = 1:length(subjectNames)
        tfData = [TF{iSubj,:}];
        nChan = unique(cellfun(@max,cellfun(@size,tfData,'UniformOutput',false)));
        if length(nChan)~=1
            nChan       = sort(nChan);
            chan_names  = cellfun(@(c) c.RowNames',data(iSubj,:),'UniformOutput',false);
            % Padd missing channels with NaN
            tfData = TF{iSubj,1};
            for ii = 2:length(chan_names)
                if ~isequal(size(chan_names{ii}),size(chan_names{1}))
                    iPos                        = ismember(chan_names{1},chan_names{ii});
                    tmp                         = cell(size(iPos,1),length(TF{iSubj,ii}));
                    tmp(iPos,:)                 = TF{iSubj,ii};
                    tmp(cellfun(@isempty,tmp))  = {NaN};
                    for i=1:size(tmp,2)
                        tfData              = [tfData,[tmp{:,i}]'];
                    end
                    
                else
                    tfData      = [tfData,TF{iSubj,2}];
                end
            end
%             warning('Error with subejct %s\n--> %s',subjectNames{iSubj},ME.message)
        end
        dataOut = [dataOut; ...
                subjectNames{iSubj},colNames{iSubj,:}, cell(1,padPerSubject(iSubj)); ...
                data{iSubj}.RowNames', num2cell([tfData{:}]), cell(size(num2cell([tfData{:}]),1),padPerSubject(iSubj)); ...
                cell(2,length([colNames{iSubj,:}])+padPerSubject(iSubj)+1)];
end

% Write excel file
disp(['Saving excel result file: ' outputFileName])
writecell(dataOut,outputFileName);
