function [resFiles,PARAM] = Import2Matlab(sFiles,PARAM)
%IMPORT2MATLAB - IMPORT RAW RECORDING IN RESPECT TO PARAM
%
% SYNOPSIS: resFiles = Import2Matlab(sFiles,PARAM)
%
% Required files:
%
% EXAMPLES:
%
% REMARKS:
%
% See also GetBSLightMarker, AskSleepStages, AddStepDuration
%
% Copyright Tomy Aumont

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Created with:
%   MATLAB ver.: 9.6.0.1135713 (R2019a) Update 3 on
%    Microsoft Windows 10 Home Version 10.0 (Build 17763)
%
% Author:     Tomy Aumont
% Work:       Center for Advance Research in Sleep Medicine
% Email:      tomy.aumont@umontreal.ca
% Website:    
% Created on: 23-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
% Start a new report
bst_report('Start',sFiles);


%% DEFINE IMPORTATION TIME LIMITS
%%%%%%%%%%%%%%%
fprintf('PREP>_____FETCHING IMPORTATION TIME LIMITS\n')

if PARAM.Import.night_only == 1
    % Check if NIGHT condition exist and load it, then return
    subjStudies = bst_get('StudyWithSubject',sFiles.SubjectFile);
    if ~isempty([subjStudies.Condition])
        nightCond = find(ismember([subjStudies.Condition],{'NIGHT'}));
        if ~isempty(nightCond)
            % Load existing epoch
            fprintf('PREP>\tFile already epoched.\n')
            fprintf('PREP>\tLOADING EPOCH FILES...\n')
            resFiles = [subjStudies(nightCond).Data];
            resFiles = bst_process('GetInputStruct', {resFiles.FileName});
            fprintf('PREP>\t%d epoch loaded.\n',length(resFiles));
            % Get night limits from first and last epoch time sample
            tmp{1} = in_bst_matrix(resFiles(1).FileName,'Time');
            tmp{2} = in_bst_matrix(resFiles(end).FileName,'Time');
            PARAM.Import.night_limits = [tmp{1}.Time(1), tmp{2}.Time(end)];
            return
        end
    end
    
    % ===== NIGHT condition do not exist =====
    
    % Get night limits in seconds from raw recording
    PARAM.Import.night_limits = GetBSLightMarker(sFiles);
    % Set night begin at the first sleep scoring event start.
    sData = in_bst_data(sFiles.FileName,'F');
    % Get sleep stages event indices
    [~,locEvt] = ismember(PARAM.SleepStages,{sData.F.events.label});
    % Get closest sleep stage start time from the Lights Off marker time
    lagPerEvtStart = cellfun(@(c) min(abs(c-PARAM.Import.night_limits(1))), ...
        {sData.F.events(locEvt).times}, 'UniformOutput', false);
    [minLagPerEvt,iMinPerEvt] = cellfun(@min, lagPerEvtStart);
    [~,iMin] = min(minLagPerEvt);
    % Get closest sleep stage end time from the Lights On marker time
    lagPerEvtEnd = cellfun(@(c) min(abs(c-PARAM.Import.night_limits(2))), ...
        {sData.F.events(locEvt).times}, 'UniformOutput', false);
    [minLagPerEvtEnd,iMinPerEvtEnd] = cellfun(@min, lagPerEvtEnd);
    [~,iMinEnd] = min(minLagPerEvtEnd);
    % Assign corresponding marker times to night_limits
    PARAM.Import.night_limits = [sData.F.events(locEvt(iMin)).times(1,iMinPerEvt(iMin)), ...
        sData.F.events(locEvt(iMinEnd)).times(1,iMinPerEvtEnd(iMinEnd))];

    % Round night limits to nearest epoch (still in seconds) | OLD WAY THAT MAY CREATE DOUBLE MARKERS
%     remains = mod(PARAM.Import.night_limits,PARAM.Import.length);
%     roundUp = remains > PARAM.Import.length / 2;
%     roundUpVal = PARAM.Import.length - remains(roundUp);
%     PARAM.Import.night_limits(roundUp) = PARAM.Import.night_limits(roundUp) + roundUpVal;
%     PARAM.Import.night_limits(~roundUp) = PARAM.Import.night_limits(~roundUp)  - remains(~roundUp);
    fprintf('PREP>\tNight limits in seconds: %d to %d\n',PARAM.Import.night_limits)
    % Name the condition that will contain imported file "NIGHT"
    PARAM.Import.condition = 'NIGHT';
else
    % Do not limit only to night segment
    PARAM.Import.night_limits = [];
    PARAM.Import.condition = [];
end

%%%%%%%%%%%%%%%
% IMPORT DATA TO MATLAB
%%%%%%%%%%%%%%%

if strcmpi(PARAM.Import.type,'continuous')
    %% IMPORT ONE CONTINUOUS FILE
    disp('PREP>_____IMPORTING CONTINUOUS...')
    % Indicate not to split in epoch
    PARAM.Import.length = 0;
    % Process: Import MEG/EEG: Time
    resFiles = bst_process('CallProcess', 'process_import_data_time', sFiles, [], ...
        'subjectname', sFiles(1).SubjectName, ...
        'condition',   [PARAM.Import.condition 'Copp'], ...
        'timewindow',  PARAM.Import.night_limits, ...
        'split',       PARAM.Import.length, ...
        'ignoreshort', 1, ...
        'usectfcomp',  0, ...
        'usessp',      1, ...
        'freq',        PARAM.ResampleFreq, ...
        'baseline',    []);
    fprintf('PREP>\tContinuous file imported\n');
else
    %% IMPORT MANY EPOCH
    disp('PREP>_____IMPORTING EPOCH...')
    if PARAM.Import.split_sleep_stages == 1
        fprintf('\nImporting epoch into sleep stages conditions...\n');
        if isempty(PARAM.SleepStages)
            % Select which stages to split from menu list
            PARAM.SleepStages = AskSleepStages();
            if isempty(PARAM.SleepStages)
                fprintf('User cancelled sleep stage selection. Exit\n')
                return
            end
        end
        %% IMPORT EPOCH ONE CONDITION AT A TIME
       resFiles = cell(1,length(PARAM.SleepStages));
       sStudies = bst_get('StudyWithSubject',sFiles.SubjectFile);
       for iStage=1:length(PARAM.SleepStages)
           fprintf('    Stage %s\n',PARAM.SleepStages{iStage})
            
           % Check if condition is already imported
           importedIdx = cellfun(@(c) strcmp(c,PARAM.SleepStages{iStage}),{sStudies.Name});
           if  any(importedIdx)
               % Get already splitted files from database
               fprintf('        Stage already in database.\n')
       %             resFiles{iStage} = cellfun(@(c) c.FileName, {sStudies(importedIdx).Data}, 'UniformOutput', false);
       %             resFiles{iStage} = bst_process('GetInputStruct', resFiles{iStage});
               inFiles = [sStudies(importedIdx).Data];
               resFiles{iStage} = bst_process('GetInputStruct', {inFiles.FileName});
               fprintf('        %d epoch loaded.\n',length(resFiles{iStage}));
           else
               % Process: Import condition's epoch files
               fprintf('    Importing stage %s epochs\n',PARAM.SleepStages{iStage});
               resFiles{iStage} = bst_process('CallProcess', 'process_import_data_event', sFiles, [], ...
                   'subjectname',  sFiles(1).SubjectName, ...
                   'condition',   PARAM.SleepStages{iStage}, ...
                   'eventname',   PARAM.SleepStages{iStage}, ...
                   'timewindow',  PARAM.Import.night_limits, ...
                   'epochtime',   [0, PARAM.Import.length], ...
                   'createcond',  1, ...
                   'ignoreshort', 1, ...
                   'usectfcomp',  0, ...
                   'usessp',      1, ...
                   'freq',        PARAM.ResampleFreq, ... % resample
                   'baseline',    []);
               
               % Remove duplicate sleep stage events
               CleanSleepStageEvent(resFiles,PARAM.SleepStages);
			   
               fprintf('        %d epoch produced\n',length(resFiles{iStage}));
           end
       end
    else
        %% IMPORT EPOCH OF EQUAL DURATION
       % Process: Import MEG/EEG: Time
       resFiles = bst_process('CallProcess', 'process_import_data_time', sFiles, [], ...
           'subjectname', sFiles(1).SubjectName, ...
           'condition',   PARAM.Import.condition, ...
           'timewindow',  PARAM.Import.night_limits, ...
           'split',       PARAM.Import.length, ...
           'ignoreshort', 1, ...
           'usectfcomp',  0, ...
           'usessp',      1, ...
           'freq',        PARAM.ResampleFreq, ...
           'baseline',    []);

       fprintf('PREP>\t%d epoch produced\n',length(resFiles));
    end
end

% Save brainstorm report
bst_report('Save', resFiles)
% Log step duration
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
	[PARAM.currentSubject ' | Import2Matlab'],toc);
