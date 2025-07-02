function [sFiles,PARAM] = Split2Epoch(insFiles,PARAM,sStudies)
%SPLIT2EPOCH - Split continuous recording to epoch. Can be divided by sleep stages or not.
%
% SYNOPSIS: sFiles = Split2Epoch(sFiles,PARAM)
%
% Required files:
%
% EXAMPLES:
%
% REMARKS:
%
% See also 
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
disp('PREP>_____EPOCHING...')
    
if PARAM.Epoch.split_sleep_stages == 1
    if isempty(PARAM.SleepStages)
        %% Select which stages to split from menu list
        PARAM.SleepStages = AskSleepStages();
        if isempty(PARAM.SleepStages)
            fprintf('User cancelled sleep stage selection. Exit\n')
            return
        end
    end
    %% Splitting
    fprintf('\nSplitting raw file into sleep stages epochs...\n');
    sFiles = cell(1,length(PARAM.SleepStages));
    %% Split epoch and group by sleep stages
    for iStage=1:length(PARAM.SleepStages)
        fprintf('    Stage %s\n',PARAM.SleepStages{iStage})

        importedIdx = cellfun(@(c) strcmp(c,PARAM.SleepStages{iStage}),{sStudies.Name});
        if  any(importedIdx)
            % Get already splitted files from database
            fprintf('        Stage already in database.\n')
    %             sFiles{iStage} = cellfun(@(c) c.FileName, {sStudies(importedIdx).Data}, 'UniformOutput', false);
    %             sFiles{iStage} = bst_process('GetInputStruct', sFiles{iStage});
            inFiles = [sStudies(importedIdx).Data];
            sFiles{iStage} = bst_process('GetInputStruct', {inFiles.FileName});
            fprintf('        %d epoch selected.\n',length(sFiles{iStage}));
        else
            % Process: Split raw files by selected sleep stages
            fprintf('    Importation of stage %s epochs\n',PARAM.SleepStages{iStage});
            sFiles{iStage} = bst_process('CallProcess', 'process_import_data_event', insFiles, [], ...
                'subjectname',  insFiles(1).SubjectName, ...
                'condition',   PARAM.SleepStages{iStage}, ...
                'eventname',   PARAM.SleepStages{iStage}, ...
                'timewindow',  [], ...
                'epochtime',   [0, 30], ... % 200 ms for future filter edge effect
                'createcond',  1, ...
                'ignoreshort', 1, ...
                'usectfcomp',  0, ...
                'usessp',      1, ...
                'freq',        PARAM.ResampleFreq, ... % resample at 100 Hz
                'baseline',    []);



    %             sFiles{iStage} = bst_process('CallProcess', 'process_split_raw_file', sRawFiles, [], ...
    %                 'eventname', sleepStages{iStage}, ...
    %                 'keepbadsegments', 0);
            fprintf('        %d epoch produced\n',length(sFiles{iStage}));
        end
    end
else
    %% SPLIT INTO EPOCH OF EQUAL DURATION
%     sCond = bst_get('ConditionsForSubject', sSubject.FileName);
    if PARAM.Epoch.night_only == 1
        % Extract epoch for night period only
        subjStudies = bst_get('StudyWithSubject',insFiles.SubjectFile);
        if ~isempty([subjStudies.Condition])
            nightCond = find(ismember([subjStudies.Condition],{'NIGHT'}));
            if ~isempty(nightCond)
                % Load existing epoch
                fprintf('PREP>\tFile already epoched.\n')
                fprintf('PREP>\tLOADING EPOCH FILES...\n')
                sFiles = [subjStudies(nightCond).Data];
                sFiles = bst_process('GetInputStruct', {sFiles.FileName});
                fprintf('PREP>\t%d epoch loaded.\n',length(sFiles));
                % Get night limits from first and last epoch time sample
                tmp{1} = in_bst_matrix(sFiles(1).FileName,'Time');
                tmp{2} = in_bst_matrix(sFiles(end).FileName,'Time');
                PARAM.Epoch.night_limits = [tmp{1}.Time(1), tmp{2}.Time(end)];
                return
            end
        end
        % Get night limits in seconds
        PARAM.Epoch.night_limits = GetBSLightMarker(insFiles);
        % Round night limits to nearest epoch (still in seconds)
        remains = mod(PARAM.Epoch.night_limits,PARAM.Epoch.length);
        roundUp = remains > PARAM.Epoch.length / 2;
        roundUpVal = PARAM.Epoch.length - remains(roundUp);
        PARAM.Epoch.night_limits(roundUp) = PARAM.Epoch.night_limits(roundUp) + roundUpVal;
        PARAM.Epoch.night_limits(~roundUp) = PARAM.Epoch.night_limits(~roundUp)  - remains(~roundUp);
        fprintf('PREP>\tNight limits in seconds: %d to %d\n',PARAM.Epoch.night_limits)
        % Tag exported dataset as only night
        PARAM.Epoch.condition = 'NIGHT';
    else
        % Extract epoch for the whole recording
        PARAM.Epoch.night_limits = [];
        PARAM.Epoch.condition = [];
    end
    
    % Start a new report
    bst_report('Start',insFiles);
        
    % Process: Import MEG/EEG: Time
    sFiles = bst_process('CallProcess', 'process_import_data_time', insFiles, [], ...
        'subjectname', insFiles(1).SubjectName, ...
        'condition',   PARAM.Epoch.condition, ...
        'timewindow',  PARAM.Epoch.night_limits, ...
        'split',       PARAM.Epoch.length, ...
        'ignoreshort', 1, ...
        'usectfcomp',  0, ...
        'usessp',      1, ...
        'freq',        PARAM.ResampleFreq, ...
        'baseline',    []);
    
    % Save and display report
    ReportFile = bst_report('Save', sFiles);
   
    fprintf('PREP>\t%d epoch produced\n',length(sFiles));
end
% Log step duration
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
	[PARAM.currentSubject ' | Split2Epoch'],toc);