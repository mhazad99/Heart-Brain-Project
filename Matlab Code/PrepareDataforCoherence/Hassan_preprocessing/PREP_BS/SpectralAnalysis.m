function [sFilesStages,PARAM] = SpectralAnalysis(sFiles,PARAM)
%SPECTRALANALYSIS - Compute spectral analysis on each sleep stage separately
%
% SYNOPSIS: [sFilesStages,PARAM] = SpectralAnalysis(sFiles,PARAM)
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
% Created on: 31-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% psdFiles = [];

disp('PREP>_____SPECTRAL ANALYSIS...');tic

if isstruct(sFiles) %size(sFiles,1) == 1
    tic
    % Start a new report
    bst_report('Start',sFiles);
    
    if isempty(PARAM.SleepStages)
        fprintf(2,'PREP> No sleep stages detected, skip spectral analysis.\n')
        % Save and display report
        bst_report('Save', []);
        return
    end
    
    %% Extract epoch per sleep stage
    for iStage = 1:length(PARAM.SleepStages)
        % Process: Import MEG/EEG: Events (sleep stages)
         sFilesStages{iStage} = bst_process('CallProcess', 'process_import_data_event', sFiles, [], ...
             'subjectname', PARAM.currentSubject, ...
             'condition',   '', ...
             'eventname',   PARAM.SleepStages{iStage}, ...
             'timewindow',  [], ...
             'epochtime',   [0 PARAM.Import.length], ... in seconds
             'createcond',  1, ...
             'ignoreshort', 1, ...
             'usectfcomp',  0, ...
             'usessp',      1, ...
             'freq',        [], ...
             'baseline',    []);
    end
    for iStage = 1:length(PARAM.SleepStages)
        fprintf('PREP>\t\t%d\t epoch of %s\n',length(sFilesStages{iStage}),PARAM.SleepStages{iStage})
    end
    % Save and display report
    ReportFile = bst_report('Save', [sFilesStages{:}]);
    
    PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
            [PARAM.currentSubject ' | SleepEpochSelection'],toc);
else
    sFilesStages = sFiles;
end

%% Process: Power spectrum density (Welch)
tic
% Start a new report
bst_report('Start', [sFilesStages{:}]);
iSubj = strcmpi(PARAM.Subjects,PARAM.currentSubject);
for iStage=1:length(sFilesStages)
    
    fprintf('\nPREP> Analysing stage %s\n',PARAM.SleepStages{iStage})
    if isempty(sFilesStages{iStage})
        fprintf('Skipping. No epoch found in stage %s.\n',PARAM.SleepStages{iStage})
        continue
    end

    PARAM.PSD{iSubj,iStage} = bst_process('CallProcess', 'process_psd', sFilesStages{iStage}, [], ...
        'timewindow', [], ...
        'win_length',  6, ...       freq_res = Fs / 2^nextpow2(win_length*Fs)
        'win_overlap', 50, ...
        'sensortypes', 'EEG', ...
        'win_std',     0, ...
        'edit',        struct(...
             'Comment',         ['Power,Avg,' PARAM.SleepStages{iStage}], ...
             'TimeBands',       [], ...
             'Freqs',           {{'delta', '0.1, 4', 'mean'; 'theta', '4, 8', 'mean'; 'alpha', '8, 12', 'mean'; 'sigma', '12, 16', 'mean'; 'beta', '15, 32', 'mean'; 'gamma', '30, 90', 'mean'}}, ...
             'ClusterFuncTime', 'none', ...
             'Measure',         'power', ...
             'Output',          'average', ...
             'SaveKernel',      0));
         

end
% Save and display report
if iscell(PARAM.PSD)
    bst_report('Save', [PARAM.PSD{:}]);
else
    bst_report('Save', PARAM.PSD);
end
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
            [PARAM.currentSubject ' | SpectralAnalysis'],toc);
        