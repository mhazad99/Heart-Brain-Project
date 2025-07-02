function [resFiles,PARAM] = filterPSG(sFiles, PARAM)
%FILTERPSG - Filter PSG recording using different cut-off frequencies 
%   according to channel type
%
% SYNOPSIS: resFile = filterPSG(sFile, PARAM)
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

resFiles = sFiles;

if PARAM.Import.split_sleep_stages == 1
    for iStage=1:length(sFiles)
        fprintf('    Stage %s\n',PARAM.SleepStages{iStage})
        if isempty(sFiles{iStage})
            fprintf('Skipping. No epoch found in stage %s.\n',PARAM.SleepStages{iStage})
            continue
        end

        filteredIdx = cellfun(@(c) contains(c,'band'),{sFiles{iStage}.FileName});
        if ~any(filteredIdx)
            % TO DO: Selection of not already filtered only
            resFiles = bst_process('CallProcess', 'process_bandpass', sFiles{iStage}, [], ...
                'sensortypes', 'EEG', ...
                'highpass',    0.3, ...
                'lowpass',     35, ...
                'tranband',    0, ...
                'attenuation', 'strict', ...  % 60dB
                'ver',         '2019', ...  % 2019
                'mirror',      0, ...
                'read_all',    1);

            % Continue with new file links if process worked
            sFiles{iStage} = CleanAndUpdateFiles(sFiles{iStage},resFiles);
        end
    end
%     timeLapsed = [timeLapsed; toc];

    % Process: Band-pass EMG: 10-100 Hz
    %%%%%%%%%%%%%%%
    fprintf('Bandpass filtering EMG...\n');
    for iStage=1:length(sFiles)
        fprintf('    Stage %s\n',PARAM.SleepStages{iStage})
        if isempty(sFiles{iStage})
            fprintf('Skipping. No epoch found in stage %s.\n',PARAM.SleepStages{iStage})
            continue
        end

        filteredIdx = cellfun(@(c) contains(c,'band'),{sFiles{iStage}.FileName});
        if ~any(filteredIdx)
            % TO DO: Selection of not already filtered only
            resFiles = bst_process('CallProcess', 'process_bandpass', sFiles{iStage}, [], ...
                'sensortypes', 'EMG', ...
                'highpass',    10, ...
                'lowpass',     100, ...
                'tranband',    0, ...
                'attenuation', 'strict', ...  % 60dB
                'ver',         '2019', ...  % 2019
                'mirror',      0, ...
                'read_all',    1);

            % Continue with new file links if process worked
            sFiles{iStage} = CleanAndUpdateFiles(sFiles{iStage},resFiles);
        end
    end
    
else
    chanTypes = unique([resFiles.ChannelTypes]);
    %% FILTER EEG
    if ~isempty(PARAM.Filter.EEG) && any(contains(chanTypes,'EEG'))
        fprintf('PREP>_____FILTERING EEG...\n');tic
        
        
        if ~isempty(PARAM.Filter.order)
            %=== Use custom filter ===
            
            % Get sampling frequency
            sData = in_bst_data(resFiles(1).FileName);
            fs = 1 / (sData.Time(2) - sData.Time(1));
            % Create filter
            wn = PARAM.Filter.EEG / (fs/2);
            eeg_filter = fir1(1000,wn);
            % Get EEG channels
            cFileName = unique({resFiles.ChannelFile});
            sChan = in_bst_channel(cFileName,'Channel');
            iChan = strcmpi({sChan.Channel.Type},'EEG');
            
            for iEp = 1:length(resFiles)
                % Progress message (i/N)
                if iEp == 1; fprintf('PREP>\tepoch (%d/%d)',iEp,length(resFiles));
                else; PrintProgress(iEp,length(resFiles)); end
                % Read data
                sData = in_bst_data(resFiles(iEp).FileName);
                % Filter EEG channels
                sData.F(iChan,:) = filtfilt(eeg_filter,1,sData.F(iChan,:)')';
                % Save dataFile with comment append with filter frequencies
                sData.Comment = [sData.Comment ' (' PARAM.Filter.EEG(1) '-' PARAM.Filter.EEG(2) ') Hz'];
                bst_save(file_fullpath(resFiles(iEp).FileName),sData)
            end
        else
            % Use Brainstorm filter
            resFiles = bst_process('CallProcess', 'process_bandpass', resFiles, [], ...
                        'sensortypes', 'EEG', ...
                        'highpass',    PARAM.Filter.EEG(1), ...
                        'lowpass',     PARAM.Filter.EEG(2), ...
                        'tranband',    0, ...
                        'attenuation', 'strict', ...  % 60dB
                        'ver',         '2019', ...  % 2019
                        'mirror',      0, ...
                        'read_all',    1, ...
                        'overwrite',    1);
        end
        PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
            [PARAM.currentSubject ' | filterEEG'],toc);
    else
        fprintf('PREP>\tSkip EEG filter\n');
    end
    %% FILTER EOG
    if ~isempty(PARAM.Filter.EOG) && any(contains(chanTypes,'EOG'))
        fprintf('PREP>_____FILTERING EOG...\n');tic
        resFiles = bst_process('CallProcess', 'process_bandpass', resFiles, [], ...
            'sensortypes', 'EOG', ...
            'highpass',    PARAM.Filter.EOG(1), ...
            'lowpass',     PARAM.Filter.EOG(2), ...
            'tranband',    0, ...
            'attenuation', 'strict', ...  % 60dB
            'ver',         '2019', ...  % 2019
            'mirror',      0, ...
            'read_all',    1, ...
            'overwrite',    1);
        PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
            [PARAM.currentSubject ' | filterEOG'],toc);
    else
            fprintf('PREP>\tSkip EOG filter\n');
    end
    %% FILTER ECG
    if ~isempty(PARAM.Filter.ECG) && any(contains(chanTypes,'ECG'))
        fprintf('PREP>_____FILTERING ECG...\n');tic
        resFiles = bst_process('CallProcess', 'process_bandpass', resFiles, [], ...
            'sensortypes', 'ECG', ...
            'highpass',    PARAM.Filter.ECG(1), ...
            'lowpass',     PARAM.Filter.ECG(2), ...
            'tranband',    0, ...
            'attenuation', 'strict', ...  % 60dB
            'ver',         '2019', ...  % 2019
            'mirror',      0, ...
            'read_all',    1, ...
            'overwrite',    1);
        PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
            [PARAM.currentSubject ' | filterECG'],toc);
    else
        fprintf('PREP>\tSkip ECG filter\n');
    end
    %% FILTER EMG
    if ~isempty(PARAM.Filter.EMG) && any(contains(chanTypes,'EMG'))
        fprintf('PREP>_____FILTERING EMG...\n');tic
        resFiles = bst_process('CallProcess', 'process_bandpass', resFiles, [], ...
            'sensortypes', 'EMG', ...
            'highpass',    PARAM.Filter.EMG(1), ...
            'lowpass',     PARAM.Filter.EMG(2), ...
            'tranband',    0, ...
            'attenuation', 'strict', ...  % 60dB
            'ver',         '2019', ...  % 2019
            'mirror',      0, ...
            'read_all',    1, ...
            'overwrite',    1);
        PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
            [PARAM.currentSubject ' | filterEMG'],toc);
    else
        fprintf('PREP>\tSkip EMG filter\n');
    end
end
