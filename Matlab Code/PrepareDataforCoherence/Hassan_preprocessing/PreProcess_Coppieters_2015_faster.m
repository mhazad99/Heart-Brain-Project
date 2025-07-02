%% PREPROCESS_COPPIETERS_2015
%
%   Clean high dentisty EEG recordings in the following way:
%       01. Re-reference
%       02. Epoching
%       03. Filtering
%       04. Bad channel rejection (Coppieters, 2015)
%       05. Bridged channel rejection (Segalowitz, 2013)
%       06. Bad segment rejection based on EEG high standard deviation
%       07. Bad segment rejection based on EMG high standard deviation
%       08. Power spectrum density
% 
%   FUTURE WORK
%       - find ref for high/low correlation method to reject events
%       - detect high gradient (negative) to reject events in freq domain
%           - for <0.45 sec or >3 sec (maybe amp * freq??) 
%       - merge events separated by less than 0.5 sec.
% 		   
%       09. TODO: Integrate Export PSD to Excel at then end
%       10. TODO: Slow waves detection
%       11. TODO: Spindles detection
%       12. TODO: Resting state PSD with wake as baseline
% 
%   BUGS:
%       - lights off appears where in should not (many epoch)
%       - F1 & F2 flag by HighAmp way too often
% 		 
% REQUIREMENTS
%       - BrainVision recordings (.eeg, .vhdr, .vmrk) in a single folder
%       - Marker files (.vmrk) containing the sleep scoring markers, else use RemLogic2VMRK.m
%       - Comma-separeted list of subject names to process
%       - Brainstorm software installed (https://neuroimage.usc.edu/brainstorm/Introduction)
% 
% OUTPUTS
%   - Sleep stage epochs with (BAD) event markers
%   - Power spectrum density per sleep stages per subject (.mat)
%       - window length = 6 seconds
%       - window overlap = 50%
%       - spectral band:
%           + delta = 0.1-4 Hz`
%           + theta = 4-8 Hz
%           + alpha = 8-12 Hz
%           + sigma = 12-16 Hz
%           + beta = 15-32 Hz
%           + gamma = 30-90 Hz
%%
clear; clc;
addpath('PREP_BS','brainstorm3','PREP_BS_2');
% mkdir  D:\MAPS\analysis SUBJECTS_INFO 
%% PARAMETERS DEFINITION
%%%%%%%%%%%%%%%
% Brainstorm-related parameters
PARAM.Brainstorm = struct( ...
    'protocol', [], ...                 Brainstorm Protocol name (i.e. 'maps', [] to select existing or create a new one)
    'UseDefaultAnat', 0, ...              1 to use default anatomy when creating new subject, 0 to use individual anatomy
    'UseDefaultChannel', 0, ...         1 to use default channel file, 0 to use one channel file per recording block
    'Subjects', [] ...                          keep empty. It will be set automatically
    );
% Channel-related parameters
PARAM.Channel = struct( ...
    'ImportPosition',[], ...                1 to import position
    'detect_very_bad',[], ...             1 to detect very bad channel (flat and very noisy), [] to skip
    'detect_noisy', 1, ...              1 to detect noisy channels, [] to skip
    'FlatChannel', 1,...                1 to detect flat channels, [] to skip 
    'Threshold',[],...                    1 to detect peak amplitudes, [] to skip 
    'detect_bridges',[], ...           'Segalowitz_2013', 'High_Correl' or [] to skip.
    'detect_low_r', [], ...              IN DEV | 1 to detect bad channels based on low correlation, [] to skip
    'reject_visual', [], ...             1 to stop pipeline after import+reref to reject channel visually, [] to skip
    'noisy_thresh', 6000, ...            BETA   |  only if detect_very_bad: in microVolt, if channel's std > flag bad per epoch (t_rn = 6000uV in Coppieters,2015)
    'flat_tresh', 1, ...                 BETA   | only if detect_very_bad: in microVolt, if channel's std < flag bad per epoch (t_rf = 1uV in Coppieters,2015)
    'noisy_ratio', 5, ...                BETA   | only if detect_noisy: if  std(Chan_i) / std(mean_others) > flag bad channel (t_rr = 5 in Coppieters,2015)
    'min_corr', [], ...                  only if detect_bridges | high correlation that that is considered bridging, default is 0.99
    'max_dist', 3, ...                   only if detect_bridges | neighbouring distance to find possible bridged neighbors, default is 0.3 if []
    'window_len', [], ...                only if detect_bridges | time window during which correlation is computer, in second, default is 0.5
    'min_corr_time', 0.8 ...             only if 'detect_bridge' = 1 | percentage of window that is bridged to reject channel pair (default is 0.8)
    );
% Rereference parameter
PARAM.Reref = 'M1,M2';              % name of electrode(s) to reference to, [] to skip, comma separated electrode names to average (i.e. average mastoid 'M1,M2')
PARAM.SleepStages = {};             % Sleep stages to process. Detected automatically
PARAM.Continuous = [];              % Convert epoch back to continuous file, [] to skip
% Importation to Matlab parameters
PARAM.Import = struct( ...
	'type',	'epoch', ...        	'epoch' or 'continuous' | Define if import continuous or epoch files to Matlab, [] to skip importation to Matlab
    'length', 30, ...               only if type = 'epoch' | epoch duration in seconds, [] to skip epoching
    'night_only', 1, ...            1 to pre-process only night data base on ["lights off" "lights on"] markers, 0 for all recording, [] to skip
    'night_limits', [], ...         detect automatically using "light on" and "light off" marker
    'condition', [], ...            Set automatically. Name of condition folder to contain the epochs (i.e. 'NIGHT'), default []
    'split_sleep_stages', 0 ...     1 to split data into sleep stages, 0 to split into adjacent epoch of same duration
    );
% Resampling
PARAM.ResampleFreq = [];      % sample frequency in Hz to resample to, [] to skip
% Filtering-related parameters
PARAM.Filter = struct( ... permanent filtering if resampling is permitting way larger frequency than interested in
    'EEG', [], ...        in Hz, [] to skip (i.e. [0.5 35])
    'EOG', [], ...       in Hz, [] to skip (i.e. [0.3 15])
    'ECG', [], ...        in Hz, [] to skip (i.e. [10 40])
    'EMG', [], ...         in  Hz, [] to skip (i.e. [10 50])
    'order', [] ...
    );
% Artifacts-related parameters
PARAM.Artifact = struct( ...
    'BadSegment', [], ...                   1 to detect EEG segment with amplitude > n * std, [] to skip
    'cardiac', [], ...                          1 to detect QRS peaks on ECG, [] to skip
    'eye_movement', [], ...               1 to detect event > 3*std in 1-7Hz frequency band, [] to skip
    'movement', [], ...                      1 to detect movement on EMG based on high standard deviation. [] to skip
    'visual_check', [], ...
    'correction', [], ...                      only if movement OR cardiac OR eye_movement = 1 | 1 to correct artifact using SSP, [] to skip
    'EEG_std_thresh', 7, ...               only if BadSegment = 1 | Multiplier of standard deviation per EEG epoch.
    'EEG_min_time_interval', 0.5, ...   only if BadSegment = 1 | in seconds, minimum time interval between 2 events
    'std_tresh_chan', 4, ...                only if movement = 1 | 6 works well for EMG. Multiplier of standard deviation
    'std_thresh_percent', 0.1, ...        only if movement = 1 | BETA: not implemented at all yet. Minimum percentage of epoch with std > threshold to reject channel
    'std_min_time_interval', 0.1 ...    only if movement = 1 | minimum interval between 2 events in seconds.
    );
% Types of analysis
PARAM.Analysis = struct( ...
    'spectral', [], ...                1 to execute spectral analysis, [] to skip
    'time_frequency', [] ...      IN DEV    | 1 to execute time-frequency analysis, [] to skip
    );
PARAM.PSD = []; % Contains all PSD data produced
PARAM.StepDuration = {'TotalTime',0};  % Step duration log

PARAM.SaveDir = ['results' filesep 'PARAMETERS'];
%% SET BRAINSTORM ENVIRONMENT
%%%%%%%%%%%%%%%
PARAM = SetupBrainstormEnv(PARAM);

%% GET LIST OF SUBJECTS TO CREATE OR PROCESS
%%%%%%%%%%%%%%%
subject_name_list   = {PARAM.Brainstorm.Subjects.Subject.Name};
if ~isempty(subject_name_list)
    [iSelect,isOk]      = listdlg(  'PromptString','Select subjects to process:', ...
                                    'SelectionMode','multiple', ...
                                    'ListString',subject_name_list);
    if isOk && any(iSelect)
        PARAM.Subjects  = subject_name_list(iSelect);
    else
        fprintf(2,'PREP> ERROR: No subject provided. Exit\n')
        return
    end
else
    disp('Enter comma separated subject names')
    PARAM.Subjects = split(inputdlg('Enter comma separated subject names','Subject names',[1 50]),',');
    if isempty(PARAM.Subjects)
        fprintf(2,'PREP> ERROR: No subject provided. Exit\n')
        return
    end
end
% TODO
% Add selection of subject to process
% RemLogic2VMRK; % Convert event file from text to BrainVision vmrk format
%% GET RECORDING FILES OF EACH SUBJECT
%%%%%%%%%%%%%%%
[rawFiles,PARAM] = GetRawFiles(PARAM); % .eeg | .vhdr | .vmrk
if isempty(rawFiles)
    fprintf(2,'PREP> ERROR: No raw files found. Exit\n')
    return
end
%% BEGIN PRE-PROCESSING
%%%%%%%%%%%%%%%
bst_report('Start', []);
for iSubj =1:length(PARAM.Subjects)
    PARAM.currentSubject = PARAM.Subjects{iSubj};
    fprintf('\n%s\n',repmat('=',1,50))
    fprintf('PREP> PROCESSING SUBJECT: %s\n', PARAM.currentSubject);
    fprintf('%s\n\n',repmat('=',1,50))
    sFiles = [];    % Input files
    %% IMPORT RAW FILES TO BRAINSTORM (RECORDINGS)
    %%%%%%%%%%%%%%%
    [sFiles,PARAM] = ImportRecordings(structfun(@(s) s(iSubj),rawFiles),PARAM);
    % Import EEG position NOT IMPLEMENTED
      if ~isempty(PARAM.Channel.ImportPosition)
        FilePath = 'D:\MAPS_Files_original\Localizer_Files_RR\MP2_0125_T1_V3.txt';
        ImportEEGposition(sFiles, FilePath);
      end
    % Check if EOG type is assigned to E1,E2,E3 and assigned it if it is not
%     sFiles = CheckEOG(sFiles);
    % Get sleep stages from recording events
    PARAM = GetSleepStages(sFiles,PARAM);
   
%     file_name = sprintf('file%d.txt',ii);    
    %% RE-REFERENCE EEG
    %%%%%%%%%%%%%%%
    if ~isempty(PARAM.Reref)
        [resFiles,PARAM] = Rereference(sFiles,PARAM);
        if isempty(resFiles)
            fprintf(2,'PREP>\tCould not re-reference files... Keeping file unchanged.\n')
        else
            sFiles = resFiles;
        end
    end
    sFiles = bst_process('CallProcess', 'process_channel_settype', sFiles, [], ...
    'sensortypes', 'M1,M2,LOC,ROC,LAT1,LAT2,RAT1,RAT2,SpO2,L-Arm1,L-Arm2,R-Arm1,R-Arm2,Thermistor, Ptrans, CHEST, ABDOMEN,Snore,Sum',...
    'newtype',     'Misc');


	%% IMPORT RECORDING TO MATLAB (EPOCH)
	%	Also resample if PARAM.ResampleFreq has a value.
    %   Can be many epochs or one continuous file
    %%%%%%%%%%%%%%%
    if ~isempty(PARAM.Import.type)
		[resFiles,PARAM] = Import2Matlab(sFiles,PARAM);
        if isempty(resFiles)
            fprintf(2,'ERROR> IMPORTATION TO MAATLAB FAILED.\n')
            return
        else
            sFiles = resFiles;
        end
    end
    %here set channels to -1 
    %% BAND-PASS FILTER
    %%%%%%%%%%%%%%%
    if any(~structfun(@isempty,PARAM.Filter))
        [resFiles, PARAM] = filterPSG(sFiles, PARAM);
        if isempty(resFiles)
            fprintf(2,'ERROR> FILTERING FAILED.\n')
        else
            sFiles = resFiles;
        end
    end
    %% IN TUNNING: SET VERY BAD CAHANNEL ACCORDING TO STD (COPPIETERS, 2015)
    if ~isempty(PARAM.Channel.detect_very_bad)
        [sFiles,PARAM,ObviouslyBadChannel] = VeryBadChannels(sFiles,PARAM);               
    end
    %% IN TUNNING: SET NOISY CHANNEL  (COPPIETERS, 2015)
    if ~isempty(PARAM.Channel.detect_noisy)
        [sFiles,PARAM,NoisyChannel] = NoisyChannels(sFiles,PARAM);    
    end
    %% SET FLAT CHANNEL  (COPPIETERS, 2015)
    if ~isempty(PARAM.Channel.FlatChannel)
        [sFiles,PARAM,flat_channel] = FlatChannel(sFiles,PARAM);
    end
        %% IN_DEV:    DETECT BAD CHANNELS BASE ON LOW PEARSON CORRELATION (R)
    if ~isempty(PARAM.Channel.detect_low_r)
        [resFiles,PARAM] = DetectBadChannels(sFiles,PARAM);
        if isempty(resFiles)
            fprintf(2,'    Could not detect bad channels... Keeping file unchanged.\n')
        else
            sFiles = resFiles;
        end
    end        
    %% BETA:    BRIDGED-CHANNELS BASED ON HIGH CORRELATION
    if ~isempty(PARAM.Channel.detect_bridges)
        [PARAM] = BridgedChannels(sFiles,PARAM); % just changed retained in DetectBridges
     end
    %% THRESHOLD:   DETECTED AMPLITUDE ABOVE A CERTAIN THRESHOLD
    if ~isempty(PARAM.Channel.Threshold)
        [sFiles,PARAM,ThresholdChannel] = Threshold(sFiles,PARAM);
    end
%  
%      artefacts = struct('obviously_bad_channel',ObviouslyBadChannel,'noisy_channel',NoisyChannel, ...
%          'flat_channel',flat_channel,'threshold',ThresholdChannel);
% 
%      nameartefact = strcat('artefact_',PARAM.currentSubject  , '.mat');
%      save(nameartefact,'artefacts',  '-v7.3');
    %% DETECT BAD EMG SEGMENT BASE ON HIGH STD
    if ~isempty(PARAM.Artifact.movement)
        [resFiles,PARAM] = DetectMovements(sFiles,PARAM);
        if isempty(resFiles)
            fprintf(2,'    Could not detect movements... Keeping file unchanged.\n')
        else
            sFiles = resFiles;
        end
    end   
    %% CONVERT TO CONTINUOUS FILE
    %%%%%%%%%%%%%%%
    if ~isempty(PARAM.Continuous)
        [resFiles,PARAM] = Epoch2Continuous(sFiles,PARAM);
        if isempty(resFiles)
            fprintf(2,'PREP> ERROR: Could not convert into continuous file...\n')
        else
            sFiles = resFiles;
        end
    end
    % REVIEW FILE AS RAW
    %%%%%%%%%%%%%%
    if ~isempty(PARAM.Continuous)
        [resFiles, PARAM] = ReviewAsRaw(sFiles,PARAM);
        if isempty(resFiles)
            fprintf(2,'PREP> ERROR: Could not review file as raw...\n')
        else
            sFiles = resFiles;
        end
    end
    % DETECT CARDIAC EVENT
    %%%%%%%%%%%%%%%
    if ~isempty(PARAM.Artifact.cardiac)
        [resFiles,PARAM] = DetectCardiac(sFiles,PARAM);
        if isempty(resFiles)
            fprintf(2,'    Could not detect heartbeats... Keeping file unchanged.\n')
        else
            sFiles = resFiles;
        end
    end
    %% DETECT EYE MOVEMENTS [ 1-7 Hz ] ON EOG
    %%%%%%%%%%%%%%%
    if ~isempty(PARAM.Artifact.eye_movement)
        [resFiles,PARAM] = DetectBadSegment(sFiles,PARAM);
        if isempty(resFiles)
            fprintf(2,'    Could not detect 1-7Hz segments... Keeping file unchanged.\n')
        else
            sFiles = resFiles;
        end
    end
    %% STOP HERE FOR VISUALLY REJECTING CHANNELS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~isempty(PARAM.Artifact.visual_check)
        WaitChannelReject(PARAM);
    end
        
    if PARAM.Artifact.correction == 1
        %% REMOVE / REJECT ARTIFACTS - RAW FILES NEEDED
        %%%%%%%%%%%%%%%  
        % EYE MOVEMENTS
        [resFiles,PARAM] = CorrectEyeMovement(sFiles,PARAM);
        
        if ~isempty(resFiles)
            sFiles = resFiles;
            % Detect Eye movement that where not corrected
            if ~isempty(PARAM.Artifact.eye_movement)
                [resFiles,PARAM] = DetectBadSegment(sFiles,PARAM);
                if isempty(resFiles)
                    fprintf(2,'    Could not detect 1-7Hz segments... Keeping file unchanged.\n')
                else
                    sFiles = resFiles;
                    % Reject detected events
                    resFiles = bst_process('CallProcess', 'process_evt_rename', sFiles, [], ...
                        'src',  '1-7Hz', ...
                        'dest', 'bad_EyeMovement');
                end
            end
        end
        
        % HEARTBEATS
        [resFiles,PARAM] = CorrectHeartbeats(sFiles,PARAM);
        if isempty(resFiles)
            fprintf(2,'PREP>\tCould not remove heartbeat\n')
        end
    end
          
     if 0
        % Process: Detect heartbeats
        fprintf('Detecting heartbeats...\n');tic
        for iStage=1:length(sFiles)
            fprintf('    Stage %s\n',sleepStages{iStage})
            if isempty(sFiles{iStage})
                fprintf('Skipping. No epoch found in stage %s.\n',sleepStages{iStage})
                continue
            end

            resFiles = bst_process('CallProcess', 'process_evt_detect_ecg', sFiles{iStage}, [], ...
                'channelname', 'ECG', ...
                'timewindow',  [], ...
                'eventname',   'cardiac');

            if isempty(resFiles)
                fprintf(2,'    Could not detect heartbeats... Keeping file unchanged.\n')
            end
        end
        timeLapsed = [timeLapsed; toc];

        % Process: SSP2 ECG: cardiac remove 1st component from first cardiac event type
        fprintf('Removing heartbeat using PCA (SSP)...\n');tic
        for iStage=1:length(sFiles)
            fprintf('   Stage %s\n',sleepStages{iStage})
            if isempty(sFiles{iStage})
                fprintf('Skipping. No epoch found in stage %s.\n',sleepStages{iStage})
                continue
            end

            % Process: SSP2 ECG: cardiac => SSP2 else files are not long enough
            resFiles = bst_process('CallProcess', 'process_ssp2_ecg', sFiles{iStage},  sFiles{iStage}, ...
                'eventname',   'cardiac', ...
                'sensortypes', 'EEG', ...
                'usessp',      1, ...
                'select',      1);

            if isempty(resFiles)
                % No projection worked
                fprintf(2,'    Could not remove heartbeats... Keeping file unchanged.\n')
            else
                % Projections worked
                PrintProcessStatusPerEpoch(sFiles{iStage},resFiles);
            end
        end
        timeLapsed = [timeLapsed; toc];
        db_save_bak;

        %% DETECT MOVEMENT ON EMG -----> DONE RIGHT AFTER FIRST EPOCHING
        %%%%%%%%%%%%%%%
        if 0
        % Process: Detect custom events: when data > std_tresh * std / channel / epoch
        % Get channel names comma separated
        fprintf('Detecting artifact by n * std method on EMG...\n');tic

        % Get EMG channel names
        ChannelStruct = in_bst_channel(sFiles{iStage}(1).ChannelFile);
        idxEMG = strcmpi({ChannelStruct.Channel.Type},'EMG');
        emgChan = strjoin({ChannelStruct.Channel(idxEMG).Name},',');

        for iStage=1:length(sFiles)
            fprintf('    Stage %s\n',sleepStages{iStage})
            if isempty(sFiles{iStage})
                fprintf('Skipping. No epoch found in stage %s.\n')
                continue
            end

            resFiles = bst_process('CallProcess', 'process_evt_detect', sFiles{iStage}, [], ...
                'eventname',    'bad_HighStd_EMG', ...             
                'channelname', emgChan, ...
                'timewindow',   [1 30], ...
                'bandpass',     [10 35], ...
                'threshold',    std_tresh, ...
                'blanking',     0.5, ...
                'isnoisecheck', 0, ...
                'isclassify',   0);

    %         sFiles{iStage} = FlagBadChannel_STD(sFiles{iStage},ChannelStruct,std_percent_thresh);
                PrintProcessStatusPerEpoch(sFiles{iStage},resFiles);
        end
        timeLapsed = [timeLapsed; toc];
        end




        %% REMOVE 1-7Hz ARTIFACTS
        %%%%%%%%%%%%%%%

        % Run 1 remove artifact with PCA | Run 2 mark as bad
        for iRun=1:2
            % Process: Detect other artifacts
            fprintf('Detecting other artifacts: 1-7Hz ...\n');tic
            for iStage=1:length(sFiles)
                fprintf('    Stage %s\n',sleepStages{iStage})
                if isempty(sFiles{iStage})
                    fprintf('Skipping. No epoch found in stage %s.\n',sleepStages{iStage})
                    continue
                end

                resFiles = bst_process('CallProcess', 'process_evt_detect_badsegment', sFiles{iStage}, [], ...
                    'timewindow',  [], ...
                    'sensortypes', 'EEG', ...
                    'threshold',   3, ...  % 3
                    'isLowFreq',   1, ...   % detect 1-7 Hz artifact: eye movement/muscular
                    'isHighFreq',  0);      % detect 40-240 Hz artifacts: electrodes & sensors artifacts

                % First run: remove event using PCA and detect again for residual
                if iRun == 1 && ~isempty(resFiles)
                    % Process: Rename event
                    fprintf('        Renaming events 1-7Hz to eye_movement\n')
                    artFiles = bst_process('CallProcess', 'process_evt_rename', resFiles, [], ...
                        'src',  '1-7Hz', ...
                        'dest', 'eye_movement');

                    if ~isempty(artFiles)
                        % Process: SSP artifact_EOG_EMG: 1-7Hz
                        fprintf('    Removing 1-7Hz artifact in stage %s\n',sleepStages{iStage})
                        resFiles = bst_process('CallProcess', 'process_ssp', artFiles,[], ...
                            'timewindow',  [], ...
                            'eventname',   'eye_movement', ...
                            'eventtime',   [-0.2, 0.2], ...
                            'bandpass',    [1, 7], ...
                            'sensortypes', 'EEG', ...
                            'usessp',      1, ...
                            'saveerp',     0, ...
                            'method',      1, ...  % PCA: One component per sensor
                            'select',      1);
                        % for SSP2 add those 2 lines as parameters to the preceding function
    %                         'nicacomp',    0, ...
    %                         ... 'ignorebad',   1, ...

                        PrintProcessStatusPerEpoch(artFiles,resFiles);
    %                     resFiles = CleanAndUpdateFiles(artFiles,resFiles);
                    else
                        fprintf(2,'ERROR renaming 1-7Hz artifacts... Files uncorrected!\n');
                    end
                elseif iRun == 2
                    % Process: Rename event
                    fprintf('        Renaming events 1-7Hz to BAD_1-7Hz\n')
                    resFiles = bst_process('CallProcess', 'process_evt_rename', resFiles, [], ...
                        'src',  '1-7Hz', ...
                        'dest', 'bad_1-7Hz');
                end

                sFiles{iStage} = CleanAndUpdateFiles(sFiles{iStage},resFiles);
            end
        end
        timeLapsed = [timeLapsed; toc];
        db_save_bak;
    end
    
    
    %% SPECTRAL ANALYSIS
    %%%%%%%%%%%%%%%
    if ~isempty(PARAM.Analysis.spectral)
        [sCondFiles,PARAM] = SpectralAnalysis(sFiles,PARAM);
        if isempty(PARAM.PSD)
            fprintf(2,'PREP> ERROR: Could not process Power Spectrum Density\n')
            continue
%        else
%            sFiles = resFiles;
%     		PARAM.PSD = [PARAM.PSD [resFiles{:}]];
        end
    end
    

    %% TIME-FREQUENCY ANALYSIS
    %%%%%%%%%%%%%%%
    if ~isempty(PARAM.Analysis.time_frequency)
        % Process: Time-frequency (Morlet wavelets)
        sFiles = bst_process('CallProcess', 'process_timefreq', sFiles, [], ...
            'sensortypes', 'EEG', ...
            'edit',        struct(...
                 'Comment',         'Avg,Power,FreqBands', ...
                 'TimeBands',       [], ...
                 'Freqs',           {{'delta', '0.3, 4', 'mean'; 'theta', '4, 8', 'mean'; 'alpha', '8, 12', 'mean'; 'sigma', '12, 16', 'mean'; 'beta', '15, 32', 'mean'; 'gamma1', '30, 90', 'mean'}}, ...
                 'MorletFc',        1, ...
                 'MorletFwhmTc',    3, ...
                 'ClusterFuncTime', 'none', ...
                 'Measure',         'power', ...
                 'Output',          'average', ...
                 'RemoveEvoked',    0, ...
                 'SaveKernel',      0), ...
            'normalize',   'none');  % None: Save non-standardized time-frequency maps
    end
end
% Display total processing time
durSec = sum([PARAM.StepDuration{:,2}]);
durMin = durSec / 60;
fprintf('PREP>_____ FINISHED in %.2f seconds (%.2f minutes)\n',durSec,durMin)

% Save PARAM structure in the "results" folder
PARAM.FileName = [PARAM.SaveDir filesep ...
    'PREP_PARAM_', datestr(now,'dd-mmm-yyyy_HH-MM-SS')];
fprintf('PREP> Saving parameters to: %s\n',PARAM.FileName)
try
    save(PARAM.FileName,'PARAM');
catch ME
    if strcmp(ME.identifier,'MATLAB:save:noParentDir')
        [p,~] = fileparts(PARAM.FileName);
        mkdir(p)
        save(PARAM.FileName,'PARAM');
    else
        rethrow(ME)
    end
end

%% EXPORT PSD TO EXCEL
if 0
    PSD2Excel();
end

