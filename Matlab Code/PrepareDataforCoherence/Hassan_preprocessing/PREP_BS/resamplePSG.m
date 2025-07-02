function resFiles = resamplePSG(sFiles,PARAM)
%RESAMPLEPSG - Resample recording
%
% SYNOPSIS: resFiles = resamplePSG(sFiles,PARAM)
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
% Created on: 25-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Process: Resample: 100 Hz > make processing faster
fprintf('Resampling...\n');%tic
if iscell(sFiles)
    for iStage=1:length(sFiles)
        fprintf('    Stage %s\n',sleepStages{iStage})

        if isempty(sFiles{iStage})
            fprintf('Skipping. No epoch found in stage %s.\n',sleepStages{iStage})
            continue
        end

        filteredIdx = cellfun(@(c) contains(c,'resample'),{sFiles{iStage}.FileName});
        if ~any(filteredIdx)
            if isempty(sFiles{iStage})
                fprintf('Skipping. No epoch found in stage %s.\n',sleepStages{iStage})
                continue
            end

            resFiles = bst_process('CallProcess', 'process_resample', sFiles{iStage}, [], ...
                'freq',     100, ...
                'read_all', 1);

            % Continue with new file links if process worked
            sFiles{iStage} = CleanAndUpdateFiles(sFiles{iStage},resFiles);
        end
    end
%     timeLapsed = [timeLapsed; toc];
    db_save_bak;
else
    resFiles = bst_process('CallProcess', 'process_resample', sFiles, [], ...
                'freq',     PARAM.ResampleFreq, ...
                'overwrite', 1);
end