function updatedFiles = CleanAndUpdateFiles(oldFiles,newFiles)
%CLEANANDUPDATEFILES - Delete oldFiles from database and replace them by newFiles
%
% SYNOPSIS: newFile = CleanAndUpdateFiles(oldFiles,newFiles)
%
% Required files:
%
% EXAMPLES:
%   sFiles{iStage} = CleanAndUpdateFiles(sFiles{iStage},resFiles)
%
% REMARKS:
%
% See also CreateMAPS

%
% Copyright Tomy Aumont

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
% Created on: 19-Jun-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isempty(newFiles)
    if length(newFiles) == length(oldFiles)
        % Remove files that changed
        toRemove = ~ismember({oldFiles.FileName},{newFiles.FileName});
        if any(toRemove)
            % Process: Delete folders: Remove old folder that are not in new
            oldFiles = bst_process('CallProcess', 'process_delete', oldFiles(toRemove), [], ...
                'target', 1);  % Delete folders
            % Update list of active files
            updatedFiles = newFiles;
        else
            % All files are the same, do not remove any.
            updatedFiles = oldFiles;
        end
    else
        updatedFiles = oldFiles;
        
        newF = find(~ismember({newFiles.FileName},{oldFiles.FileName})); % 1= fichier qui n'est pas membre des vieux
        % Remove files that changed
         [~,fn,~] = cellfun(@(c) fileparts(c), {oldFiles.FileName}, 'UniformOutput', false);    % fn nom sans extension des vieux fichiers
         toReplace = [];
         toReplace = find(contains({newFiles(newF).FileName},fn)); % vieux a remplace
         updatedFiles(toReplace) = newFiles(newF); % update file
%             
%          for iNewFile=1:length(newF)
%             toReplace(iNewFile) = find(cellfun(@(c) contains({newFiles(newF).FileName},c),fn)); % vieux a remplace
%             updatedFiles(toReplace(iNewFile)) = newFiles(newF(iNewFile)); % update file
%          end
         
         if ~isempty(toReplace)
            % Process: Delete folders: Remove old folder that are now replaced
            oldFiles = bst_process('CallProcess', 'process_delete', oldFiles(toReplace), [], ...
                'target', 1);  % Delete files
         end
    end
    
%     % Process: Delete folders
%     oldFiles = bst_process('CallProcess', 'process_delete', oldFiles, [], ...
%         'target', 2);  % Delete folders
%     updatedFiles = newFiles; % Update with new file links
else
    fprintf(2,'    ERROR: Keeping files unchanged.\n')
    updatedFiles = oldFiles;
end
