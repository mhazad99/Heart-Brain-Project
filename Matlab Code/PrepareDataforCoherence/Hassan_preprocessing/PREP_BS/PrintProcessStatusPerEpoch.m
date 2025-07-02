function PrintProcessStatusPerEpoch(sFiles,resFiles)
%PRINTPROCESSSTATUSPEREPOCH - Print the file for which the process didn't work 
%   in yellow and the one for which it worked in green
%
% SYNOPSIS: PrintProcessStatusPerEpoch(sFiles,resFiles)
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
% Created on: 20-Jun-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(resFiles)
    % All epoch not processed
    errIdx = 1:length(sFiles);
elseif length(resFiles) ~= length(sFiles)
    % Not all of them... list bad ones
    errIdx = ~ismember({sFiles.FileName},{resFiles.FileName});
else
    % All epoch successfully processed
    errIdx = zeros(1,length(sFiles));
end
errList = {sFiles(errIdx==1).FileName};
goodList = {sFiles(errIdx==0).FileName};

% print the list of files unprocessed
for iEpoch=1:length(errList)
    if iEpoch == 1
        cprintf('Yellow','Could not process %d out of %d epoch:\n',length(errList),length(sFiles))
    end
    cprintf('Yellow','    %s\n',errList{iEpoch})
end
    
% print the list of processed files
for iEpoch=1:length(goodList)
    if iEpoch == 1
        cprintf('Comments','Successfully process %d out of %d epoch:\n',length(goodList),length(sFiles))
    end
    cprintf('Comments','    %s\n',goodList{iEpoch})
end

end