function sFiles = CheckEOG(sFiles)
%CHECKEOG - Check if EOG type is well assign to corresponding electrodes
%
% SYNOPSIS: resFiles = CheckEOG(sFiles)
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
% Created on: 30-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

isSave = false;
chan = in_bst_channel(sFiles.ChannelFile);

iEOG = find(contains({chan.Channel.Name},{'E1','E2','E3'}));
for iChan = 1:length(iEOG)
    if ~strcmpi(chan.Channel(iEOG(iChan)).Type,'EOG')
        chan.Channel(iEOG(iChan)).Type = 'EOG';
        isSave = true;
    end
end

if isSave
    fullChanPath = file_fullpath(sFiles.ChannelFile);
    fprintf('PREP> Updating EOG channel type in: %s\n',fullChanPath)
    db_set_channel(sFiles.iStudy,chan,2); % overwrite channel file
    sStudies = bst_get('Study');
    sFiles = bst_process('GetInputStruct', sStudies.Data.FileName);
end



