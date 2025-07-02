function sFiles = FlagBadChannel_STD(sFiles,sChannels,std_percent_thresh)
% SYNOPSIS: FlagBadChannel_STD()
%
% Required files:
%
% EXAMPLES:
%
% REMARKS:
%
% See also 

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
% Created on: 11-Jun-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


sChan = {sChannels.Channel.Name};
mrkCount = zeros(size(sChan));
disp('Looking for bad channels...')

for i=1:length(sFiles)
    tmp = in_fopen(sFiles(i).FileName, 'BST-DATA');
    mrk = find(contains({tmp.events.label},'HighStd'));
    % Count number of marker per channels
    if ~isempty(mrk)
        
        iEvt = find(~cellfun(@isempty,{tmp.events(mrk).channels}));
        mrk = mrk(iEvt);
        for j=1:length(mrk)
            mrkMask = ~cellfun(@isempty,tmp.events(mrk(j)).channels);
            if any(mrkMask)
                toAdd = ismember(sChan,[tmp.events(mrk(j)).channels{mrkMask}]);
                mrkCount = mrkCount + toAdd;
            end
        end
    end
end

isBad = find(mrkCount > std_percent_thresh * length(sFiles));

disp('Marking channel as bad...')

badChan = strjoin({sChannels.Channel(isBad).Name},',');

% Process: Set bad channels     |   QUITE SLOW
sFiles = bst_process('CallProcess', 'process_channel_setbad', sFiles, [], ...
    'sensortypes', badChan);




% for i=1:length(sFiles)
%     tmp = in_fopen(sFiles(i).FileName, 'BST-DATA');
%     tmp.channelflag(isBad) = -1;
% 
%     % May need to mark it in another file under structure F.ChannelFlag but not found
%     tmp = in_fopen(tFileName, 'BST-DATA');
% end
