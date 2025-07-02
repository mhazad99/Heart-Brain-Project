function WaitChannelReject(PARAM)
% SYNOPSIS: WaitChannelReject()
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


% if ~isempty(PARAM.Channel.reject_visual)
    disp('PREP> TIME TO REJECT CHANNELS VISUALLY IN BRAINSTORM!')
    brainstorm; % start brainstorm GUI here
    CreateStruct.Interpreter = 'tex';
    CreateStruct.WindowStyle = 'modal';
    uiwait(msgbox({'Reject bad channels using Brainstorm interface now.'; '';
        'Click \bfOK\rm to continue the pre-processing'}, 'Reject bad channels','help',CreateStruct));
% end
    