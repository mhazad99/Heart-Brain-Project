function PARAM = SetupBrainstormEnv(PARAM)
%SETUPBRAINSTORMENV - Load a protocol and return its list of subjects
%
% SYNOPSIS: PARAM = SetupBrainstormEnv(PARAM)
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
% Created on: 22-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('PREP>_____SETTING BRAINSTORM ENVIRONMENT...')

% make sure to return something (avoid error)
iProtocol = [];

% Get parameters to ease reading
protocol = PARAM.Brainstorm.protocol;
UseDefaultAnat = PARAM.Brainstorm.UseDefaultAnat;
UseDefaultChannel = PARAM.Brainstorm.UseDefaultChannel;

% Start brainstorm if not already running
if ~brainstorm('status')
    brainstorm nogui
end


if isempty(protocol)
    % No protocol given: ask user to select one
    lsdir = dir(bst_get('BrainstormDbDir'));
    lsdir = { 'new protocol', lsdir(3:end).name}; % 3 to avoid '.' and '..' directories

    [iSelect,tf] = listdlg('PromptString','Select a protocol:', ...
        'ListString',lsdir, ...
        'SelectionMode','single', ...
         'InitialValue',1, ...
         'Name','Protocol Selection');
   
     if ~tf; fprintf('PREP> User cancelled protocol selection. Exit\n'); return; end
    
    if iSelect == 1
        % Get the new protocol name
        protocol = char(inputdlg('Create protocol','Protocol name:',[1 50]));
        while any(contains(lsdir, protocol,'IgnoreCase',true)) && ~isempty(protocol)
            f = msgbox('Protocol already exists. Please enter a new protocol name.', ...
                'Error','error');
           uiwait(f);
           protocol = char(inputdlg('Create protocol','Protocol name:',[1 50]));
        end
    else
        % Set protocol name
        protocol = lsdir(iSelect);
    end 
end
    
if isempty(protocol); error('Protocol name can not be empty.'); end

% Get index of "ProtocolName"
iProtocol = bst_get('Protocol', protocol);

if isempty(iProtocol)
    fprintf('PREP> Creating protocol: %s\n', protocol);
    % Create new protocol
    gui_brainstorm('CreateProtocol', protocol, ...
        UseDefaultAnat, ...
        UseDefaultChannel);
    % Reset colormaps
    bst_colormaps('RestoreDefaults', 'eeg');
end
% Select/load the current procotol
gui_brainstorm('SetCurrentProtocol', iProtocol);
% Get subjects from that protocol
PARAM.Brainstorm.Subjects = bst_get('ProtocolSubjects');
