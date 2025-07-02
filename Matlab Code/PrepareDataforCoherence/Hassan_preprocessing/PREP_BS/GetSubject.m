function sSubject = GetSubject(subjNames, subjList, defaultAnat, defaultChannel)
%GETSUBJECT - Check if subject exists in subject list. If yes, get it, else create it.
%
% SYNOPSIS: sSubject = GetSubject(subjectNames, subjectList, UseDefaultAnat, 
%              UseDefaultChannel);
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
% Created on: 11-Jul-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~ismember(subjNames,{subjList.Subject.Name})
    % new subject: create it
    [sSubject, ~] = db_add_subject(subjNames, [], defaultAnat, defaultChannel);
    if isempty(sSubject)
        fprintf(2,'ERROR: Could not create subject %s\n',subjNames);
    end
else
    % subject already exists: load it
    sSubject = bst_get('Subject', subjNames);
end
