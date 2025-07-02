function [sFiles, PARAM] = ImportRecordings(sFiles,PARAM)
%IMPORTRECORDINGS - Simple wrapper to make code cleaner
%
% SYNOPSIS: [sFiles, PARAM] = ImportRecordings(sFiles,PARAM)
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
% Created on: 15-Aug-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic

 % Get Brainstorm subject
sSubject = GetSubject(PARAM.currentSubject, ...
    PARAM.Brainstorm.Subjects, ...
    PARAM.Brainstorm.UseDefaultAnat, ...
    PARAM.Brainstorm.UseDefaultChannel);

% Get raw files from this subject
[sFiles, ~] = ImportRawFiles(sSubject,sFiles);

% Log step duration
PARAM.StepDuration = AddStepDuration(PARAM.StepDuration, ...
    [PARAM.currentSubject ' | ImportRecordings'],toc);
