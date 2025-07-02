function [sRawFiles, sStudies] = ImportRawFiles(sSubject,rawFiles)
%IMPORTRAWFILES - Get subject studies from database and create or load the raw recording file.
%
% SYNOPSIS: [sRawFiles, sStudies] = ImportRawFiles(sSubject,rawFiles);
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

[~,fname,fExt] = fileparts(rawFiles{1});
sStudies = bst_get('StudyWithSubject', sSubject.FileName);

if ~ismember(['@raw' fname],{sStudies.Name})
    % Get format to read
    switch fExt
        case '.eeg'
            dataFileType = 'EEG-BRAINAMP';
            evtFileType = 'BRAINAMP';
        case {'.edf' '.EDF' '.REC'}
            dataFileType = 'EEG-EDF';
            evtFileType =  'BRAINAMP'; % doit faire la conversion de RemLogic vers vmrk
        otherwise
                fprintf(2,'ERROR: Unrecognized file type.\n')
                return
    end
    
    % Start a new report
    bst_report('Start', rawFiles{1});
    
    % Process: Create link to raw file
    disp('PREP>_____IMPORTING RAW FILES...') %tic
    sRawFiles = bst_process('CallProcess', 'process_import_data_raw',[] ,[], ...
        'subjectname',    sSubject.Name, ...
        'datafile',       {rawFiles{1}, dataFileType}, ...
        'channelreplace', 1, ...
        'channelalign',   1, ...
        'evtmode',        'value');

       
    % Process: Import events to recording
    if ~isempty(evtFileType)
        % Start a new report
        bst_report('Start',sRawFiles);
        sRawFiles = bst_process('CallProcess', 'process_evt_import', sRawFiles, [], ...
            'evtfile', {rawFiles{2}, evtFileType}, ...
            'evtname', '');
        % Save and display report
        ReportFile = bst_report('Save', sRawFiles);
    end

 
    if 0 % MISSING EMG, ECG, EOG
        
%     testing = 'D:\MAPS_Files_original\Localizer_Files_RR\MP2_0146_V3_T1.txt';
    
%    Process: Set channel file
    sRawFiles = bst_process('CallProcess', 'process_import_channel', sRawFiles, [], ...
        'channelfile',  {testing,'ASCII_NXYZ'}, ...
        'usedefault',   1, ...  % 21" Colin27: BrainProducts EasyCap 128
        'channelalign', 0, ...
        'fixunits',     1, ...
        'vox2ras',      0);

    % Process: Add "DEFAULT" EEG positions
     
    sRawFiles = bst_process('CallProcess', 'process_channel_addloc', sRawFiles, [], ...
        'channelfile',  {testing,'ASCII_NXYZ'}, ...
        'usedefault',  1, ...  % 21 :Colin27: BrainProducts EasyCap 128
        'fixunits',    1, ...
        'vox2ras',     0);
    end
    
    
    
    % Save and display report
    ReportFile = bst_report('Save', sRawFiles);
    
else
    % Load link to raw file
    disp('PREP>_____LOAD EXISTING RAW FILES...')
    sFileIdx = ismember({sStudies.Name},['@raw' fname]);
    sRawFiles = bst_process('GetInputStruct', sStudies(sFileIdx).Data.FileName);
end