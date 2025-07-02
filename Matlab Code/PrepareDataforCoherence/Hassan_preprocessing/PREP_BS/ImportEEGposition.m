function [] = ImportEEGposition(sFiles,filepath)
%------------------------------------------
% ImportEEGposition = Import EEG coordinates from text file in brainstorm 
%                     Set non EEG channel types to their appropriate Channel type
%
% SYNOPSIS: ImportEEGposition(sFiles,filepath)
%
% Required files: Text file containing coordinates 
%
%
%------------------------------------------
% Process: Add EEG positions from a text file
sFiles = bst_process('CallProcess', 'process_channel_addloc', sFiles, [], ...
    'channelfile',  {filepath,'ASCII_NXYZ'}, ...
    'usedefault',  1, ...  % 21 :Colin27: BrainProducts EasyCap 128
    'fixunits',    1, ...
    'vox2ras',     1);

%Make sure non EEG channel types are set to their appropriate Channel type
% List : 'EOG1','EOG2','EMG1','EMG2','EOG3','ECG'
sFiles = bst_process('CallProcess', 'process_channel_settype', sFiles, [], ...
    'sensortypes', 'EOG1,EOG2,EOG3', ...
    'newtype',     'EOG');

sFiles = bst_process('CallProcess', 'process_channel_settype', sFiles, [], ...
    'sensortypes', 'EMG1,EMG2', ...
    'newtype',     'EMG');

sFiles = bst_process('CallProcess', 'process_channel_settype', sFiles, [], ...
    'sensortypes', 'ECG', ...
    'newtype',     'ECG');

bst_report('Save', sFiles);


