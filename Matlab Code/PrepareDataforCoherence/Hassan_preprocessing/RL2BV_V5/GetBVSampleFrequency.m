function [Fs,startDateTime,recFileName] = GetBVSampleFrequency(fullFileName)
%GETBVSAMPLEFREQUENCY - Get sample frequency of the recording from vhdr or EDF (.rec) file
%
% SYNOPSIS: Fs = GetBVSampleFrequency(fname)
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
% Created on: 06-Jun-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Avoid exit error for undefined output argument
startDateTime = [];
recFileName = [];
Fs = [];

if ~nargin
    [fname,fpath] = uigetfile({'*.vhdr','BrainVision header file';'*.rec','EDF/EDF+'});
    fullFileName = fullfile(fpath,fname);
end
[~,~,fExt] = fileparts(fullFileName);

switch lower(fExt)
    case '.vmrk'
        % Change file extension
        fullFileName = strrep(fullFileName,fExt,'.vhdr');
        % Try again
        [Fs,startDateTime,~] = GetBVSampleFrequency(fullFileName);
    case '.vhdr'
        fprintf('RL2BV>     Reading BrainVision header file...\n')
        % Replace isfile(fullFileName) by the 3 following line to avoid Matlab older than 2017a error
        fid = fopen(fullFileName);
        if fid ~= -1 %isfile(fullFileName)
            fclose(fid);
            % read entire file in a single string
            str = fileread(fullFileName);
            
            % Get recording sample frequency
            %---------------
            key = 'Sampling Rate [Hz]: ';   % pattern to look for...
            idx = strfind(str,key) + length(key); % start index of character value
            Fs = sscanf(str(idx:end), '%g', 1); % get the value

            if isempty(idx)
                key = 'SamplingInterval=';
                idx = strfind(str,key) + length(key); % start index of character value
                Ts = sscanf(str(idx:end), '%g', 1);
                Fs = 1 / (Ts * 1e-6); % get the value
            end
        else
            fprintf('RL2BV>     WARNING: File %s could not be opened\n',fullFileName)
            fprintf('RL2BV>     Trying EDF/EDF+ (.rec) instead...\n')
            % Change file extension to edf/rec
            if exist(strrep(fullFileName,fExt,'.rec'),'file')
                recFileName = strrep(fullFileName,fExt,'.rec');
            elseif exist(strrep(fullFileName,fExt,'.edf'),'file')
                recFileName = strrep(fullFileName,fExt,'.edf');
            end
            % Try again
            [Fs,startDateTime,~] = GetBVSampleFrequency(recFileName);
        end
    case {'.rec', '.edf'}
        fprintf('RL2BV>     Reading EDF/EDF+ file...\n')
        [hdr,~] = edfread(fullFileName,'targetSignals',1);
        Fs = hdr.frequency(7); % change to 7
        dtNum = datenum([hdr.startdate '.' hdr.starttime],'dd.mm.yy.HH.MM.SS');
        startDateTime = datetime(dtNum,'Format','yyyyMMddHHmmssSSSSSS','ConvertFrom','datenum'); % datetime of begining of recording
        recFileName = fullFileName;
end
    
end