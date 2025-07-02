function AddRemLogicEvt2VMRK(fullFileName,evt)
%ADDREMLOGICEVT2VMRK - Append event markers from RemLogic to BrainVision marker file
%
% SYNOPSIS: AddRemLogicEvt2VMRK(events)
%
% Required files: same name pair of .vhdr .vmrk in the same folder
%                   Remlogic event file X.txt
%
% EXAMPLES:
%
% REMARKS:
%
% See also readRemLogicEvtFile

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Created with:
%   MATLAB ver.: 9.5.0.1049112 (R2018b) Update 3 on
%    Linux 5.0.10-arch1-1-ARCH #1 SMP PREEMPT Sat Apr 27 20:06:45 UTC 2019 x86_64
%
% Author:     Tomy Aumont
% Work:       University of Montreal
% Email:      tomy.aumont@umontreal.ca
% Website:    
% Created on: 09-May-2019
% Revised on:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Check Input Argument
%%%%%%%%%%%%%%%%
if nargin == 0
    [~,evt] = readRemLogicEvtFile;
end

% Select and read BrainVision file
%---------------
if nargin < 2
    [fName,fPath] = uigetfile('*.vmrk','Select BrainVision file');
    fullFileName = fullfile(fPath,fName);
end

% Get sample frequency, recording start date and time and recording filename
if endsWith(fullFileName,'.mff','IgnoreCase',true) % introduced in r2016b
    % EGI PHILIPS
    [Fs,dt,recFileName,fullFileName] = f_GetMFFSampleFrequency(fullFileName);
    
elseif endsWith(fullFileName,{'.vmrk','.vhdr'},'IgnoreCase',true) % introduced in r2016b
    % BRAINVISION
    [Fs,dt,recFileName] = GetBVSampleFrequency(fullFileName);
    if endsWith(fullFileName,{'.edf','.rec'},'IgnoreCase',true)
        fullFileName = [fullFileName(1:end-3), 'vmrk'];
    end
elseif endsWith(fullFileName,{'.edf','.rec'},'IgnoreCase',true) % introduced in r2016b
    % EDF
    [Fs,dt,recFileName] = GetBVSampleFrequency(fullFileName);
    fullFileName = [fullFileName(1:end-3), 'vmrk'];
end

% Open and read VMRK file
%---------------
fId = fopen(fullFileName);
if fId == -1
    fprintf('RL2BV>     Creating marker file...\n');
    % Write file header and begining of recording
    WriteVMRKHeader(fullFileName,recFileName,dt);
    Nmk = 1; % marker of recording start
    toBckup = 0;
else
    str = fileread(fullFileName); % read entire file in a single string
    fclose(fId);
    % Get number of existing markers
    key = 'Mk';   % pattern to look for...
    idx = strfind(str,key) + length(key); % start indexes of last "Mk" occurences + 2
    % get last marker number (number of existing markers)
    if isempty(idx); Nmk = 0; % not marker existing
    else; Nmk = sscanf(str(idx(end):end), '%g', 1);
    end
    % Get recording start datetime
    idx = strfind(str,'Mk1=New Segment'); % Marker for begining of recording
    readLen = length(str) - idx; if readLen > 100; readLen = 100; end
    lines = regexp(regexp(str(idx:idx+readLen),'\r\n|\r|\n', 'split')',',','split'); % split lines into cell array
    dt = datetime(lines{1}{end},'Format','yyyyMMddHHmmssSSSSSS'); % datetime of begining of recording
    toBckup = 1;
end
 
%% Backup file before modification
%%%%%%%%%%%%%%%%
if toBckup
    BackupFileWithNumber(fullFileName);
end

%% Append new markers to existing marker file
%%%%%%%%%%%%%%%%
fId = fopen(fullFileName,'a');
if fId == -1
    fclose(fId);
    error('ERROR: File %s could not be opened',fullFileName)
end

fprintf('RL2BV>     Appending file %s with new markers...\n',fullFileName)

tCol   = contains(evt(1,:),'Time');         % Column index of event time
eCol   = contains(evt(1,:),'Event');        % Column index of event name
dCol   = contains(evt(1,:),'Duration');     % Column index of event duration
epCol  = contains(evt(1,:),'Epoch');        % Column index of epoch number
slpCol = contains(evt(1,:),'Sleep Stage');  % Column index of sleep stages

evtTime = dt; % init event time at the beginning of the recording
epLen = 30; % in sec, length of a scoring epoch
nSkippedEvt = 0;
for iEvt=2:size(evt,1) % starts at 2 to avoid header row
    
    % ===== GET EVENT NAME =====
    if any(slpCol)
        evtName = evt{iEvt,slpCol};
        if isempty(evtName) || strcmpi(evtName,'N/A')
           if any(eCol)
               evtName = evt{iEvt,eCol};
           end
        end
    elseif any(eCol)
        evtName = evt{iEvt,eCol};
    else
        nSkippedEvt = nSkippedEvt + 1;
        continue
    end
    % if empty or N/A, skip to next marker
    if isempty(evtName) || strcmpi(evtName,'N/A')
        nSkippedEvt = nSkippedEvt + 1;
        continue
    else
        evtName = upper(evtName);            % capitalize event name for easier detection
        iMrk = Nmk + iEvt - nSkippedEvt - 1; % Compute marker index position
    end
    
    % ===== CONVERT MARKER TIME TO DATETIME FORMAT =====
    if any(tCol)
        % Append real time to datetime format
        evtTime = datetime([evtTime.Year,evtTime.Month,evtTime.Day, ...
            str2double(split(evt{iEvt,tCol},':'))']);
        % Adjust date if event time is smaller than start time
        if evtTime.Hour < dt.Hour && evtTime.Day == dt.Day
            evtTime.Day = evtTime.Day + 1;
        end
        
        % get number of second between begining of recording and marker
        nbrSecondFromStart = seconds(evtTime-dt);
    else
        % PROCESS MARKER TIME FROM EPOCH NUMBER
        iEpoch = str2double(evt{iEvt,epCol});
        nbrSecondFromStart = iEpoch * epLen; % in seconds
        
    end
    
    % ===== GET EVENT DURATION IN SAMPLES =====
    if any(dCol)
        mkSz = str2double(evt{iEvt,dCol}) * Fs;
    else
        mkSz = 1;%epLen * Fs;
    end
    
    % ===== COMPUTE MARKER POSITION IN SAMPLES =====
    % reference: https://pressrelease.brainproducts.com/markers/
    pos = round(nbrSecondFromStart * Fs) + 1;
    
    % ===== APPEND MARKER TO FILE =====
    fprintf(fId,'Mk%d=Event,%s,%d,%d,%d\n',iMrk,evtName,pos,mkSz,0);
end
fclose(fId);
fprintf('RL2BV>     Done\n')
