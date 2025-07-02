function [hdr,evt] = readRemLogicEvtFile(fullFileName,convXML)
%READREMLOGICEVTFILE - Read event text file exported by RemLogic into a cell array
%
% SYNOPSIS: [hdr,evt] = readRemLogicEvtFile()
%               hdr contains all experiment information. evt contains events
%               timestamp, name, duration. Its first line is the column names.
%
% Required files:
%
% EXAMPLES:
%       [hdr,evt] = readRemLogicEvtFile() Ask to select a file with the GUI and
%           read tis header in a single string 'hdr' and events, including first line as
%           header in evt. Behave like when convXML is false.
%       [hdr,evt] = readRemLogicEvtFile(fullFileName) Read fullFileName
%           header in a single string 'hdr' and events, including first line as header
%           in evt. Behave like when convXML is false.
%       [hdr,evt] = readRemLogicEvtFile(fullFileName,XML) Read fullFileName
%           header in a single string 'hdr' and events, including first line as header
%           in evt. If convXML is true, convert sleep stages to number, else
%           keep them as is.
%
% REMARKS:
%
% See also 

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

%% CHECK INPUT VALIDITY
%%%%%%%%%%%%%%%%
if nargin > 0
    if ~ischar(fullFileName)
        fprintf(2,'RL2BV>     ERROR: First input must be a char array or string.')
        return
    end
    
    if nargin == 2
        if ~islogical(convXML)
            fprintf(2,'RL2BV>     ERROR: second argument must be logical');
            return
        end
    end
end

%% SELECT FILE TO READ
%%%%%%%%%%%%%%%%
if nargin == 0
    [evtFile, p] = uigetfile('*.txt','Select RemLogic event file');
    fullFileName = fullfile(p,evtFile);
end

% Check if valid file
fid = fopen(fullFileName);
if fid == -1; error('ERROR: File %s could not be opened',fullFileName); end

%% READ HEADER
%%%%%%%%%%%%%%%%
fprintf('RL2BV>     Reading scoring file | %s\n',fullFileName)

hdr = fullFileName;
% Read and split header lines at every tabulation
% rawData = regexp(fgetl(fId),'\t','split');
rawData = fgetl(fid);
while ~((any(contains(rawData,'Time')) && ...
        any(contains(rawData,'Event')) && ...
        any(contains(rawData,'Duration'))) ...
        || ...
        (any(contains(rawData,'Epoch')) && ...
        any(contains(rawData,'Event'))) ...
        || ...
        (any(contains(rawData,'Sleep Stage')) && ...
        any(contains(rawData,'Timee')) && ...
        any(contains(rawData,'Event')) && ...
        any(contains(rawData,'Duration'))) )

    hdr = [hdr newline rawData];
    rawData = fgetl(fid);
end

%% READ EVENTS
%%%%%%%%%%%%%%%%
% Read event per epoch
evt = regexp(rawData,'\t','split'); % Keep header
if any(contains(rawData,'Epoch'))
    while ~feof(fid)
        % remove space in event name
        tmpInfo = strrep(regexp(fgetl(fid),'\t','split'),' ','');
        % Pad missing column data with empty cell
        nMissingInfo = size(evt,2) - length(tmpInfo);
        tmpInfo = [tmpInfo repmat({''},1,nMissingInfo)];
        % Add new event data in event structure 'evt'
        evt(end+1,:) =  tmpInfo;
    end
    fclose(fid);
else
    % Read event with timestamp
    while ~feof(fid)
        evt(end+1,:) = strrep(regexp(fgetl(fid),'\t','split'),' ','');
    end
    fclose(fid);
end

% Convert stage name to number
% ---------------
formatNames = 'Yes';
if nargin == 2
    if convXML; con2XML = 'Yes';
    else; con2XML = 'No';
    end
else
    question  = 'Do you want to convert sleep stage for XML processing?';
    con2XML = questdlg(question,'Convert timestamp','Yes','No','No');
%     if strcmp(con2XML,'No')
%         question  = 'Do you want to convert sleep stage name from SLEEP-S# to N1,N2,N3,R,W?';
%         formatNames = questdlg(question,'Convert timestamp','Yes','No','No');
%     end
end
% Convertion to XML...
if strcmp(con2XML,'Yes')
    evtCol = contains(evt(1,:),'Event','IgnoreCase',true);
    for i=2:size(evt,1)
       switch evt{i,evtCol}
           case {'R','REM'}
               evt{i,evtCol} = 5;
           case {'W','WAKE'}
               evt{i,evtCol} = 0;
           case {'N1','NREM1'}
               evt{i,evtCol} = 1;
           case {'N2','NREM2'}
               evt{i,evtCol} = 2;
           case {'N3','NREM3'}
               evt{i,evtCol} = 3;
       end
    end
end
% Convertion to standard names
if strcmp(formatNames,'Yes')
    evtCol = contains(evt(1,:),'Event','IgnoreCase',true);
    for i=2:size(evt,1)
       switch upper(evt{i,evtCol})
           case {'SLEEP-REM'}
               evt{i,evtCol} = 'R';
           case {'SLEEP-S0'}
               evt{i,evtCol} = 'W';
           case {'SLEEP-S1'}
               evt{i,evtCol} = 'N1';
           case {'SLEEP-S2'}
               evt{i,evtCol} = 'N2';
           case {'SLEEP-S3'}
               evt{i,evtCol} = 'N3';
           case {'SLEEP-UNSCORED'}
               evt{i,evtCol} = 'N/A';
       end
    end
end


% Convert timestamp
% ---------------
timeColumn = find(contains(evt(1,:),'Time','IgnoreCase',true));
if any(timeColumn)
    newTimestamp = cellfun(@(x) datestr(x,'HH:MM:SS.FFF'),evt(2:end,timeColumn),'UniformOutput',0);
    evt(2:end,timeColumn) = newTimestamp;
end

end