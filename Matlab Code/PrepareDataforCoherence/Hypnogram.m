function [StartTime, EndTime, y, idx, FinalLen] = Hypnogram(TextfileName, folderPath)

%Uncomment it when you want to work with control cases!
%T = readtable(TextfileName,'VariableNamingRule','preserve', 'HeaderLines',13);

% Uncomment it when you want to work with Depression cases!
T = readtable(TextfileName,'VariableNamingRule','preserve');
%T(5,:) = [];
Remainingrows = T.Event;
ValidElements = {'Wake','NREM 1','NREM 2','NREM 3','REM'};
list = ismember(Remainingrows, ValidElements);
T(~list, :) = [];
T.Duration(:) = 30;
writetable(T, TextfileName, 'Delimiter','\t')
StartTime = timeofday(T.("Start Time")(1));
EndTime = timeofday(T.("Start Time")(end));

%T = readtable(TextfileName,'VariableNamingRule','preserve');
T.Event = string(T.Event);
[numRows,numCols] = size(T);
i = 1;
y = []; % array to save each sleep stage
for item = 1: numRows % counting the number of each sleep stage for hypnogram generation
    if T.Event(i) == "NREM 3"
        y(i) = 3;
    elseif T.Event(i) == "NREM 2"
        y(i) = 2;
    elseif T.Event(i) == "NREM 1"
        y(i) = 1;
    elseif T.Event(i) == "REM"
        y(i) = 4;
    elseif T.Event(i) == "Wake"
        y(i) = 5;
    else
        y(i) = 0;
    end
    i = i + 1;
end

%y = y(1:end-1); % Removing the last epoch to syncronize the data with the timing.

fileList = dir(fullfile(folderPath, '*.mat'));
len = numel(fileList) - 2;
FinalLen = 0;
if len > length(y)
    FinalLen = length(y);
else
    FinalLen = len;
end

y = y(1:FinalLen);

idx.NREM3idx = find(y == 3);
idx.NREM2idx = find(y == 2);
idx.NREM1idx = find(y == 1);
idx.REMidx = find(y == 4);
idx.Wake = find(y == 5);

% x = [];
% j = 0;
%
% for item2 = 1: numRows % This loop will remove outliers from the array
%     if y(item2) == 0
%         continue
%     else
%         j = j+1;
%         x(j) = y(item2);
%     end
% end
%

% Adjust_num = T.Epoch(find(T.Event(:) == "Wake", 1, 'first')) - 1; %finds the first epoch that the Wake stage happend.
% % 68 should be changed based on each subject! 68 = first wake epoch - 1 =
% % outliers in recording that should become zero in the array!
% newy = zeros(1, Adjust_num); % adding 68 zeros to the start of the array to modify the length of array
% [numRows1,numCols1] = size(x);
% for item3 = 1:numCols1
%     newy(1, Adjust_num+item3) = x(item3);
% end

end