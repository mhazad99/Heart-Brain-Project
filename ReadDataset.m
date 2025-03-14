% function [rawdata] = ReadDataset(Filelocation, StartTime, EndTime)

% %% NEW Version of edfread
% [header, recordData] = edfread(Filelocation);
% [~,ekgChan] = find(contains(split(header.label),'ekg','IgnoreCase',true));
% rawdata.fs = header.frequency(ekgChan(1));
% rawdata.Samplingtime = 0:1/rawdata.fs:(header.records*header.duration) - 1/rawdata.fs;
% rawdata.Secondtime = header.records*header.duration;
% rawdata.duration = header.duration;
% rawdata.ECG1 = recordData(15, 1:rawdata.Secondtime*rawdata.fs);
% rawdata.ECG2 = recordData(16, 1:rawdata.Secondtime*rawdata.fs);
% rawdata.F3 = recordData(25, 1:rawdata.Secondtime*rawdata.fs);
% rawdata.M1 = recordData(7, 1:rawdata.Secondtime*rawdata.fs);
% rawdata.M2 = recordData(8, 1:rawdata.Secondtime*rawdata.fs);
% rawdata.C3 = recordData(9, 1:rawdata.Secondtime*rawdata.fs);
% rawdata.O1 = recordData(10, 1:rawdata.Secondtime*rawdata.fs);

% end
%% Previous Version of edfread
% Uncomment if you are reading with previous version %
function [rawdata] = ReadDataset(Filelocation, StartTime, EndTime)
%%%%% == VERY IMPORTANT NOTE == %%%%% : When you are reading the dataset
%%%%% and you want to align the start and end times with the txt file and
%%%%% other outputs, consider this fact that the last epoch is not complete
%%%%% in most of the time and if anything is inaligned with the timing or
%%%%% the number of epochs, ... usually comes from last epoch. So by
%%%%% removing it from the dataset you can align the rest of the data! e.x.
%%%%% in the edfread function, the last row which corresponds to the last 5
%%%%% second of the data, should be removed from the data, since it passes
%%%%% the ending time of the data that we don't need it.
[data, annotations] = edfread(Filelocation,'DataRecordOutputType','timetable','TimeOutputType','datetime');
%Timedata = data.("Record Time");
timeOnly = timeofday(data.("Record Time"));
Startindx = find(timeOnly == StartTime);
Endindx = find(timeOnly == EndTime);
data(Endindx:end, :) = [];
data(1:Startindx-1, :) = [];
info = edfinfo(Filelocation); %Dataset infromation
fs = info.NumSamples/seconds(info.DataRecordDuration); %samplimg rate
%NumList = [15, 16, 25, 7, 8, 9, 10];
%A = cell(length(NumList),1);
rawdata.ECG1 = vertcat(data.EKG1{:}); rawdata.ECG2 = vertcat(data.EKG2{:});
rawdata.ECG1 = double(rawdata.ECG1.EKG1); rawdata.ECG2 = double(rawdata.ECG2.EKG2);
rawdata.fs = fs(15);
% for iter = 1:length(NumList)
%     signum = NumList(iter); % ECG1 = 15, ECG2 = 16, F3 = 25, M1 = 7, M2 = 8, C3 = 9, O1 = 10
%     recnum = info.NumDataRecords;
%     t0 = 0;
%     timestep = size(data.O1{1,1}, 1)/fs(signum);
%     i = -1;
%     j = -timestep;
%     TotalY = zeros(recnum*fs(signum)*timestep,1);
%     for item = 1:recnum
%         i = i + 1;
%         j = j + timestep;
%         t = (t0+(i*timestep):1/fs(signum):(timestep+j-(1/fs(signum))));
%         y = data.(signum){item};
%         TotalY(size(data.O1{1,1}, 1)*i+1:item*size(data.O1{1,1}, 1)) = y;
%         %plot(t,y,'b')
%         %hold on
%     end
%     %% Saving variables
%     A{iter} = TotalY;
% end
% rawdata.ECG1 = A{1};
% rawdata.ECG2 = A{2};
% rawdata.F3 = A{3};
% rawdata.M1 = A{4};
% rawdata.M2 = A{5};
% rawdata.C3 = A{6};
% rawdata.O1 = A{7};
% rawdata.fs = fs(15);
%T = 0:1/rawdata.fs(signum):(recnum*5)-1/rawdata.fs(signum);
% figure
% plot(T,TotalY,'r')
% save data.mat
end




%% finding difference between to time sequence in seconds
% t1={'01-Oct-2011 23:44:21'};
% t2={'02-Oct-2011 6:51:33'};
% t11=datevec(datenum(t1));
% t22=datevec(datenum(t2));
% time_interval_in_seconds = etime(t22,t11)
%csvwrite('ECG1.csv', FileData.ECG1);

%% Save the data to run in the Python
% SleepECG1 = ECG1(12*fs(signum)*30:end-(12*fs(signum))-1); %% Values need to be changed based on the subjects!
% writematrix(SleepECG1, 'ECG1.csv')
% load F3.mat
% load M1.mat
% load M2.mat
% Array = [F3, M1, M2];
% save Array Array