function [IbiData,PipelineStatus] = ReadIbiTimeSeriesFromCSV(fileName)
% ReadIbiTimeSeriesFromCSV
	fid = fopen(fileName,'r');
	% The first line is the feader.
	header = fgetl(fid);
    ParticipantID = string.empty;
    TimeFromStart = double.empty;
	RRintervals = double.empty;
    PipelineStatus.MissingPercent = 0;
    PipelineStatus.CorrectedPercent = 0;
    PipelineStatus.ValidPercent = 0;
	% Read-in the data 
	% ParticipantId, Time from start [s], Date Time [dd-mmm-yyyy HH:MM:SS.FFF], RR interval [s]
	i = 1;
    while ~feof(fid)
        fullLine = fgetl(fid);
        dummyStrings = char(split(fullLine,','));
        ParticipantID(i) = strtrim(dummyStrings(1,:));
        TimeFromStart(i) = str2num(dummyStrings(2,:));
        DateTime(i) = datetime(dummyStrings(3,:),'InputFormat','dd-MMM-yyyy HH:mm:ss.SSS') ;
        RRintervals(i) = str2num(dummyStrings(4,:));
        if (i == 1)
            PipelineStatus.MissingPercent = str2num(dummyStrings(5,:));
            PipelineStatus.CorrectedPercent = str2num(dummyStrings(6,:));
            PipelineStatus.DataQualityFactor = str2num(dummyStrings(7,:));
        end    
        i = i+1;
    end % End of while ~feof(fid)
	
    fclose(fid);
   
    HeartRates = 60.0./RRintervals; % Beats per minute
    IbiData = table(ParticipantID',TimeFromStart',DateTime',RRintervals',HeartRates','VariableNames',{'ParticipantID' 'TimeFromStart' 'DateTime' 'RRintervals' 'HeartRates'});
end % End of ReadIbiTimeSeriesFromCSV function