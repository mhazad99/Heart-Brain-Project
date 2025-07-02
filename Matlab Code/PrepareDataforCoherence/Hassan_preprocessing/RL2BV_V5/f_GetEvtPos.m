function nSec = f_GetEvtPos(recStart,evtTime)
    % Get time lapse between recording start and event
    dtRecStart = datetime(recStart,'ConvertFrom','datenum');
    dtEvt = datetime(evtTime,'InputFormat','HH:mm:ss.SSS');
    % Assign recording date to evt datetime object
    dtEvt.Year = dtRecStart.Year;
    dtEvt.Month = dtRecStart.Month;
    dtEvt.Day = dtRecStart.Day;
    dtDiff = abs(datevec(time(between(dtEvt,dtRecStart,'time'))));
    if dtDiff(4) > 12
        dtEvt.Day = dtEvt.Day + 1;
        dtDiff = abs(datevec(time(between(dtEvt,dtRecStart,'time'))));
    end
    % Convert time to seconds
    nSec = 0;
    for i = length(dtDiff):-1:1
       nSec = nSec + (60^(i-1) * dtDiff(end-i+1));
    end
end
