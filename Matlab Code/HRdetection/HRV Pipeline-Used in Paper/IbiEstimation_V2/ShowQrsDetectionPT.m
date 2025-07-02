function ShowQrsDetection(qrsPreProcesing, qrsDetectionStage1)
%SHOWQRSDETECTION 
global PEAK_DETECTION

time = qrsPreProcesing.time;
ecgInt = qrsPreProcesing.ecgInt/max(abs(qrsPreProcesing.ecgInt));
ecgHpf = qrsPreProcesing.ecgHpf/max(abs(qrsPreProcesing.ecgHpf));
nbSamples = length(ecgHpf);


ecgMax = max(abs(ecgHpf));
ecgMinThresh = PEAK_DETECTION.ECG_MIN_THRESH*ecgMax.*ones(1,nbSamples);
ecgMaxThresh = PEAK_DETECTION.ECG_MAX_THRESH*ecgMax.*ones(1,nbSamples);

lower = 0.5.*ones(1,nbSamples);
    figure(22)
    plot(time, ecgHpf, 'k', ...
        time(R_loc), ecgHpf(R_loc),'c*', ...
        time,qrsPreProcesing.ecgInt, 'm', ...
        time(R_loc),qrsPreProcesing.ecgInt(R_loc),'g*', ...
        time, thresh1, 'r--', ...
        time, thresh2, 'b--');
    hold on;
    for i=1:nbLeft
       line([time(left(i)) time(left(i))], get(gca, 'ylim'),'LineStyle',':','LineWidth',1,'Color','blue');
       line([time(right(i)) time(right(i))], get(gca, 'ylim'),'LineStyle',':','LineWidth',1,'Color','blue');
    end
    for i=1:length(Q_loc)
       line([time(Q_loc(i)) time(Q_loc(i))], get(gca, 'ylim'),'Color','red');
       line([time(S_loc(i)) time(S_loc(i))], get(gca, 'ylim'),'Color','red');
    end
    grid on;
    hold off;
    
    figure(23)
    plot(time, ecgHpf, 'k', ...
        time(R_loc),ecgHpf(R_loc),'c*', ...       
        time, ecgMinThresh, 'r--', ...
        time, ecgMaxThresh, 'b--');
    hold on;
    
    hold off
    xlabel('Second');
    ylabel('Normalized ECG');
    xlim([time(1) time(end)])
    grid on;
 
    %w = waitforbuttonpress;
	pause(0.5);
end

