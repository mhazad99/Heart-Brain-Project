function SHOW_IbiTimeSeries(outputFolder, ibiDataTable)
% GraphicsIbiTimeSeries 

    % Display of the Heart Rate Time Series
    figure(1);
    plot(ibiDataTable.TimeFromStart./3600, ibiDataTable.HeartRates);
    xlabel('Time from Start [hours]');
    ylabel('Heart Rate [bpm]');
    grid on;
    title('Heart Rate Time Series');

    destinationFolder = strcat(outputFolder, ...
                               filesep, ...
                               "GRAPHICS");
    if ~exist(destinationFolder, 'dir')
        mkdir(destinationFolder)
    end    
    
    outputPngFile = strcat(destinationFolder, ...
                           filesep, ...
                           ibiDataTable.ParticipantID(1), ...
                           '_HR.png');
	saveas(gcf,outputPngFile);
    
    % Display of the IBI Time Series
    figure(2);
    plot(ibiDataTable.TimeFromStart./3600, ibiDataTable.RRintervals);
    xlabel('Time from Start [hours]');
    ylabel('IBI [second]');
    grid on;
    title('IBI Time Series');
    
    outputPngFile = strcat(destinationFolder, ...
                           filesep, ...
                           ibiDataTable.ParticipantID(1), ...
                           '_IBI.png');
	saveas(gcf,outputPngFile);
    
end % End of GraphicsIbiTimeSeries

