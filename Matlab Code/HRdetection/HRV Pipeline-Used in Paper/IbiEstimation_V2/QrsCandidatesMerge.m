function mergedQrsCandidates = QrsCandidatesMerge(qrsCandidates, fs, timeOffset)
%QRSCANDIDATESMERGE 
   
   mergedQrsCandidates.qrsIndex        = int32.empty;
   mergedQrsCandidates.rrIntervals     = double.empty; % In seconds
   mergedQrsCandidates.qrsTimeStamps   = double.empty; % Time from start in seconds
   mergedQrsCandidates.qrsAmplitudes   = double.empty; % Normalized amplitude
   
   nbEpochs = length(qrsCandidates); % Number of epochs with QRS
   Ts = 1.0/fs; % Sampling Period
     
   for i=1:nbEpochs  
       newIndexes = (qrsCandidates{i}.startIndex + ...
                     qrsCandidates{i}.qrs_i_raw - 1);
       mergedQrsCandidates.qrsIndex = [mergedQrsCandidates.qrsIndex newIndexes];
       newTimeStamps = double(newIndexes-1)*Ts;
       mergedQrsCandidates.qrsTimeStamps = [mergedQrsCandidates.qrsTimeStamps newTimeStamps];
       mergedQrsCandidates.qrsAmplitudes = [mergedQrsCandidates.qrsAmplitudes qrsCandidates{i}.qrs_amp];       
   end
   mergedQrsCandidates.rrIntervals = diff(mergedQrsCandidates.qrsTimeStamps);
   mergedQrsCandidates.qrsTimeStamps = mergedQrsCandidates.qrsTimeStamps + timeOffset;
   
end % End of QrsCandidatesMerge fucntion

