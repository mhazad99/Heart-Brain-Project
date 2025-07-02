function missingPrct = RawEcgSignalAnalysis(fSamp, ekg)
%RAWECGSIGNALANALYSIS
% globals

    GlobalDefs();
    global POST_PROCESSING

    %fprintf('Analyzing raw EKG signal...\n');
        
     % Number of samples in zeroEkgDurationThresh seconds
     zeroEkgDurationThresh = POST_PROCESSING.RR_MAX_VALUE; % seconds
     zeroEkgSamplesThresh  = round(zeroEkgDurationThresh*fSamp);
     hZeros = ones(1,zeroEkgSamplesThresh);
     zerosSeq = conv(abs(ekg),hZeros,'same');
     idxZeros = find(zerosSeq == 0);
     diffIdxZeros = diff(idxZeros);
     nbSamples = length(diffIdxZeros);
     totZeroSamples = 0;
     j = 1;
     while(j <= nbSamples)
         totZeroSamples = totZeroSamples + zeroEkgSamplesThresh;
         if (diffIdxZeros(j) == 1)
            j = j + 1; 
            while(j < nbSamples && diffIdxZeros(j) == 1)  
                totZeroSamples = totZeroSamples + 1;
                j = j + 1;
                if (j >= nbSamples)
                    break;
                end    
            end
         end
         j = j + 1;
     end    
     missingPrct = round(100*totZeroSamples/length(ekg));
    
end % End of RawEcgSignalAnalysis function