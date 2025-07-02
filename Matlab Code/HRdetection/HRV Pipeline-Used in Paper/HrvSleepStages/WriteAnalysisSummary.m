function WriteAnalysisSummary(  destinationFolder, ...
                                partcipantIds, ...
                                stagePeriods, ...
                                maxPeriods, ...
                                stageStats, ...
                                hrLinearFit, ...
                                hrvParameters, ...
                                dataQuality)
% WriteStageStatsToFile 

% Creating file name and opening file for writing.
currentDateTime = datetime();
currentDateTimeString = datestr(currentDateTime,'dd-mmm-yyyy_HH-MM-SS');
dum = strsplit(destinationFolder,filesep);
summaryFileName = strcat("SUMMARY_",string(dum(end)),"_",currentDateTimeString,".csv");
fprintf("\nWRITING Summary file %s\n",summaryFileName);
summaryFileName = strcat(destinationFolder,filesep,summaryFileName);
fid = fopen(summaryFileName,'w');
if fid == -1
    fprintf(2,"\nERROR creating file %s\n",summaryFileName);
    return;
end

%% Column headers for the Sleep Variables.
headers = ["Particpant_ID" "lights_Off" "lights_On" "Pre_Wake" "Post_Wake" "TiB" "SOL" ...
           "WASO" "REM_lat" "TST" "SE" "N1_min" "N2_min" "N3_min" "REM_min" "N1_pc" ... 
           "N2_pc" "N3_pc" "REM_pc" "NREMp_tot_min" "NREMp_tot_pc"];
for i=1:maxPeriods.nrem
    colHeader = strcat("NREMp",num2str(i),"_min");
    headers = [headers colHeader];
    colHeader = strcat("NREMp",num2str(i),"_NREM_pc");
    headers = [headers colHeader];
    colHeader = strcat("NREMp",num2str(i),"_N1_pc");
    headers = [headers colHeader];
    colHeader = strcat("NREMp",num2str(i),"_N2_pc");
    headers = [headers colHeader];
	colHeader = strcat("NREMp",num2str(i),"_N3_pc");
    headers = [headers colHeader];
    colHeader = strcat("NREMp",num2str(i),"_REM_pc");
    headers = [headers colHeader];
	colHeader = strcat("NREMp",num2str(i),"_WAKE_pc");
    headers = [headers colHeader];
end
headers = [headers "REMp_tot_min"];
headers = [headers "REMp_tot_pc"];
for i=1:maxPeriods.rem
    colHeader = strcat("REMp",num2str(i),"_min");
    headers = [headers colHeader];
    colHeader = strcat("REMp",num2str(i),"_REM_pc");
    headers = [headers colHeader];
    colHeader = strcat("REMp",num2str(i),"_NREM_pc");
    headers = [headers colHeader];
    colHeader = strcat("REMp",num2str(i),"_N1_pc");
    headers = [headers colHeader];
    colHeader = strcat("REMp",num2str(i),"_N2_pc");
    headers = [headers colHeader];
	colHeader = strcat("REMp",num2str(i),"_N3_pc");
    headers = [headers colHeader];
	colHeader = strcat("REMp",num2str(i),"_WAKE_pc");
    headers = [headers colHeader];
end


%% Column headers for the Linear Regression Fit data.
headers = [headers "Slope_Night"];
for i=1:maxPeriods.nrem
    colHeader = strcat("Slope_NREMp",num2str(i));
    headers = [headers colHeader];
end
for i=1:maxPeriods.rem
    colHeader = strcat("Slope_REMp",num2str(i));
    headers = [headers colHeader];
end
headers = [headers "Intercept_Night"];
for i=1:maxPeriods.nrem
    colHeader = strcat("Intercept_NREMp",num2str(i));
    headers = [headers colHeader];
end
for i=1:maxPeriods.rem
    colHeader = strcat("Intercept_REMp",num2str(i));
    headers = [headers colHeader];
end
headers = [headers "R2_Night"];
for i=1:maxPeriods.nrem
    colHeader = strcat("R2_NREMp",num2str(i));
    headers = [headers colHeader];
end
for i=1:maxPeriods.rem
    colHeader = strcat("R2_REMp",num2str(i));
    headers = [headers colHeader];
end
headers = [headers "Delta_Night"];
for i=1:maxPeriods.nrem
    colHeader = strcat("Delta_NREMp",num2str(i));
    headers = [headers colHeader];
end
for i=1:maxPeriods.rem
    colHeader = strcat("Delta_REMp",num2str(i));
    headers = [headers colHeader];
end

%% Column headers for the HRV Analysis data.
headers = [headers "HR_WAKE" "HR_N1" "HR_N2" "HR_N3" "HR_REM" "HR_NREMp" ];
for i=1:maxPeriods.nrem
    colHeader = strcat("HR_NREMp",num2str(i));
    headers = [headers colHeader];
end
headers = [headers "HR_REMp"];
for i=1:maxPeriods.rem
    colHeader = strcat("HR_REMp",num2str(i));
    headers = [headers colHeader];
end
headers = [headers "RMSSD_WAKE" "RMSSD_N1" "RMSSD_N2" "RMSSD_N3" "RMSSD_REM" "RMSSD_NREMp"];
for i=1:maxPeriods.nrem
    colHeader = strcat("RMSSD_NREMp",num2str(i));
    headers = [headers colHeader];
end
headers = [headers "RMSSD_REMp"];
for i=1:maxPeriods.rem
    colHeader = strcat("RMSSD_REMp",num2str(i));
    headers = [headers colHeader];
end
headers = [headers "SDNN_WAKE" "SDNN_N1" "SDNN_N2" "SDNN_N3" "SDNN_REM" "SDNN_NREMp"];
for i=1:maxPeriods.nrem
    colHeader = strcat("SDNN_NREMp",num2str(i));
    headers = [headers colHeader];
end
headers = [headers "SDNN_REMp"];
for i=1:maxPeriods.rem
    colHeader = strcat("SDNN_REMp",num2str(i));
    headers = [headers colHeader];
end
headers = [headers "MissingPercent" "CorrectedPercent" "DataQualityFactor" "ValidEpochPercent"];

stringHeader = strjoin(headers,',');
fprintf(fid,"%s\n",stringHeader);

%% Writing the data.
nbParticpants = length(partcipantIds);
for i = 1:nbParticpants
	if stagePeriods(i).valid == false || ...
       stageStats(i).valid == false    
        continue;
    else
        nbNremPeriods = length(stagePeriods(i).nrem.startIdx);
        nbRemPeriods = length(stagePeriods(i).rem.startIdx);
        %% Writing Sleep Variables data.
        % 1: Participant_ID
        sleepVariablesStr = strcat(partcipantIds(i),',');
        % 2: Lights_Off
        sleepVariablesStr = strcat(sleepVariablesStr,stageStats(i).lightsOff,',');
        % 3: Lights_On
        sleepVariablesStr = strcat(sleepVariablesStr,stageStats(i).lightsOn,',');      
        % 4: Pre_Wake
        if ~isnan(stageStats(i).preWake)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).preWake),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 5: Post_Wake
        if ~isnan(stageStats(i).postWake)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).postWake),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 6: TiB
        if ~isnan(stageStats(i).TiB)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).TiB),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 7: SOL
        if ~isnan(stageStats(i).SOL)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).SOL),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 8: WASO
        if ~isnan(stageStats(i).WASO)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).WASO),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 9: REM_lat
        if ~isnan(stageStats(i).REM_Lat)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).REM_Lat),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 10: TST
        if ~isnan(stageStats(i).TST)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).TST),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 11: SE
        if ~isnan(stageStats(i).SleepEfficiency)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).SleepEfficiency),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 12: N1_min
        if ~isnan(stageStats(i).N1_min)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).N1_min),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 13: N2_min
        if ~isnan(stageStats(i).N2_min)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).N2_min),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 14: N3_min
        if ~isnan(stageStats(i).N3_min)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).N3_min),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 15: REM_min
        if ~isnan(stageStats(i).REM_min)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).REM_min),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 16: N1_pc
        if ~isnan(stageStats(i).N1_PC)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).N1_PC),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 17: N2_pc
        if ~isnan(stageStats(i).N2_PC)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).N2_PC),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 18: N3_pc
        if ~isnan(stageStats(i).N3_PC)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).N3_PC),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 19: REM_pc
        if ~isnan(stageStats(i).REM_PC)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).REM_PC),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 20: NREMp_tot_min
        if ~isnan(stageStats(i).NREMp_Tot_min)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).NREMp_Tot_min),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % 21: NREMp_tot_pc
        if ~isnan(stageStats(i).NREMp_Tot_pc)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).NREMp_Tot_pc),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % All NREM periods data
        for j=1:nbNremPeriods
            % NREMpX_min
            if ~isnan(stageStats(i).NREMp_min(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).NREMp_min(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
            % NREMpX_NREM_pc
            if ~isnan(stageStats(i).NREMp_NREM_pc(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).NREMp_NREM_pc(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
            % NREMpX_N1_pc
            if ~isnan(stageStats(i).NREMp_N1_pc(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).NREMp_N1_pc(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
            % NREMpX_N2_pc
            if ~isnan(stageStats(i).NREMp_N2_pc(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).NREMp_N2_pc(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
            % NREMpX_N3_pc
            if ~isnan(stageStats(i).NREMp_N3_pc(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).NREMp_N3_pc(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
            % NREMpX_REM_pc
            if ~isnan(stageStats(i).NREMp_REM_pc(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).NREMp_REM_pc(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
            % NREMpX_WAKE_pc
            if ~isnan(stageStats(i).NREMp_Wake_pc(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).NREMp_Wake_pc(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
        end 
        for j=nbNremPeriods+1:maxPeriods.nrem
            sleepVariablesStr = strcat(sleepVariablesStr,',,,,,,,');
        end  
        % REMp_tot_min
        if ~isnan(stageStats(i).REMp_Tot_min)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).REMp_Tot_min),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % REMp_tot_pc
        if ~isnan(stageStats(i).REMp_Tot_pc)
            sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).REMp_Tot_pc),',');
        else
            sleepVariablesStr = strcat(sleepVariablesStr,',');
        end
        % All REM periods data
        for j=1:nbRemPeriods
            % REMpX_min
            if ~isnan(stageStats(i).REMp_min(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).REMp_min(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
            % REMpX_REM_pc
            if ~isnan(stageStats(i).REMp_REM_pc(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).REMp_REM_pc(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
            % REMpX_NREM_pc
            if ~isnan(stageStats(i).REMp_NREM_pc(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).REMp_NREM_pc(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
            % REMpX_N1_pc
            if ~isnan(stageStats(i).REMp_N1_pc(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).REMp_N1_pc(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
            % REMpX_N2_pc
            if ~isnan(stageStats(i).REMp_N2_pc(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).REMp_N2_pc(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
            % REMpX_N3_pc
            if ~isnan(stageStats(i).REMp_N3_pc(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).REMp_N3_pc(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
            % REMpX_WAKE_pc
            if ~isnan(stageStats(i).REMp_Wake_pc(j))
                sleepVariablesStr = strcat(sleepVariablesStr,num2str(stageStats(i).REMp_Wake_pc(j)),',');
            else
                sleepVariablesStr = strcat(sleepVariablesStr,',');
            end
        end 
        for j=nbRemPeriods+1:maxPeriods.rem
            sleepVariablesStr = strcat(sleepVariablesStr,',,,,,,,');
        end
        
        %% Writing Linear Regression Fit data.

        linearFitStr = "";
        % Slope_Night
        if ~isnan(hrLinearFit(i).slope)
            linearFitStr = strcat(linearFitStr,num2str(hrLinearFit(i).slope),',');
        else
            linearFitStr = strcat(linearFitStr,',');
        end
        % Slope_NREMp1, Slope_NREMp2, ...
        for j=1:nbNremPeriods
            if ~isnan(hrLinearFit(i).nrem.slope(j))
                linearFitStr = strcat(linearFitStr,num2str(hrLinearFit(i).nrem.slope(j)),',');
            else
                linearFitStr = strcat(linearFitStr,',');
            end
        end
        for j=nbNremPeriods+1:maxPeriods.nrem
            linearFitStr = strcat(linearFitStr,',');
        end
        % Slope_REMp1, Slope_REMp2, ...
        for j=1:nbRemPeriods
            if ~isnan(hrLinearFit(i).rem.slope(j))
                linearFitStr = strcat(linearFitStr,num2str(hrLinearFit(i).rem.slope(j)),',');
            else
                linearFitStr = strcat(linearFitStr,',');
            end
        end
        for j=nbRemPeriods+1:maxPeriods.rem
            linearFitStr = strcat(linearFitStr,',');
        end
        % Intercept_Night
        if ~isnan(hrLinearFit(i).intercept)
            linearFitStr = strcat(linearFitStr,num2str(hrLinearFit(i).intercept),',');
        else
            linearFitStr = strcat(linearFitStr,',');
        end 
        % Intercept_NREMp1, Intercept_NREMp2, ...
        for j=1:nbNremPeriods
            if ~isnan(hrLinearFit(i).nrem.intercept(j))
                linearFitStr = strcat(linearFitStr,num2str(hrLinearFit(i).nrem.intercept(j)),',');
            else
                linearFitStr = strcat(linearFitStr,',');
            end
        end
        for j=nbNremPeriods+1:maxPeriods.nrem
            linearFitStr = strcat(linearFitStr,',');
        end
        % Intercept_REMp1, Intercept_REMp2, ...
        for j=1:nbRemPeriods
            if ~isnan(hrLinearFit(i).rem.intercept(j))
                linearFitStr = strcat(linearFitStr,num2str(hrLinearFit(i).rem.intercept(j)),',');
            else
                linearFitStr = strcat(linearFitStr,',');
            end
        end
        for j=nbRemPeriods+1:maxPeriods.rem
            linearFitStr = strcat(linearFitStr,',');
        end
        % R2_Night
        if ~isnan(hrLinearFit(i).R2)
            linearFitStr = strcat(linearFitStr,num2str(hrLinearFit(i).R2),',');
        else
            linearFitStr = strcat(linearFitStr,',');
        end
        % R2_NREMp1, R2_NREMp2, ...
        for j=1:nbNremPeriods
            if ~isnan(hrLinearFit(i).nrem.R2(j))
                linearFitStr = strcat(linearFitStr,num2str(hrLinearFit(i).nrem.R2(j)),',');
            else
                linearFitStr = strcat(linearFitStr,',');
            end
        end
        for j=nbNremPeriods+1:maxPeriods.nrem
            linearFitStr = strcat(linearFitStr,',');
        end
        % R2_REMp1, R2_REMp2, ...
        for j=1:nbRemPeriods
            if ~isnan(hrLinearFit(i).rem.R2(j))
                linearFitStr = strcat(linearFitStr,num2str(hrLinearFit(i).rem.R2(j)),',');
            else
                linearFitStr = strcat(linearFitStr,',');
            end
        end
        for j=nbRemPeriods+1:maxPeriods.rem
            linearFitStr = strcat(linearFitStr,',');
        end
        % Delta_Night
        if ~isnan(hrLinearFit(i).delta)
            linearFitStr = strcat(linearFitStr,num2str(hrLinearFit(i).delta),',');
        else
            linearFitStr = strcat(linearFitStr,',');
        end
        % Delta_NREMp1, Delta_NREMp2, ...
        for j=1:nbNremPeriods
            if ~isnan(hrLinearFit(i).nrem.delta(j))
                linearFitStr = strcat(linearFitStr,num2str(hrLinearFit(i).nrem.delta(j)),',');
            else
                linearFitStr = strcat(linearFitStr,',');
            end
        end
        for j=nbNremPeriods+1:maxPeriods.nrem
            linearFitStr = strcat(linearFitStr,',');
        end
        % Delta_REMp1, Delta_REMp2, ...
        for j=1:nbRemPeriods
            if ~isnan(hrLinearFit(i).rem.delta(j))
                linearFitStr = strcat(linearFitStr,num2str(hrLinearFit(i).rem.delta(j)),',');
            else
                linearFitStr = strcat(linearFitStr,',');
            end
        end
        for j=nbRemPeriods+1:maxPeriods.rem
            linearFitStr = strcat(linearFitStr,',');
        end

        %% Writing HRV Analysis data.

        hrvDataStr = "";
        % HR_WAKE
        if ~isnan(hrvParameters(i).WAKE.HR_AVG)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).WAKE.HR_AVG),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % HR_N1 
        if ~isnan(hrvParameters(i).N1.HR_AVG)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).N1.HR_AVG),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % HR_N2 
        if ~isnan(hrvParameters(i).N2.HR_AVG)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).N2.HR_AVG),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % HR_N3 
        if ~isnan(hrvParameters(i).N3.HR_AVG)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).N3.HR_AVG),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end 
        % HR_REM 
        if ~isnan(hrvParameters(i).REM.HR_AVG)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).REM.HR_AVG),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % HR_NREMp
        if ~isnan(hrvParameters(i).NREMp.HR_AVG(nbNremPeriods+1))
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).NREMp.HR_AVG(nbNremPeriods+1)),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end   
        % HR_NREMp1, HR_NREMp2, ...
        for j=1:nbNremPeriods
            if ~isnan(hrvParameters(i).NREMp.HR_AVG(j))
                hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).NREMp.HR_AVG(j)),',');
            else
                hrvDataStr = strcat(hrvDataStr,',');
            end
        end
        for j=nbNremPeriods+1:maxPeriods.nrem
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % HR_REMp
        if ~isnan(hrvParameters(i).REMp.HR_AVG(nbRemPeriods+1))
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).REMp.HR_AVG(nbRemPeriods+1)),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % HR_REMp1, HR_REMp2, ...
        for j=1:nbRemPeriods
            if ~isnan(hrvParameters(i).REMp.HR_AVG(j))
                hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).REMp.HR_AVG(j)),',');
            else
                hrvDataStr = strcat(hrvDataStr,',');
            end
        end
        for j=nbRemPeriods+1:maxPeriods.rem
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % RMSSD_WAKE
        if ~isnan(hrvParameters(i).WAKE.RMSSD)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).WAKE.RMSSD),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % RMSSD_N1 
        if ~isnan(hrvParameters(i).N1.RMSSD)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).N1.RMSSD),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % RMSSD_N2 
        if ~isnan(hrvParameters(i).N2.RMSSD)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).N2.RMSSD),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % RMSSD_N3 
        if ~isnan(hrvParameters(i).N3.RMSSD)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).N3.RMSSD),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end 
        % RMSSD_REM 
        if ~isnan(hrvParameters(i).REM.RMSSD)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).REM.RMSSD),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % RMSSD_NREMp
        if ~isnan(hrvParameters(i).NREMp.RMSSD(nbNremPeriods+1))
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).NREMp.RMSSD(nbNremPeriods+1)),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end  
        % RMSSD_NREMp1, RMSSD_NREMp2, ...
        for j=1:nbNremPeriods
            if ~isnan(hrvParameters(i).NREMp.RMSSD(j))
                hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).NREMp.RMSSD(j)),',');
            else
                hrvDataStr = strcat(hrvDataStr,',');
            end
        end
        for j=nbNremPeriods+1:maxPeriods.nrem
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % RMSSD_REMp
        if ~isnan(hrvParameters(i).REMp.RMSSD(nbRemPeriods+1))
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).REMp.RMSSD(nbRemPeriods+1)),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % RMSSD_REMp1, RMSSD_REMp2, ...
        for j=1:nbRemPeriods
            if ~isnan(hrvParameters(i).REMp.RMSSD(j))
                hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).REMp.RMSSD(j)),',');
            else
                hrvDataStr = strcat(hrvDataStr,',');
            end
        end
        for j=nbRemPeriods+1:maxPeriods.rem
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % SDNN_WAKE
        if ~isnan(hrvParameters(i).WAKE.SDNN)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).WAKE.SDNN),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % SDNN_N1 
        if ~isnan(hrvParameters(i).N1.SDNN)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).N1.SDNN),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % SDNN_N2 
        if ~isnan(hrvParameters(i).N2.SDNN)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).N2.SDNN),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % SDNN_N3 
        if ~isnan(hrvParameters(i).N3.SDNN)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).N3.SDNN),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end 
        % SDNN_REM 
        if ~isnan(hrvParameters(i).REM.SDNN)
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).REM.SDNN),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end    
        % SDNN_NREMp
        if ~isnan(hrvParameters(i).NREMp.SDNN(nbNremPeriods+1))
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).NREMp.SDNN(nbNremPeriods+1)),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % SDNN_NREMp1, SDNN_NREMp2, ...
        for j=1:nbNremPeriods
            if ~isnan(hrvParameters(i).NREMp.RMSSD(j))
                hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).NREMp.SDNN(j)),',');
            else
                hrvDataStr = strcat(hrvDataStr,',');
            end
        end
        for j=nbNremPeriods+1:maxPeriods.nrem
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % SDNN_REMp
        if ~isnan(hrvParameters(i).REMp.SDNN(nbRemPeriods+1))
            hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).REMp.SDNN(nbRemPeriods+1)),',');
        else
            hrvDataStr = strcat(hrvDataStr,',');
        end
        % SDNN_REMp1, SDNN_REMp2, ...
        for j=1:nbRemPeriods
            if ~isnan(hrvParameters(i).REMp.SDNN(j))
                hrvDataStr = strcat(hrvDataStr,num2str(hrvParameters(i).REMp.SDNN(j)),',');
            else
                hrvDataStr = strcat(hrvDataStr,',');
            end
        end
        for j=nbRemPeriods+1:maxPeriods.rem
            hrvDataStr = strcat(hrvDataStr,',');
        end

        %% Writing Data quality parameters.

        dataQualityStr = "";
        % MissingPercent
        if ~isnan(dataQuality(i).MissingPercent)
            dataQualityStr = strcat(dataQualityStr,num2str(dataQuality(i).MissingPercent),',');
        else
            dataQualityStr = strcat(dataQualityStr,',');
        end
        % CorrectedPercent
        if ~isnan(dataQuality(i).CorrectedPercent)
            dataQualityStr = strcat(dataQualityStr,num2str(dataQuality(i).CorrectedPercent),',');
        else
            dataQualityStr = strcat(dataQualityStr,',');
        end
        % DataQualityFactor
        if ~isnan(dataQuality(i).CorrectedPercent)
            dataQualityStr = strcat(dataQualityStr,num2str(dataQuality(i).DataQualityFactor),',');
        else
            dataQualityStr = strcat(dataQualityStr,',');
        end
        % ValidEpochPercent
        if ~isnan(dataQuality(i).ValidEpochPercent)
            dataQualityStr = strcat(dataQualityStr,num2str(dataQuality(i).ValidEpochPercent),',');
        else
            dataQualityStr = strcat(dataQualityStr,',');
        end

        dataString = strcat(sleepVariablesStr,linearFitStr,hrvDataStr,dataQualityStr);
        fprintf(fid,"%s\n",dataString);
	end    
end % End of for i = 1:nbParticpants

fclose(fid);
end % End of WriteAnalysisSummary function. 

