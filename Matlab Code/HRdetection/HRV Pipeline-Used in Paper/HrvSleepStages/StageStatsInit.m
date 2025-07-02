function stageStatsStruct = StageStatsInit()
%StagePeriodInit 
	stageStatsStruct.lightsOff = "";
    stageStatsStruct.lightsOn = "";
    stageStatsStruct.preWake = NaN;
    stageStatsStruct.postWake = NaN;
    stageStatsStruct.TiB = NaN;
    stageStatsStruct.SOL = NaN;
    stageStatsStruct.WASO = NaN;
    stageStatsStruct.REM_Lat = NaN;
    stageStatsStruct.WAKE_min = double.empty;
    stageStatsStruct.N1_min = NaN;
    stageStatsStruct.N2_min = NaN;
    stageStatsStruct.N3_min = NaN;
    stageStatsStruct.NREM_min = NaN;
    stageStatsStruct.REM_min = NaN;
	stageStatsStruct.TST = NaN;
    stageStatsStruct.N1_PC = NaN;
    stageStatsStruct.N2_PC = NaN;
    stageStatsStruct.N3_PC = NaN;     
    stageStatsStruct.REM_PC = NaN;
    stageStatsStruct.SleepEfficiency= NaN;
    
    stageStatsStruct.NREMp_min = double.empty; % Number of minutes spent in each NREM period.
    stageStatsStruct.NREMp_Wake_pc = double.empty; % Pourcentage of epoch scored as WAKE in each NREM period.
    stageStatsStruct.NREMp_NREM_pc = double.empty; % Pourcentage of epoch scored as NREM (1,2,3, or 4) in each NREM period.
    stageStatsStruct.NREMp_N1_pc = double.empty; % Pourcentage of epoch scored as NREM 1 in each NREM period.
    stageStatsStruct.NREMp_N2_pc = double.empty; % Pourcentage of epoch scored as NREM 2 in each NREM period.
    stageStatsStruct.NREMp_N3_pc = double.empty; % Pourcentage of epoch scored as NREM 3 and 4 in each NREM period.
    stageStatsStruct.NREMp_REM_pc = double.empty; % Pourcentage of epoch scored as REM in each NREM period.
    stageStatsStruct.NREMp_Tot_min = NaN; % Number of minutes spent in NREM periods
    stageStatsStruct.NREMp_Tot_pc = NaN; % Pourcentage of TST spent in NREM periods
    
    stageStatsStruct.REMp_min = double.empty; % Number of minutes spent in each REM period.
    stageStatsStruct.REMp_Wake_pc = double.empty; % Pourcentage of epoch scored as WAKE in each REM period.
    stageStatsStruct.REMp_REM_pc = double.empty; % Pourcentage of epoch scored as REM in each REM period.
    stageStatsStruct.REMp_NREM_pc = double.empty; % Pourcentage of epoch scored as NREM (1,2,3, or 4) in each NREM period. 
    stageStatsStruct.REMp_N1_pc = double.empty; % Pourcentage of epoch scored as NREM 1 in each REM period.
    stageStatsStruct.REMp_N2_pc = double.empty; % Pourcentage of epoch scored as NREM 2 in each REM period.
    stageStatsStruct.REMp_N3_pc = double.empty; % Pourcentage of epoch scored as NREM 3 and 4 in each REM period.    
    stageStatsStruct.REMp_Tot_min = NaN; % Number of minutes spent in NREM periods
    stageStatsStruct.REMp_Tot_pc = NaN; % Pourcentage of TST spent in NREM periods
    stageStatsStruct.TSTp = NaN;
    stageStatsStruct.valid = false;   
end
