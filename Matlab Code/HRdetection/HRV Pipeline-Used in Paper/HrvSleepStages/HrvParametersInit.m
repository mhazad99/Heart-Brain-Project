function hrvParametersStruct = HrvParametersInit()
%StagePeriodInit 
	hrvParametersStruct.WAKE.RMSSD = NaN;
    hrvParametersStruct.WAKE.SDNN = NaN;
    hrvParametersStruct.WAKE.HR_AVG = NaN;
	hrvParametersStruct.N1.RMSSD = NaN;
    hrvParametersStruct.N1.SDNN = NaN;
    hrvParametersStruct.N1.HR_AVG = NaN;
    hrvParametersStruct.N2.RMSSD = NaN;
    hrvParametersStruct.N2.SDNN = NaN;
    hrvParametersStruct.N2.HR_AVG = NaN;
    hrvParametersStruct.N3.RMSSD = NaN;
    hrvParametersStruct.N3.SDNN = NaN;
    hrvParametersStruct.N3.HR_AVG = NaN;
    hrvParametersStruct.REM.RMSSD = NaN;
    hrvParametersStruct.REM.SDNN = NaN;
    hrvParametersStruct.REM.HR_AVG = NaN;
    hrvParametersStruct.WAKEp.RMSSD = NaN;
	hrvParametersStruct.WAKEp.SDNN = NaN;
    hrvParametersStruct.WAKEp.HR_AVG = NaN;
   	hrvParametersStruct.NREMp.RMSSD = double.empty;
	hrvParametersStruct.NREMp.SDNN = double.empty;
    hrvParametersStruct.NREMp.HR_AVG = double.empty;
  	hrvParametersStruct.REMp.RMSSD = double.empty;
	hrvParametersStruct.REMp.SDNN = double.empty;
    hrvParametersStruct.REMp.HR_AVG = double.empty;
    hrvParametersStruct.valid = false;  
end
