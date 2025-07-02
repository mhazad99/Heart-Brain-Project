function hrLinearFitStruct = HrLinearFitInit()
%StagePeriodInit 
	hrLinearFitStruct.slope = NaN;
    hrLinearFitStruct.intercept = NaN;
    hrLinearFitStruct.R2 = NaN;
    hrLinearFitStruct.delta = NaN;
    hrLinearFitStruct.nrem.slope = double.empty;
    hrLinearFitStruct.nrem.intercept = double.empty;
    hrLinearFitStruct.nrem.R2 = double.empty;
    hrLinearFitStruct.nrem.delta = double.empty;
    hrLinearFitStruct.rem.slope = double.empty;
    hrLinearFitStruct.rem.intercept = double.empty;
    hrLinearFitStruct.rem.R2 = double.empty;
    hrLinearFitStruct.rem.delta = double.empty;
    hrLinearFitStruct.valid = false;  
end
