
function hrvParameters = ComputeHrvParameters(stageIbiData)
%ComputeHrvParameters 
    if (~isempty(stageIbiData))
        N = length(stageIbiData);
        hrvParameters.RMSSDN = sqrt(sum(diff(stageIbiData).^2)/(N-1));
        hrvParameters.SDNN = std(stageIbiData);
        hrvParameters.HR = mean(60.0./stageIbiData);
    else
        hrvParameters.RMSSDN = NaN;
        hrvParameters.SDNN = NaN;
        hrvParameters.HR = NaN;
    end
end % End of hrvParameters

