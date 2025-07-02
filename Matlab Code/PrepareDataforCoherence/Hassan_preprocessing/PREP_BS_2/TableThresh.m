function [ThreshTS, nbrThresh] = TableThresh(ThreshR ,ThreshTS)

if ~isempty(ThreshR)
    g = load(ThreshR );
end
for r = 1:length(ThreshTS)
    ThreshTS{r,4} = g.Reports{r,2}.options.sensortypes.Value;
end

nbrThresh = length(ThreshTS);
end
