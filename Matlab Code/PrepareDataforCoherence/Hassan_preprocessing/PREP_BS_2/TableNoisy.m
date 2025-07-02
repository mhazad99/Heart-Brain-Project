function [NoisyTS, nbrNoisy] = TableNoisy(NoisyReport ,NoisyTS)

if ~isempty(NoisyReport)
    d = load(NoisyReport );
end
for r = 1:length(NoisyTS)
    NoisyTS{r,4} = d.Reports{r,2}.options.sensortypes.Value;
end

nbrNoisy = length(NoisyTS);

end
