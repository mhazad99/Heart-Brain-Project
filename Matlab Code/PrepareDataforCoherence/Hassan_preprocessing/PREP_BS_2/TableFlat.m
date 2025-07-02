function [FlatTS, nbrFlat] = TableFlat(FlatReport ,FlatTS)

if ~isempty(FlatReport)
    f = load(FlatReport );
end
for r = 1:length(FlatTS)
    FlatTS{r,4} = f.Reports{r,2}.options.sensortypes.Value;
end

nbrFlat = length(FlatTS);

end
