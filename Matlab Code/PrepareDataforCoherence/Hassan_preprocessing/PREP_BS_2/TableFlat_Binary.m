function [Flat_binary, nbrFlat] = TableFlat_Binary(binFlat ,FlatTS)

binFlat = num2cell(binFlat);
Flat_binary = [FlatTS , binFlat];


nbrFlat = length(FlatTS);

end
