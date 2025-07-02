function [obvsBadTS,nbrart] = TableBad(obvsbadReport ,obvsBadTS)

if ~isempty(obvsbadReport)
    c = load(obvsbadReport );
end
for r = 1:length(obvsBadTS)
    obvsBadTS{r,4} = c.Reports{r,2}.options.sensortypes.Value;
end

nbrart = length(obvsBadTS);

end
