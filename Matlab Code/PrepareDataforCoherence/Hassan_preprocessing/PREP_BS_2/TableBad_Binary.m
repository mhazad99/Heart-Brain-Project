function [obvsBadTS_binary,nbrart] = TableBad_Binary(binVeryBad ,obvsBadTS)
%check when empty
binVeryBad = num2cell(binVeryBad);
obvsBadTS_binary = [obvsBadTS , binVeryBad];

nbrart = length(obvsBadTS);

end
