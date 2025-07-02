function [Thresh_binary, nbrThresh] = TableThresh_Binary(binThresh ,ThreshTS)

binThresh = num2cell(binThresh);
Thresh_binary = [ThreshTS , binThresh];

nbrThresh = length(ThreshTS);
end
