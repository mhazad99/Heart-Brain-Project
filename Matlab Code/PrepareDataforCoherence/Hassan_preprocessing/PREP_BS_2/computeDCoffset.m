function sign = computeDCoffset(sData,eegChan)
% computeDCoffset - DC offset with detreding of signal use the pseudo-inverse
%
% SYNOPSIS: sign = computeDCoffset(sData,eegChan)
%
%

if ~isempty(sData(eegChan,:))
    nTime = length(sData(eegChan,:));
    %   iTime = 0:nTime-1;

    x = [ones(1,nTime); 0:nTime-1 ];
    % (0:nTime-1).^2
    invxcov = inv(x * x');

    sc = sData(eegChan,:);
    beta    =  sc * x' * invxcov;
else
    beta = 0; x = 0;

end
sign = sData(eegChan,:) - beta*x;
end

