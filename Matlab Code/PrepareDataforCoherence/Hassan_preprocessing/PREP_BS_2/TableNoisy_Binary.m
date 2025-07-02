function [Noisy_Binary, nbrNoisy] = TableNoisy_Binary(binNoisy ,NoisyTS)

binNoisy = num2cell(binNoisy);
Noisy_Binary = [NoisyTS , binNoisy];

nbrNoisy = length(NoisyTS);

end
