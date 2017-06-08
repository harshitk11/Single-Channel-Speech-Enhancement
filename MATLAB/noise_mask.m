function [alpha, beta] = noise_mask(threshold)

% Parameters taken from Paper 18
alpha_min = 1;
alpha_max = 6;
beta_min = 0;
beta_max = 0.02;

tmin = min(threshold);
tmax = max(threshold);

alpha = zeros(length(threshold),1);

for i = 1:length(threshold)
    if threshold(i) == tmin
        alpha(i) = alpha_max;
    elseif threshold(i) == tmax
        alpha(i) = alpha_min;
    else
        alpha(i) = alpha_max*((tmax-threshold(i))/(tmax-tmin)) + alpha_min*((threshold(i)-tmin)/(tmax-tmin)); 
    end
end

beta = zeros(length(threshold),1);

for i = 1:length(threshold)
    if threshold(i) == tmin
        beta(i) = beta_max;
    elseif threshold(i) == tmax
        beta(i) = beta_min;
    else
        beta(i) = beta_max*((tmax-threshold(i))/(tmax-tmin)) + beta_min*((threshold(i)-tmin)/(tmax-tmin)); 
    end
end

end