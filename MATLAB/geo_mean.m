function gm = geo_mean(x)
% Take into account the number of elements in the vector
n = length(x);

% Initialize some variables
gm = 1;

% Iterate through all of the elements
    for i = 1 : n
        d = x(i);
    
        % Compute mean
        gm = gm * d^(1/n);
    
    end
end
