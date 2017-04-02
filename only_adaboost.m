T = 100;

F = zeros(m, 1);             
Z = 0;                       

D_cur  = zeros(m, 1);        
D_prev = zeros(m, 1);        
D_prev(1:m) = 1 / m;         
D_cur(1:m)  = 1 / m;         

min_ada_index = zeros(T, 1); 
alpha         = zeros(T, 1); 

for t = 1:T
    % weighted_error = zeros(delta_size, 1);
    % find the lowest weighted error and its associated index
    [error, index] = findMinWtErr(D_cur, error_matrix, delta_size, DEBUG, t);

    % calculate alpha -- used to weight the w.c. classifier chosen at iter t
    alpha(t) = 0.5 * log((1 - error) / error);
    min_ada_index(t) = index;
    
    D_prev = D_cur;
    
    % chosen_class = class_matrix(:, min_ada_index(1:t));
    % h = chosen_class(:, t);

    h = class_matrix(:,index);

    % 'boosting' previous strong classifer with additional weighted w.c.
    F     =  F + alpha(t) .* h;
    yh    =  exp(-Y .* F);   % exp(-alpha) if correct, exp(alpha) if incorrect
    Z     =  sum(yh) / m;
    D_cur =  1/Z * 1/m * yh; % update current weights

    calcClassError(Y, F);

end