% adaboost.m
% boost weak classifiers
% training set consists of: faces (1), non_faces (-1), negatives (-1)

last_step = 0;

DEBUG = 0; % debug != 0 for extra iteration info

if last_step == 1
    n_negs     = size(sub_images, 2);
    m = n_faces + n_nonfaces + n_negs;

    % X --  256 x 4730
    % Y -- 4730 x    1 (m x 1) -- correct classification of each image (+/-1)
    [X, Y] = createTrain(faces, nonfaces, sub_images,...
                          n_faces, n_nonfaces, n_negs);
else
    m = n_faces + n_nonfaces; % 3200
    % X : m x 256
    [X, Y] = createTrain(faces, nonfaces, 0,...
                          n_faces, n_nonfaces, 0);
end

%% adaboost initialization
F = zeros(m, 1);             % store 'strong' classifications at each iteration
Z = 0;                       % normalizing value
D_cur  = zeros(m, 1);        % weights for current iteration
D_prev = zeros(m, 1);        % weights for previous iteration

D_prev(1:m) = 1 / m;         % initial weights (sum to 1)
D_cur(1:m)  = 1 / m;         % initial weights (sum to 1)

min_ada_index = zeros(T, 1); % index associated with the w.c. with min. wt. err
alpha         = zeros(T, 1); % weights for each of the weak classifers


% number of iterations of adaboost
T = 10;

% pre-compute projections
ip_mat = X * delta;
class_matrix = zeros(m, delta_size);
error_matrix = zeros(m, delta_size);

% pre-compute classifications, error matrix
for i = 1:delta_size
    ip = ip_mat(:, i);
    [h, ratio] = gauss_classify(ip, delta_face_means(i),...
        delta_face_sd(i), delta_nonface_means(i), delta_nonface_sd(i));
    error_matrix(:,i) = h ~= Y;
    class_matrix(:,i) = h;

end

% ---------------------   begin adaboost  --------------------------------------
tic
for t = 1:T
    weighted_error = zeros(delta_size, 1);
    delta_reverse = ones(delta_size, 1); 
    % find the lowest weighted error and its associated index
    [error, index] = findMinWtErr(D_cur, error_matrix, delta_size, DEBUG);

    % calculate alpha -- used to weight the w.c. classifier chosen at iter t
    alpha(t) = 0.5 * log((1 - error) / error);
    min_ada_index(t) = index;
    
    D_prev = D_cur;
    
    chosen_class = class_matrix(:, min_ada_index(1:t));
    h = chosen_class(:, t);

    % 'boosting' previous strong classifer with additional weighted w.c.
    F = F + alpha(t) .* (delta_reverse(min_ada_index(t)) * h);
    
    yh = exp(-Y .* F);
    Z = sum(yh) / m;
    
    D_cur = 1/Z * 1/m * yh;

end
toc
% ---------------------   end adaboost  ----------------------------------------
min_ada_index % top T weak classifers used in creation of strong classifer