% adaboost.m
% training set consists of: faces (1), non_faces (-1), negatives (-1)

DEBUG = 0; % debug != 0 for extra iteration info

m = n_faces + n_nonfaces; % 3200
[X, Y] = createTrain(faces, nonfaces, 0, n_faces, n_nonfaces, 0); % X : m x 256


%csvwrite('faces.csv', faces);
%csvwrite('nonfaces.csv', nonfaces);

T = 10;
%% begin adaboost initialization
[F, Z, D_cur, D_prev,...
          min_ada_index, alpha, ...
          class_matrix, error_matrix] = ...
    initializeAdaBoost(m, T, delta_size);

%% end adaboost initialization

% pre-compute projections
projections = X * delta;

% pre-compute classifications, error matrix
for i = 1:delta_size
    %ip = ip_mat(:, i);
    [h, ~] = gauss_classify(projections(:,i), delta_face_means(i),...
        delta_face_sd(i), delta_nonface_means(i), delta_nonface_sd(i));
    error_matrix(:,i) = h ~= Y;
    class_matrix(:,i) = h;
end

%csvwrite('h_mat.csv', class_matrix);
%csvwrite('err_mat.csv', error_matrix);

% ---------------------   begin adaboost  --------------------------------------
tic
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

end
toc
% ---------------------   end adaboost  ----------------------------------------
min_ada_index % top T weak classifers used in creation of strong classifer