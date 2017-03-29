% adaboost.m
% boost weak classifiers
% training set consists of: faces (1), non_faces (-1), negatives (-1)

last_step = 0;


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
F = zeros(m, 1);
Z = 0; % normalizing value
D_cur  = zeros(m, 1);  % 4730 x 1
D_last = zeros(m, 1);
D_last(1:m) = 1 / m;   % initial weights (sum to 1)
D_cur(1:m)  = 1 / m;    

% T = # of iterations of adaboost
T = 10;

delta_ada_chosen_index = zeros(T, 1);
alpha = zeros(T, 1);

ip_mat = X * delta;
class_matrix = zeros(m, delta_size);
error_matrix = zeros(m, delta_size);
for i = 1:delta_size
    ip = ip_mat(:, i);
    [h, ratio] = gauss_classify(ip, delta_face_means(i),...
        delta_face_sd(i), delta_nonface_means(i), delta_nonface_sd(i));
    error_matrix(:,i) = h ~= Y;
    class_matrix(:,i) = h;

end


% begin adaboost
tic
for t = 1:T
    weighted_error = zeros(delta_size, 1);
    delta_reverse = ones(delta_size, 1); 
    disp(['iter: ' num2str(t)]);
    for i = 1:delta_size   % iterate by column     
        error = D_cur' * error_matrix(:,i); % weighted_error
        weighted_error(i, 1) = error;
    end
    

    [error, index] = min(weighted_error);

    disp(['weak classifier: ' num2str(index) ' chosen with '...
           ' weighted error: ' num2str(error)]);

    alpha(t) = 0.5 * log((1 - error) / error);
    delta_ada_chosen_index(t) = index;
    
    D_last = D_cur;
    
    chosen_class = class_matrix(:, delta_ada_chosen_index(1:t));
    h = chosen_class(:, t);
    F = F + alpha(t) .* (delta_reverse(delta_ada_chosen_index(t)) * h);
    
    yh = exp(-Y .* F);
    Z = sum(yh) / m;
    
    D_cur = 1/Z * 1/m * yh;

end
toc