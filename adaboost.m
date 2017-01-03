% adaboost.m
% boost weak classifiers
% training set consists of: faces (1), non_faces (-1), negatives (-1)

n_faces    = 800;
n_nonfaces = 3200;
n_negs     = size(sub_images, 2);

m = n_faces + n_nonfaces + n_negs;

% X --  256 x 4730
% Y -- 4730 x    1 (m x 1) -- correct classification of each image (+/-1)
[X, Y] = createTrain(face16, nonface16, sub_images,...
                      n_faces, n_nonfaces, n_negs);

%% adaboost initialization
D_cur  = zeros(m, 1);  % 4730 x 1
D_last = zeros(m, 1);
D_last(1:m) = 1 / m;   % initial weights (sum to 1)
D_cur(1:m)  = 1 / m;    

% T = # of iterations of adaboost
T = 100;

delta_ada_chosen_index = zeros(T, 1);
alpha = zeros(T, 1);


ip_mat = X' * delta; 
% begin adaboost
for t = 1:T
    weighted_error = zeros(delta_size, 1);
    delta_reverse = ones(delta_size, 1);
    tic 
    disp(['Staring iteration: ' num2str(t)]);
    for i = 1:delta_size   % iterate by column     
        error = 0;
        ip = ip_mat(:, i);
        [h, ratio] = matrix_gauss_classify(ip, delta_face_means(i),...
            delta_face_sd(i), delta_nonface_means(i), delta_nonface_sd(i));
            
        indicator = h ~= Y;   
            
        error = D_cur' * indicator;
        
        weighted_error(i, 1) = error;
        
    end
    toc

    [error, index] = min(weighted_error);
    alpha(t) = 0.5 * log((1-error)/error);
    delta_ada_chosen_index(t) = index;
    
    D_last = D_cur;
    Z = 0;
    F_all(1:m) = 0;
    
    ip_mat_t = X' * delta(:,delta_ada_chosen_index(1:t)); 
    for i = 1:m  % iterate rows of X
        F = 0;
        for j = 1:t
            ip = ip_mat_t(i, j);
            [h, ratio] = gauss_classify(ip,...
                delta_face_means(delta_ada_chosen_index(j)),...
                delta_face_sd(delta_ada_chosen_index(j)),...
                delta_nonface_means(delta_ada_chosen_index(j)),...
                delta_nonface_sd(delta_ada_chosen_index(j)));
            F = F + alpha(j) * delta_reverse(delta_ada_chosen_index(j)) * h;
        end
        F_all(i) = F;
        Z = Z + exp(-Y(i)*F);
    end
    Z = Z/m;
    
    for i = 1:m
        D_cur(i) = 1/Z * 1/m * exp(-Y(i)*F_all(i));
    end
end