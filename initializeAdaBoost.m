function [F, Z, D_cur, D_prev,...
	      min_ada_index, alpha, ...
	      class_matrix, error_matrix] = ...
	initializeAdaBoost(num_imgs, iters, delta_size)

	% input params:
		% num_imgs   = faces + nonfaces used to train
		% iters      = number of iterations of AdaBoost
		% delta_size = number of weak classifiers
	% return values:
		% F = vector storing weak classifiers
		% Z = normalizing value
		% D_cur  = weights of current iteration
		% D_prev = weights of previous iteration
		% min_ada_index = vector storing index ~ w.c. w/ min wt. err
		% alpha = weights for each of the weak classifiers
		% class_matrix = store classifications for each image
		% error_matrix = errors of each classification by each w.c.

	F = zeros(num_imgs, 1);             
	Z = 0;                       
	D_cur  = zeros(num_imgs, 1);        
	D_prev = zeros(num_imgs, 1);        

	D_prev(1:num_imgs) = 1 / num_imgs;         
	D_cur(1:num_imgs)  = 1 / num_imgs;         

	min_ada_index = zeros(iters, 1); 
	alpha         = zeros(iters, 1); 

	class_matrix = zeros(num_imgs, delta_size); 
	error_matrix = zeros(num_imgs, delta_size); 

