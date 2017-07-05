% batchMinimize.m

% findMinWtErr() rewritten to minimize over batches

% weights (m x 1) holds weights for each of the images
function [err, ind, h] = batchMinimize(weights, dim, num_batches, train_batch)
	% return quantities:
		% weighted error (needed for updating alphas)
		% batch_id: used to determine which train_batch to find the wk class.
		% ind: index of the w.c. within the train_batch
		% h: the classification that gives the lowest weighted error

	min_id     = -1;
	min_wt_err = 9999;

	% read in error_matrix, class_matrix for each of the k batches

	err_vec_i = zeros(1, dim);
	tic
	for i = 1:num_batches

		batch_error_name = ['err_mat_', num2str(i), '.csv'];
		batch_class_name = ['class_mat_', num2str(i), '.csv'];

		err_mat_i   = csvread(batch_error_name);
		class_mat_i = csvread(batch_class_name); 

		wt_i = weights(train_batch(i)+1:train_batch(i+1))';

		err_vec_i = err_vec_i +  wt_i * err_mat_i; % 1 x delta_size

		disp(['train_batch ', num2str(i), ' loaded']);

	end % outer for loop
	toc

	% find minimum weighted error across all batches
	for j = 1:dim % iterate thru weak classifiers
		if err_vec_i(j) < min_wt_err
			min_wt_err = err_vec_i(j);
			min_ind = j;
		end
	end % inner for loop

	% line below needs to be fixed to get the i-th column of each 
	% train_batch matrix -- looks like it will need reading in of each train_batch again

	% use cached train_batch
	h = zeros(size(weights,1), 1); % m x 1
	h(train_batch(num_batches)+1:train_batch(num_batches+1)) = class_mat_i(:, min_ind); 

	for j = 1:(num_batches - 1)

		batch_class_name = ['class_mat_', num2str(j), '.csv'];

		class_mat_i = csvread(batch_class_name); % read in each train_batch

		h(train_batch(j)+1:train_batch(j+1)) = class_mat_i(:, min_ind); % get col: min_ind

	end

	err      = min_wt_err;
	ind      = min_ind;

% end batchMinimize.m
