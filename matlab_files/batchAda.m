% batchAda.m

% see first part of adaboost.m

if 1 == 1
	load_data();
	generate_weak_classifiers();
	calc_threshold();

	DEBUG = 0; % debug != 0 for extra iteration info

	% when faces, nonfaces come into adaboost, they have already been transposed
	% faces    -- n_faces x 256
	% nonfaces -- n_nonfaces x 256

	% n_negs = 8190;
	n_negs = 0;
	m = n_faces + n_nonfaces + n_negs;
	% [X, Y] = createTrain(faces, nonfaces, sub_images, n_faces, n_nonfaces, n_negs);
	[X, Y] = createTrain(faces, nonfaces, 0, n_faces, n_nonfaces, 0);

	% X : m x 256

	clear faces; 
	clear nonfaces;
	T = 10;

	%% begin adaboost initialization
	[F, Z, D_cur, D_prev,...
	          wc_ind, alpha, ...
	          class_matrix, error_matrix] = ...
	    initializeAdaBoost(m, T, delta_size);

	disp('first chunk complete');
end

% rewrite the the loop that precomputes the error matrix, classification matrix
% rewrite findMinWtErrr() function to iterate thru each train_batch to find the
% index, train_batch of weak classifier that minimizes the weighted error

% ---------------------- ADABOOST PRECOMPUTATIONS ---------------------------- %
if 1 == 1
	k = 4; % split the number of faces/nonface into k batches
	       % X is 5000 x 256 --> dim(X_i) = 1250 x 256
	train_batch = linspace(0, m, k + 1);
	
	tic
	for i = 1:k
		m_i    = train_batch(2) - train_batch(1); % number of images
		X_i    = X(train_batch(i) + 1:train_batch(i+1), :);
		proj_i = X_i * delta; % m_i x delta_size

		% initialize err_i, class_i with proper dim (both m_i x delta_size)
		err_i   = zeros(m_i, delta_size);
		class_i = zeros(m_i, delta_size);

		% calculate error matrix, classification matrix for the i-th train_batch
		% 1 <= i <= k
		for j = 1:delta_size
			[h_i, ~]   = gauss_classify(proj_i(:,j), delta_face_means(j),...
				delta_face_sd(j), delta_nonface_means(j), delta_nonface_sd(j));
			err_i(:,j)   = h_i ~= Y(train_batch(i) + 1:train_batch(i+1));
			class_i(:,j) = h_i; 
		end % inner for loop

		batch_error_name = ['err_mat_', num2str(i), '.csv'];
		batch_class_name = ['class_mat_', num2str(i), '.csv'];

		%csvwrite(batch_error_name, err_i);
		%csvwrite(batch_class_name, class_i);

		disp(['Finished train_batch ', num2str(i), ' classifications']);
	end % outer for loop
	toc

	disp('Finished train_batch calculations');
end

% ---------------------- END ADABOOST PRECOMPUTATIONS ------------------------ %

disp('begin adaboost calculations');

for t = 1:T
	disp(['iter ' num2str(t)]);
	tic

	[err, ind, h] = batchMinimize(D_cur, delta_size, k, train_batch);

	alpha(t) = 0.5 * log((1 - err) / err);
	
	wc_ind(t) = ind; 

	F     = F + alpha(t) .* h;
	yh    = exp(-Y .* F);
	Z     = sum(yh) / m;
	D_cur = 1/Z * 1/m * yh; % update weights

	calcClassError(Y, F); 
	disp(['iter ' num2str(t), ': weak classifier ', num2str(wc_ind(t))]);

	toc

end % adaboost iterations

% end batchAda.m
