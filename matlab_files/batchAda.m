% batchAda.m

% see first part of adaboost.m


% rewrite the the loop that precomputes the error matrix, classification matrix
% rewrite findMinWtErrr() function to iterate thru each train_batch to find the
% index, train_batch of weak classifier that minimizes the weighted error

% ---------------------- ADABOOST PRECOMPUTATIONS ---------------------------- %
if 1 == 0
	k = 4; % split the number of faces/nonface into k batches
	       % X is 5000 x 256 --> dim(X_i) = 1250 x 256
	train_batch = linspace(0, m, k + 1);
	for i = 1:k

		m_i    = train_batch(2) - train_batch(1); % number of images
		X_i    = X(train_batch(i) + 1:train_batch(i+1), :);
		proj_i = X_i * delta; % m_i x delta_size

		% initialize err_i, class_i with proper dimensions (both m_i x delta_size)
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

		csvwrite(batch_error_name, err_i);
		csvwrite(batch_class_name, class_i);

		disp(['Finished train_batch ', num2str(i), ' classifications']);
	end % outer for loop
	disp('Finished train_batch calculations');
end

% ---------------------- END ADABOOST PRECOMPUTATIONS ------------------------ %
T = 10;

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
