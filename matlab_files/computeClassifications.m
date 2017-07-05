% computeErrClass.m

for i = 1:num_batches
	
	m_i    = train_batch(2) - train_batch(1); % number of images
	X_i    = X(train_batch(i) + 1:train_batch(i+1), :);
	proj_i = X_i * delta; % m_i x dim

	% initialize err_i, class_i with proper dim (both m_i x dim)
	err_i   = zeros(m_i, dim);
	class_i = zeros(m_i, dim);

	% calculate error matrix, classification matrix for the i-th train_batch
	% 1 <= i <= k
	for j = 1:dim
		[h_i, ~] = gauss_classify(proj_i(:,j), delta_face_means(j),...
				delta_face_sd(j), delta_nonface_means(j), delta_nonface_sd(j));

		err_i(:,j)   = h_i ~= (Y(train_batch(i) + 1:train_batch(i+1)));
		class_i(:,j) = h_i; 
	end % inner for loop

	wt_i = weights(train_batch(i)+1:train_batch(i+1))';
	err_vec_i = err_vec_i +  wt_i * err_i;

end % outer for loop
