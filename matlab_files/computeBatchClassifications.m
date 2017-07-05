% computeBatchClassifications.m

for i = 1:k

	m_i    = train_batch(2) - train_batch(1); % number of images
	X_i    = X(train_batch(i) + 1:train_batch(i+1), :);
	proj_i = X_i * delta; % m_i x delta_size

	% initialize err_i, class_i with proper delta_size (both m_i x delta_size)
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

	disp(['Finished train_batch ', num2str(i), ' classifications']);

end % outer for loop
