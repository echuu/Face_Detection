

function [h_mat, err_mat] = computeBatchClassification(batch, batch_size)

	for i = 1:batch_size
		[h, ~] = gauss_classify(batch(:,i), delta_face_means(i),...
			            delta_face_sd(i), delta_nonface_means(i),
			            delta_nonface_sd(i));
		err_mat(:,i) = h ~= Y;
		class_matrix(:,i) = h;

	end
end
