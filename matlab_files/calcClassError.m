function x = calcClassError(Y_train, strong_class)
	classifications = sign(strong_class);
	indicator = (Y_train ~= classifications);

	error = sum(indicator) / size(Y_train, 1);

	disp(['classification error: ' num2str(error)]);
	x = 0;
end
