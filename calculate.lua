local calculate = {}

local function classError(Y_train, strong_class)

	class     = torch.sign(strong_class); 
	indicator = torch.ne(Y_train, class);
	error     = torch.sum(indicator) / Y_train:size()[1];


	print('classification error: '..error);

	return error;
end


local function displayErrorTime(iter, error, start_time)

	err      = torch.squeeze(error);
	end_time = os.time();
	time     = os.difftime(end_time, start_time);

	print('iter '..iter.. '  '..
		'empirical error: '.. err.. '\n'..
		'\telapsed time:    '.. time.. ' seconds');
end


-------- function delcarations -------------------------------------------------
calculate.displayErrorTime  = displayErrorTime;
calculate.classError        = classError;
-------- end function delcarations ---------------------------------------------


return calculate;