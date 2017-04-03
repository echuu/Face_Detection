-- call boost.lua to find weak classifiers


function findWC(T)

	print('inside findWC() function');

	b = require('boost');

	print('successfully loaded boost.lua file');

	print("Boosting weak classifiers -- iterations: "..T);
	b.adaboost(T);

	return 1;
end


findWC(30);
