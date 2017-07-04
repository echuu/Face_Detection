-- findWC.lua
-- call boost.lua to find weak classifiers

function findWC(T)

	b = require('boost');
	print("Boosting weak classifiers -- iterations: "..T);
	b.adaboost(T);

	return 1;
end

findWC(10)
