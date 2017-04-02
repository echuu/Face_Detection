-- call boost.lua to find weak classifiers

b = require('boost');

local function findWC(T)
	print("Boosting weak classifiers -- iterations: "..T);
	b.adaboost(T);
end


findWC(30);
