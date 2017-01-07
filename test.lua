


local test = {}

local calc = require "calculate";

local function modify(x)

	y = torch.Tensor({{3},{2},{1}});
	print(y)
	x = y;
	print(x);
	print('done');

end

test.modify = modify;

calc.displayErrorTime(1, 0.3, 4.5);


return test;