--package.path = ";;/home/eric/torch/install/bin/luajit"

function testFunction(x, y)

	print('inside lua file');
	other = require('anotherTest');
	local class = require('common_defs');
	local calc  = require('calculate');
	local class = require('classify');
	local ext   = require('externalFunctions');
	local ld    = require('load_data');
	local csv2tensor  = require('csv2tensor');
	local torch = require('torch');

	example = torch.Tensor(3,1):zero();
	print(example);

	print('anotherTest.lua included');

	sum = x + y ;
	--prod = other.multiply10(sum);
	prod = other.multiply10(sum);


	return prod;
end


x = testFunction(10, 15)
print(x)