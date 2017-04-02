--package.path = ";;/home/eric/Face_Detection/?.lua"



function testFunction(x, y)

	print('inside lua file');
	other = require('anotherTest');
	print('anotherTest.lua included');

	sum = x + y ;
	--prod = other.multiply10(sum);
	prod = other.multiply10(sum);


	return prod;
end


x = testFunction(10, 15)
print(x)