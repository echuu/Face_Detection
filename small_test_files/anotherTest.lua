
local anotherTest = {}



local function multiply10(input)
	print('inside multiply10');

	return input * 10;

end

anotherTest.multiply10 = multiply10;

return anotherTest;



