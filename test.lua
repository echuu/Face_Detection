


local test = {}


local function modify(x)

	y = torch.Tensor({{3},{2},{1}});
	print(y)
	x = y;
	print(x);
	print('done');

end

test.modify = modify;

x = torch.Tensor({{1},{2},{3}});
print('premodify:');
print(x);
print('postmodify:');
test.modify(x)
print(x);


return test;