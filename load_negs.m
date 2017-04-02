
% load and process negative_images
% create subimages from the background images (negative)
neg1     = rgb2gray(imread('test/Background_1.jpg'));
neg1     = rgb2gray(imread('test/Test_Image_1.jpg'));
neg1_mod = imresize(neg1, 0.2);

neg2     = rgb2gray(imread('test/Background_2.jpg'));
neg2_mod = imresize(neg2, 0.2);

neg3     = rgb2gray(imread('test/Background_3.jpg'));
neg3_mod = imresize(neg3, 0.2);


start_x = 5;
step    = 16;      % dimension of sub-images
start_y = 4;	

num_images = 2730 * 3; % # of negative images add to training set (nonfaces)

dim = 16;

sub_images = zeros(dim * dim, num_images);

n = 1;
r = start_y;
while (r < size(neg2_mod, 1))
	next_r = r + dim - 1;
	c = start_x;
	while (c < size(neg2_mod, 2))
		next_c = c + dim - 1;
		
		sub = neg1_mod(r:next_r, c:next_c);
		%imshow(sub);
		%pause(0.01)
		sub_images(:, n) = reshape(sub, dim * dim, 1);
		disp(['Created subimage ' num2str(n)])

		sub = neg2_mod(r:next_r, c:next_c);
		sub_images(:, n+1) = reshape(sub, dim * dim, 1);
		disp(['Created subimage ' num2str(n+1)]) 

		sub = neg3_mod(r:next_r, c:next_c);
		sub_images(:, n+2) = reshape(sub, dim * dim, 1);
		disp(['Created subimage ' num2str(n+2)]) 

		n = n + 3;
		c = next_c;
		if c + dim - 1 > size(neg2_mod, 2)
			break;
		end
	end
	disp(['Next row of image']);
	r = next_r;
	if r + dim - 1 > size(neg2_mod, 1)
			break;
	end
end