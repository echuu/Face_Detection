% create the feature window 
% height, width of the rectange
% dim  : dimension of window
% type : (1) 2-rectangle, (2) 3-rectangle feature, (3) diagonal feature

function [window, window_t] = create_window(pos_i, pos_j, height, width, dim, type)
	window = zeros(dim, dim);
	window_t = zeros(dim, dim);

	if type == 1
		for h = 1:height
			for w = 1:width
				if h <= height / 2
					window(pos_i + h, pos_j + w) = 1;      % left rectangle
				else
					window(pos_i + h, pos_j + w) = -1;     % right triangle
				end
			end
		end
		window_t = window';
	elseif type == 2
		for h = 1:height
			for w = 1:width
				if h > height / 3 && h <= 2 * height / 3   % middle rectange
					window(pos_i + h, pos_j + w) = 1; 
				else
					window(pos_i + h, pos_j + w) = -1;     % outer rectangles
				end
			end
		end
		window_t = window';
	else
		for h = 1:height
			for w = 1:width
				if h <= height / 2 && w <= width / 2       % up-left rectangle
					window(pos_i + h, pos_j + w) = 1;
					window_t(pos_i + h, pos_j + w) = -1;
				elseif h > height / 2 && w > width / 2     % low-right rectangle
					window(pos_i + h, pos_j + w) = 1;
					window_t(pos_i + h, pos_j + w) = -1;
				else 
					window(pos_i + h, pos_j + w) = -1;
					window_t(pos_i + h, pos_j + w) = 1;
				end
			end
		end
	end




