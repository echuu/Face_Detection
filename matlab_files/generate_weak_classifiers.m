% clear;

dim = 16;
delta_size = 36480;

% weak classifier matrix, each w.c. stored as column vector
delta = zeros(dim * dim, delta_size);

n_features = 5;
features = [2 1; 3 1; 2 2;];

n = 1;
step = 1;
enlarge = 1;


xSize = enlarge * features(1,1);
ySize = features(1,2);
for i = 0:step:dim
    for j = 0:step:dim
        for height = xSize:2:dim-i
            for width = ySize:1:dim-j
               
                [window, window_t]     = create_window(i, j, height, width, dim, 1);
                delta(:, n)            = reshape(window, dim * dim, 1);
                delta(:, n + 1)        = reshape(window_t, dim * dim, 1);

                %disp(['Generating weak classifer ' num2str(n)]);
                %disp(['Generating weak classifer ' num2str(n + 1)]);

                %disp(window);                
                n = n + 2;
            end
        end
    end
end

count2 = 0;
xSize = enlarge * features(2,1);
ySize = enlarge * features(2,2);
for i = 0:step:dim
    for j = 0:step:dim
        for height = xSize:xSize:dim-i
            for width = ySize:ySize:dim-j

                [window, window_t]   = create_window(i, j, height, width, dim, 2);
                delta(:, n)          = reshape(window,   dim * dim, 1);
                delta(:, n + 1)      = reshape(window_t, dim * dim, 1);

                %disp(['Generating weak classifer ' num2str(n)]);
                %disp(['Generating weak classifer ' num2str(n + 1)]);
                %disp(window);
                n = n + 2;
            end
        end
    end
end


xSize = enlarge * features(3,1);
ySize = enlarge * features(3,2);
for i = 0:step:dim
    for j = 0:step:dim
        for height = xSize:xSize:dim-i
            for width = ySize:ySize:dim-j
            
                 [window, window_t] = create_window(i, j, height, width, dim, 3);
             
                %disp(window);
                delta(:, n)     = reshape(window, dim*dim, 1);
                delta(:, n + 1) = reshape(window_t, dim*dim,1);

                %disp(['Generating weak classifer ' num2str(n)]);
                %disp(['Generating weak classifer ' num2str(n + 1)]);

                n = n + 2;
            end
        end
    end
end

