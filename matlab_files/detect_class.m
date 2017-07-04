% detect_class.m
% run this script to run classifier on class images
test_image = rgb2gray(imread('test/Test_Image_1.jpg'));
figure;
imshow(test_image);
hold on;
step = 4;
count = 0;
for i = 1:5
    multiplier = 0.25 - (i-1) * 0.047;
    test_image_mod = imresize(test_image, multiplier);
    
    for j = 1:step:size(test_image_mod,1)-dim
        for k = 1:step:size(test_image_mod,2)-dim

            s = test_image_mod(j:j+dim-1, k:k+dim-1);
            s_vec = double(reshape(s,dim*dim,1));

            % projection of image onto T weak classifiers
            result = delta(:,min_ada_index)' * s_vec; % T x 1

            F = 0;
            for t = 1:T
                [h, ~] = gauss_classify(result(t),...
                    delta_face_means(min_ada_index(t)),...
                    delta_face_sd(min_ada_index(t)),...
                    delta_nonface_means(min_ada_index(t)),...
                    delta_nonface_sd(min_ada_index(t)));
                
                % weight classification, with alpha (?)
                % calculate strong classifier
                F = F + alpha(t) .* h;
            end
            
            if F > 0
                box = 1/multiplier * [k j 24 24];
                if i == 1
                    c = 'blue';
                elseif i == 2
                    c = 'red';
                elseif i == 3
                    c = 'yellow';
                elseif i == 4
                    c = 'cyan';
                elseif i == 5
                    c = 'white';
                end
                rectangle('Position', box, 'LineWidth', 2 , 'EdgeColor', c);
                disp(['face detected ' num2str(k) ' ' num2str(j)]);
                count = count + 1;
            end
        end
    end
    
    
end
hold off;