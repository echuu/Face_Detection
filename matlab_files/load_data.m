% load_data.m

faces = zeros(FACE_DIM, FACE_SIZE);
nonfaces = zeros(FACE_DIM, NONFACE_SIZE);

for i = 1:FACE_SIZE
    face_name = '/home/eric/Documents/face_files/faces/face16_%06d.bmp';
    raw = rgb2gray(imread(sprintf(face_name, i)));
    faces(:, i) = reshape(raw, FACE_DIM, 1);
    if mod(i, 1000) == 0
    	disp(['importing face ' int2str(i)]);
    end
end

for i = 1:NONFACE_SIZE
    nonface_name = '/home/eric/Documents/face_files/nonfaces/nonface16_%06d.bmp';
    raw = rgb2gray(imread(sprintf(nonface_name, i)));
    nonfaces(:, i) = reshape(raw, FACE_DIM, 1);
    if mod(i, 1000) == 0
	    disp(['importing nonface ' int2str(i)]);
	end
end

csvwrite('faces.csv', faces);
csvwrite('nonfaces.csv', nonfaces);

