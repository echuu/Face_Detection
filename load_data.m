face_size = 11838;
nonface_size = 45356;

face_size = 4000;
nonface_size = 5000;


face_dims = 16 * 16;


faces = zeros(face_dims, face_size);
nonfaces = zeros(face_dims, nonface_size);

for i = 1:face_size
    raw = rgb2gray(imread(sprintf('/home/eric/Documents/face_files/faces/face16_%06d.bmp', i)));
    faces(:, i) = reshape(raw, face_dims, 1);
    if mod(i, 1000) == 0
    	disp(['importing face ' int2str(i)]);
    end
end

for i = 1:nonface_size
    raw = rgb2gray(imread(sprintf('/home/eric/Documents/face_files/nonfaces/nonface16_%06d.bmp', i)));
    nonfaces(:, i) = reshape(raw, face_dims, 1);
    if mod(i, 1000) == 0
	    disp(['importing nonface ' int2str(i)]);
	end
end

csvwrite('faces.csv', faces);
csvwrite('nonfaces.csv', nonfaces);

