----------- read in training data:
-- faces.csv     256 x 11838
-- nonfacs.csv   256 x 45356


------- variable info:
-- delta_size = 36480 (number of weak classifiers)
----------------------------------

--csv2tensor   =  require 'csv2tensor';
local ext    =  require('externalFunctions');
local boost  =  require('slower_adaboost');
local debug  = 0;

print('Begin reading in training data');
-- faces = csv2tensor.load("/home/wayne/Face_Detection/faces.csv");
pathname = "/home/wayne/Desktop/data_files/";
faces = torch.load(pathname..'faces.dat');
faces = faces:t();


--nonfaces = csv2tensor.load("/home/wayne/Face_Detection/faces.csv");
nonfaces = torch.load(pathname..'nonfaces.dat');
nonfaces = nonfaces:t();

-- X = torch.cat(faces:t(), nonfaces:t(), 1); -- total_imgs x 256

if debug == 1 then
	numRows_faces = faces:size()[1];
	numCols_faces = faces:size()[2];

	print(numCols_faces .. ' of faces (columns)');
	print(numRows_faces .. ' pixels each (rows)');

	numRows_nonfaces = nonfaces:size()[1];
	numCols_nonfaces = nonfaces:size()[2];

	print(numCols_nonfaces .. ' of nonfaces (columns)');
	print(numRows_nonfaces .. ' pixels each (rows)');

	--total_rows = X:size()[1];
	--total_cols = X:size()[2];

	--print('X : '..total_rows.. ' x '..total_cols);
end

-------- generate weak classifiers ---------------------------------------------
delta_size = 36480;
dim = 16;

-- weak classifier matrix, each w.c. stored as column vector 
delta = torch.Tensor(dim * dim, delta_size):zero();

-- populate each column of delta with haar-feature
delta = ext.generateWC(dim, delta_size);
--torch.save('delta.dat', delta); -- write out delta matrix to data file
-------- finished generating weak classifiers ----------------------------------


------ calculate threshold -----------------------------------------------------
total_images = ext.NUM_FACES + ext.NUM_NONFACES;

--face_mean    = torch.FloatTensor(delta_size, 1):zero();
--face_sd      = torch.FloatTensor(delta_size, 1):zero();
--nonface_mean = torch.FloatTensor(delta_size, 1):zero();
--nonface_sd   = torch.FloatTensor(delta_size, 1):zero();
--proj         = torch.FloatTensor(total_images, delta_size):zero();

print('begin calculating threshold');
start_time = os.time();

--[[
face_mean, face_sd, nonface_mean, nonface_sd, proj = ext.calcThreshold(delta, 
	delta_size, faces, nonfaces);
--]]
pathname = "/home/wayne/Desktop/data_files/";
proj         = torch.load(pathname..'projections.dat');
face_mean    = torch.load(pathname..'face_mean.dat');
nonface_mean = torch.load(pathname..'nonface_mean.dat');
face_sd      = torch.load(pathname..'face_sd.dat');
nonface_sd   = torch.load(pathname..'nonface_sd.dat');
Y_train      = torch.load(pathname..'Y_train.dat');


if debug == 1 then
	print('rows of projection: '..proj:size()[1]);
	print('cols of projection: '..proj:size()[2]);


	print('face_mean size: ' .. face_mean:size()[1]);
	print('face_sd size: ' .. face_sd:size()[1]);

	print('nonface_mean size: ' .. nonface_mean:size()[1]);
	print('nonface_sd size: ' .. nonface_sd:size()[1]);
end

end_time = os.time();
elapsed_time = os.difftime(end_time, start_time);
print('total runtime: ' .. elapsed_time .. 'seconds');

if debug == 1 then
	print('writing data files');
	torch.save('face_mean.dat',     face_mean);
	torch.save('face_sd.dat',       face_sd);
	torch.save('nonface_mean.dat',  nonface_mean);
	torch.save('nonface_sd.dat',    nonface_sd);
	torch.save('projections.dat',   proj);
	print('finished writing data files');
end
------ finished calculating threshold ------------------------------------------


----- create training matrix ---------------------------------------------------

if debug == 1 then
	Y_train = ext.createTrain(faces, nonfaces);
	print('Y_train: ' .. Y_train:size()[1] .. ' results');
	torch.save('Y_train.dat', Y_train);
end
------ finished creating training matrix ---------------------------------------


----- free up memory ----------------
faces    = nil;
nonfaces = nil;
delta    = nil;
----- end free memory ---------------


------- run adaboost -----------------
T = 100;
boost.adaboost(proj, face_mean, nonface_mean, 
	face_sd, nonface_sd, Y_train, T);
