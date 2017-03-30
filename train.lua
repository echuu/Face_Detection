----------- read in training data:
-- faces.csv     256 x 11838
-- nonfacs.csv   256 x 45356


------- variable info:
-- delta_size = 36480 (number of weak classifiers)
----------------------------------

--csv2tensor   =  require 'csv2tensor';
local ext    =  require('externalFunctions');
local boost  =  require('adaboost');
local load   = require('load_data')

local debug  = 0;
local FIRST_TIME_RUN = 0;

local subset_faces    = 800;
local subset_nonfaces = 3200;

print('Begin reading in training data');
--faces = csv2tensor.load("/home/wayne/Face_Detection/faces.csv");
--nonfaces = csv2tensor.load("/home/wayne/Face_Detection/nonfaces.csv");

pathname = "/home/eric/data_files/";
faces    = torch.load(pathname..'faces.dat');
faces    = faces[{{},{1,subset_faces}}];
faces    = faces:t();
nonfaces = torch.load(pathname..'nonfaces.dat');
nonfaces = nonfaces[{{},{1,subset_nonfaces}}];
nonfaces = nonfaces:t();


local num_faces = faces:size()[1];
local num_nonfaces = nonfaces:size()[1];


if debug == 0 then
	numRows_faces = faces:size()[1];
	numCols_faces = faces:size()[2];

	numRows_nonfaces = nonfaces:size()[1];
	numCols_nonfaces = nonfaces:size()[2];

	print(numRows_faces .. ' of faces (columns)');
	print(numCols_faces .. ' pixels each (rows)');

	print(numRows_nonfaces .. ' of nonfaces (columns)');
	print(numCols_nonfaces .. ' pixels each (rows)');
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
total_images = num_faces + num_nonfaces;

--face_mean    = torch.FloatTensor(delta_size, 1):zero();
--face_sd      = torch.FloatTensor(delta_size, 1):zero();
--nonface_mean = torch.FloatTensor(delta_size, 1):zero();
--nonface_sd   = torch.FloatTensor(delta_size, 1):zero();
--proj         = torch.FloatTensor(total_images, delta_size):zero();

print('begin calculating threshold');



if FIRST_TIME_RUN == 1 then
	start_time = os.time();

	face_mean, face_sd, nonface_mean, nonface_sd, proj = ext.calcThreshold(delta, 
		delta_size, faces, nonfaces);

	end_time = os.time();
	elapsed_time = os.difftime(end_time, start_time);
	print('total runtime: ' .. elapsed_time .. ' seconds');

	print('writing data files');

	torch.save('face_mean.dat',     face_mean);
	torch.save('face_sd.dat',       face_sd);
	torch.save('nonface_mean.dat',  nonface_mean);
	torch.save('nonface_sd.dat',    nonface_sd);
	torch.save('projections.dat',   proj);

	print('finished writing data files');
elseif FIRST_TIME_RUN == 0 then
	--pathname1    = "/home/wayne/Desktop/data_files/";
	--pathname     = '/home/eric/Desktop/data_files/';
	pathname = '';
	proj         = torch.load(pathname..'projections.dat');
	face_mean    = torch.load(pathname..'face_mean.dat');
	nonface_mean = torch.load(pathname..'nonface_mean.dat');
	face_sd      = torch.load(pathname..'face_sd.dat');
	nonface_sd   = torch.load(pathname..'nonface_sd.dat');
	Y_train      = torch.load(pathname..'Y_train.dat');
end


if debug == 0 then
	print('rows of projection: '..proj:size()[1]);
	print('cols of projection: '..proj:size()[2]);


	print('face_mean size: ' .. face_mean:size()[1]);
	print('face_sd size: ' .. face_sd:size()[1]);

	print('nonface_mean size: ' .. nonface_mean:size()[1]);
	print('nonface_sd size: ' .. nonface_sd:size()[1]);
end
------ finished calculating threshold ------------------------------------------


----- create training matrix ---------------------------------------------------

if debug == 0 then
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
T = 10;
boost.adaboost(proj, face_mean, nonface_mean, 
	face_sd, nonface_sd, Y_train, T);
