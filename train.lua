----------- read in training data:
-- faces.csv     256 x 11838
-- nonfacs.csv   256 x 45356


------- variable info:
-- delta_size = 36480 (number of weak classifiers)
----------------------------------

csv2tensor   =  require 'csv2tensor';
local ext    =  require('externalFunctions');


print('Begin reading in training data');
faces = csv2tensor.load("/home/eric/Face_Detection/faces.csv");
numRows_faces = faces:size()[1];
numCols_faces = faces:size()[2];

print(numCols_faces .. ' of faces (columns)');
print(numRows_faces .. ' pixels each (rows)');

nonfaces = csv2tensor.load("/home/eric/Face_Detection/nonfaces.csv");
numRows_nonfaces = nonfaces:size()[1];
numCols_nonfaces = nonfaces:size()[2];

print(numCols_nonfaces .. ' of nonfaces (columns)');
print(numRows_nonfaces .. ' pixels each (rows)');



-------- generate weak classifiers ---------------------------------------------
delta_size = 36480;
dim = 16;

-- weak classifier matrix, each w.c. stored as column vector 
delta = torch.Tensor(dim * dim, delta_size):zero();

-- populate each column of delta with haar-feature
delta = ext.generateWC(dim, delta_size);
torch.save('delta.dat', delta); -- write out delta matrix to data file
-------- finished generating weak classifiers ----------------------------------


------ calculate threshold -----------------------------------------------------
total_images = ext.NUM_FACES + ext.NUM_NONFACES;

face_mean    = torch.FloatTensor(delta_size, 1):zero();
face_sd      = torch.FloatTensor(delta_size, 1):zero();
nonface_mean = torch.FloatTensor(delta_size, 1):zero();
nonface_sd   = torch.FloatTensor(delta_size, 1):zero();
proj         = torch.FloatTensor(total_images, delta_size):zero();

start_time = os.time();

face_mean, face_sd, nonface_mean, nonface_sd, proj = ext.calcThreshold(delta, 
	delta_size, faces:t(), nonfaces:t());

print('face_mean: ' .. face_mean:size()[1]);
print('face_sd: ' .. face_sd:size()[1]);

print('nonface_mean: ' .. nonface_mean:size()[1]);
print('nonface_sd: ' .. nonface_sd:size()[1]);


end_time = os.time();
elapsed_time = os.difftime(end_time, start_time);
print('total runtime: ' .. elapsed_time .. 'seconds');


print('writing data files');
torch.save('face_mean.dat',     face_mean);
torch.save('face_sd.dat',       face_sd);
torch.save('nonface_mean.dat',  nonface_mean);
torch.save('nonface_sd.dat',    nonface_sd);
torch.save('proj.dat',          proj);
print('finished writing data files');
------ finished calculating threshold ------------------------------------------


----- create training matrix ---------------------------------------------------
Y_train = ext.createTrain(faces, nonfaces);

print('Y_train: ' .. Y_train:size()[1] .. ' results');

torch.save('Y_train.dat', Y_train);
------ finished creating training matrix ---------------------------------------





