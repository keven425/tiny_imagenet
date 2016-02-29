-- throwing out of memory error


function createModel(nGPU)
   
   local downSample = nn.Sequential() -- branch 1
   downSample:add(cudnn.SpatialConvolution(3,32,5,5,1,1,2,2))       -- 64 -> 32      
   downSample:add(cudnn.ReLU(true))
   downSample:add(cudnn.SpatialMaxPooling(2,2,2,2))
   downSample:add(cudnn.SpatialConvolution(32,64,5,5,1,1,2,2))       -- 32 -> 16      
   downSample:add(cudnn.ReLU(true))
   downSample:add(cudnn.SpatialMaxPooling(2,2,2,2))
   

   local features1 = nn.Concat(2)
   local fb13 = nn.Sequential() -- branch 2
   fb13:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1))       
   fb13:add(cudnn.ReLU(true))
   
   local fb15 = nn.Sequential() -- branch 3
   fb15:add(cudnn.SpatialConvolution(64,64,5,5,1,1,2,2))       
   fb15:add(cudnn.ReLU(true))
   
   local fb17 = nn.Sequential() -- branch 2
   fb17:add(cudnn.SpatialConvolution(64,64,7,7,1,1,3,3))       
   fb17:add(cudnn.ReLU(true))
   
   
   features1:add(fb13)
   features1:add(fb15)
   features1:add(fb17)
   

   local features2 = nn.Concat(2)
   local fb23 = nn.Sequential() -- branch 2
   fb23:add(cudnn.SpatialConvolution(128*2,128,1,1,1,1,0,0))       
   fb23:add(cudnn.SpatialConvolution(128,128,3,3,1,1,1,1))       
   fb23:add(cudnn.ReLU(true))
   
   local fb25 = nn.Sequential() -- branch 3
   fb25:add(cudnn.SpatialConvolution(128*2,64,1,1,1,1,0,0))       
   fb25:add(cudnn.SpatialConvolution(64,64,5,5,1,1,2,2))       
   fb25:add(cudnn.ReLU(true))
   
   local fb27 = nn.Sequential() -- branch 2
   fb27:add(cudnn.SpatialConvolution(128*2,64,1,1,1,1,0,0))       
   fb27:add(cudnn.SpatialConvolution(64,64,7,7,1,1,3,3))       
   fb27:add(cudnn.ReLU(true))
   
   
   features2:add(fb23)
   features2:add(fb25)
   features2:add(fb27)


   local features3 = nn.Concat(2)
   local fb33 = nn.Sequential() -- branch 2
   fb33:add(cudnn.SpatialConvolution(128*2,128,1,1,1,1,0,0))       
   fb33:add(cudnn.SpatialConvolution(128,128,3,3,1,1,1,1))       
   fb33:add(cudnn.ReLU(true))
   
   local fb35 = nn.Sequential() -- branch 3
   fb35:add(cudnn.SpatialConvolution(128*2,64,1,1,1,1,0,0))       
   fb35:add(cudnn.SpatialConvolution(64,64,5,5,1,1,2,2))       
   fb35:add(cudnn.ReLU(true))
   
   local fb37 = nn.Sequential() -- branch 2
   fb37:add(cudnn.SpatialConvolution(128*2,64,1,1,1,1,0,0))       
   fb37:add(cudnn.SpatialConvolution(64,64,7,7,1,1,3,3))       
   fb37:add(cudnn.ReLU(true))
   
   
   features3:add(fb33)
   features3:add(fb35)
   features3:add(fb37)


   local classifier = nn.Sequential()
   classifier:add(nn.View(128*14*14 * 2))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(128*14*14 * 2, 1024))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(1024, nClasses))
   classifier:add(nn.LogSoftMax())


   local model = nn.Sequential():add(downSample):add(features1):add(features2):add(features3):add(classifier)

   return model
end
