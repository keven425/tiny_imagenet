function createModel(nGPU)
   
   -- 1.1
   local features1 = nn.Concat(2)
   
   local fb11 = nn.Sequential() -- branch 1
   fb11:add(cudnn.SpatialConvolution(3,64,3,3,1,1,1,1))       -- 64 -> 64      
   fb11:add(cudnn.ReLU(true))
   fb11:add(cudnn.SpatialMaxPooling(2,2,2,2))
   fb11:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1))       
   fb11:add(cudnn.ReLU(true))
   fb11:add(cudnn.SpatialMaxPooling(2,2,2,2))
   fb11:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1))      
   fb11:add(cudnn.ReLU(true))
   fb11:add(nn.Reshape(128 * 14 * 14, true))

   local fb12 = nn.Sequential() -- branch 2
   fb12:add(cudnn.SpatialMaxPooling(2,2,2,2))                   -- 64 ->  32   
   fb12:add(cudnn.SpatialConvolution(3,64,3,3,1,1,1,1))       
   fb12:add(cudnn.ReLU(true))
   fb12:add(cudnn.SpatialMaxPooling(2,2,2,2))
   fb12:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1))       
   fb12:add(cudnn.ReLU(true))
   fb12:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1))      
   fb12:add(cudnn.ReLU(true))
   fb12:add(nn.Reshape(128 * 14 * 14, true))

   local fb13 = nn.Sequential() -- branch 3
   fb13:add(cudnn.SpatialMaxPooling(4,4,4,4))                   -- 64 ->  16
   fb13:add(cudnn.SpatialConvolution(3,64,3,3,1,1,1,1))       
   fb13:add(cudnn.ReLU(true))
   fb13:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1))       
   fb13:add(cudnn.ReLU(true))
   fb13:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1))      
   fb13:add(cudnn.ReLU(true))
   fb13:add(nn.Reshape(128 * 14 * 14, true))

   local fb4 = nn.Sequential() -- branch 2
   fb4:add(cudnn.SpatialMaxPooling(8,8,8,8))                   -- 64 ->  8
   fb4:add(cudnn.SpatialConvolution(3,64,3,3,1,1,1,1))       
   fb4:add(cudnn.ReLU(true))
   fb4:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1))       
   fb4:add(cudnn.ReLU(true))
   fb4:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1))      
   fb4:add(cudnn.ReLU(true))
   fb4:add(nn.Reshape(128 * 7 * 7, true))

   features1:add(fb11)
   features1:add(fb12)
   features1:add(fb13)
   features1:add(fb4)




   -- local features2 = nn.Concat(2)
   
   -- local fb21 = nn.Sequential() -- branch 1
   -- fb21:add(cudnn.SpatialConvolution(3*128,128,1,1,1,1,0,0))       
   -- fb21:add(cudnn.ReLU(true))
   -- fb21:add(cudnn.SpatialConvolution(128,128,3,3,1,1,1,1))       
   -- fb21:add(cudnn.ReLU(true))
   -- fb21:add(cudnn.SpatialConvolution(128,128,3,3,1,1,1,1))      
   -- fb21:add(cudnn.ReLU(true))
   -- fb21:add(nn.View(-1):setNumInputDims(2))

   -- local fb22 = nn.Sequential() -- branch 2
   -- fb22:add(cudnn.SpatialConvolution(3*128,128,1,1,1,1,0,0))       
   -- fb22:add(cudnn.ReLU(true))
   -- fb22:add(cudnn.SpatialConvolution(128,128,3,3,1,1,1,1))       
   -- fb22:add(cudnn.ReLU(true))
   -- fb22:add(cudnn.SpatialConvolution(128,128,3,3,1,1,1,1))      
   -- fb22:add(cudnn.ReLU(true))
   -- fb22:add(nn.View(-1):setNumInputDims(2))

   -- local fb23 = nn.Sequential() -- branch 3
   -- fb23:add(cudnn.SpatialConvolution(3*128,128,1,1,1,1,0,0))       
   -- fb23:add(cudnn.ReLU(true))
   -- fb23:add(cudnn.SpatialConvolution(128,128,3,3,1,1,1,1))       
   -- fb23:add(cudnn.ReLU(true))
   -- fb23:add(cudnn.SpatialConvolution(128,128,3,3,1,1,1,1))      
   -- fb23:add(cudnn.ReLU(true))
   -- fb23:add(nn.View(-1):setNumInputDims(2))

   -- features2:add(fb21)
   -- features2:add(fb22)
   -- features2:add(fb23)





   local classifier = nn.Sequential()
   
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(128*14*14 * 3 + 128 * 7 * 7, 1024))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(1024, nClasses))
   classifier:add(nn.LogSoftMax())



   local model = nn.Sequential():add(features1):add(classifier)

   return model
end
