function createModel(nGPU)
   
   -- 1.1
   local features = nn.Concat(2)
   
   local fb1 = nn.Sequential() -- branch 1
   fb1:add(cudnn.SpatialConvolution(3,64,3,3,1,1,1,1))       -- 64 -> 64      
   fb1:add(cudnn.ReLU(true))
   fb1:add(cudnn.SpatialMaxPooling(2,2,2,2))
   fb1:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1))       
   fb1:add(cudnn.ReLU(true))
   fb1:add(cudnn.SpatialMaxPooling(2,2,2,2))
   fb1:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1))      
   fb1:add(cudnn.ReLU(true))
   fb1:add(nn.View(128*14*14))
   fb1:add(nn.Dropout(0.5))
   fb1:add(nn.Linear(128*14*14, 512))
   fb1:add(nn.Threshold(0, 1e-6))
   fb1:add(nn.Dropout(0.5))
   

   local fb2 = nn.Sequential() -- branch 2
   fb2:add(cudnn.SpatialMaxPooling(2,2,2,2))                   -- 64 ->  32   
   fb2:add(cudnn.SpatialConvolution(3,64,3,3,1,1,1,1))       
   fb2:add(cudnn.ReLU(true))
   fb2:add(cudnn.SpatialMaxPooling(2,2,2,2))
   fb2:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1))       
   fb2:add(cudnn.ReLU(true))
   -- fb2:add(cudnn.SpatialMaxPooling(2,2,2,2))
   fb2:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1))      
   fb2:add(cudnn.ReLU(true))
   fb2:add(nn.View(128*14*14))
   fb2:add(nn.Dropout(0.5))
   fb2:add(nn.Linear(128*14*14, 512))
   fb2:add(nn.Threshold(0, 1e-6))
   fb2:add(nn.Dropout(0.5))

   local fb3 = nn.Sequential() -- branch 2
   fb3:add(cudnn.SpatialMaxPooling(4,4,4,4))                   -- 64 ->  16
   fb3:add(cudnn.SpatialConvolution(3,64,3,3,1,1,1,1))       
   fb3:add(cudnn.ReLU(true))
   fb3:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1))       
   fb3:add(cudnn.ReLU(true))
   fb3:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1))      
   fb3:add(cudnn.ReLU(true))
   fb3:add(nn.View(128*14*14))
   fb3:add(nn.Dropout(0.5))
   fb3:add(nn.Linear(128*14*14, 512))
   fb3:add(nn.Threshold(0, 1e-6))
   fb3:add(nn.Dropout(0.5))

   local fb4 = nn.Sequential() -- branch 2
   fb4:add(cudnn.SpatialMaxPooling(8,8,8,8))                   -- 64 ->  8
   fb4:add(cudnn.SpatialConvolution(3,64,3,3,1,1,1,1))       
   fb4:add(cudnn.ReLU(true))
   fb4:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1))       
   fb4:add(cudnn.ReLU(true))
   fb4:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1))      
   fb4:add(cudnn.ReLU(true))
   fb4:add(nn.View(128*7*7))
   fb4:add(nn.Dropout(0.5))
   fb4:add(nn.Linear(128*7*7, 512))
   fb4:add(nn.Threshold(0, 1e-6))
   fb4:add(nn.Dropout(0.5))

   features:add(fb1)
   features:add(fb2)
   features:add(fb3)
   features:add(fb4)

   -- 1.3. Create Classifier (fully connected layers)
   -- local classifier = nn.Sequential()
   -- classifier:add(nn.View(128*64*64))
   -- classifier:add(nn.Dropout(0.5))
   -- -- classifier:add(nn.Linear(128*64*64 + 128*32*32 + 128*16*16 + 128*8*8, 1024))
   -- classifier:add(nn.Linear(128*64*64, 1024))
   -- classifier:add(nn.Threshold(0, 1e-6))
   -- classifier:add(nn.Dropout(0.5))
   -- classifier:add(nn.Linear(1024, 1024))
   -- classifier:add(nn.Threshold(0, 1e-6))
   -- classifier:add(nn.Linear(1024, nClasses))
   -- classifier:add(nn.LogSoftMax())

   local classifier = nn.Sequential()
   classifier:add(nn.Linear(512 * 4, nClasses))
   classifier:add(nn.LogSoftMax())



   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)

   return model
end
