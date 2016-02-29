function createModel(nGPU)
   assert(nGPU == 1 or nGPU == 2, '1-GPU or 2-GPU supported for Kevenet')
   local features
   if nGPU == 1 then
      features = nn.Concat(2)
   else
      require 'fbnn'
      require 'fbcunn'
      features = nn.ModelParallel(2)
   end

   local fb1 = nn.Sequential() -- branch 1
   fb1:add(cudnn.SpatialConvolution(3,32,5,5,1,1,2,2))       -- 64 -> 64
   fb1:add(cudnn.ReLU(true))
   fb1:add(cudnn.SpatialMaxPooling(2,2,2,2))                   -- 64 ->  32
   fb1:add(cudnn.SpatialConvolution(32,32,5,5,1,1,2,2))       --  32 -> 32
   fb1:add(cudnn.ReLU(true))
   fb1:add(cudnn.SpatialConvolution(32,64,5,5,1,1,2,2))      --  32 ->  32
   fb1:add(cudnn.ReLU(true))
   fb1:add(cudnn.SpatialConvolution(64,64,5,5,1,1,2,2))      --  32 ->  32
   fb1:add(cudnn.ReLU(true))

   local fb2 = fb1:clone() -- branch 2
   for k,v in ipairs(fb2:findModules('cudnn.SpatialConvolution')) do
      v:reset() -- reset branch 2's weights
   end

   features:add(fb1)
   features:add(fb2)

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(128*32*32))
   -- classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(128*32*32, 1024))
   classifier:add(nn.Threshold(0, 1e-6))
   -- classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(1024, nClasses))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)

   return model
end
