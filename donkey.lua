--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local gm = assert(require 'graphicsmagick')
paths.dofile('dataset.lua')
paths.dofile('util.lua')
ffi=require 'ffi'

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, opt.augment ~= 'none' and 'trainCacheAugment.t7' or 'trainCache.t7')
local validateCache = paths.concat(opt.cache, opt.augment ~= 'none' and 'validateCacheAugment.t7' or 'validateCache.t7')
local testCache = paths.concat(opt.cache, opt.augment ~= 'none' and 'testCacheAugment.t7' or 'testCache.t7')
local meanstdCache = paths.concat(opt.cache, opt.augment ~= 'none' and 'meanstdCacheAugment.t7' or 'meanstdCache.t7')

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, 64, 64}
local sampleSize = {3, 64, 64}
if opt.augment ~= 'none' then
   sampleSize = {3, 56, 56}
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
--[[ Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
]]--

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   -- load image with size hints
   local input = gm.Image():load(path, self.loadSize[3], self.loadSize[2])
   -- -- find the smaller dimension, and resize it to 256 (while keeping aspect ratio)
   if opt.augment ~= 'none' then
      local iW, iH = input:size()
      -- do random crop
      local oW = sampleSize[3];
      local oH = sampleSize[2]
      local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
      local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
      input = input:crop(oW, oH, w1, h1)

      -- rotate image slightly, in degrees
      input:rotate(math.floor(torch.uniform(0, 3)) * 8 - 8)
      local rotatedW, rotatedH = input:size()
      local rotatedW1 = (rotatedW - oW) / 2
      input:crop(oW, oH, rotatedW1, rotatedW1)

      -- do hflip with probability 0.5
      if torch.uniform() > 0.5 then input:flop(); end
   end

   local out = input:toTensor('float','RGB','DHW')
   -- mean/std
   for i=1,3 do -- channels
      if mean then out[{{i},{},{}}]:add(-mean[i]) end
      if std then out[{{i},{},{}}]:div(std[i]) end
   end
   return out
end

-- classesList = { 
--          'n02099601', 
--          'n03930313',
--          'n04532670',
--          'n02074367',
--          'n04456115'
--       }

if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.data, 'train')},
      loadSize = {3, 64, 64},
      sampleSize = sampleSize,
      split = 100,
      verbose = true
      -- forceClasses = classesList
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")

end

-- End of train loader section
--------------------------------------------------------------------------------

--[[ Section 2: Create a validate data loader (validateLoader),
   which can iterate over the validate set and returns an image's
   10 crops (center + 4 corners) and their hflips]]--

-- function to load the image, do 10 crops (center + 4 corners) and their hflips
local validateHook = function(self, path)
   local oH = sampleSize[2]
   local oW = sampleSize[3];
   
   local input = gm.Image():load(path, self.loadSize[3], self.loadSize[2])
   local im = input:toTensor('float','RGB','DHW')
   -- mean/std
   for i=1,3 do -- channels
      if mean then im[{{i},{},{}}]:add(-mean[i]) end
      if  std then im[{{i},{},{}}]:div(std[i]) end
   end

   if opt.augment ~= 'none' then

      local out = torch.Tensor(10, 3, oW, oH)
      local iW, iH = input:size()
      local w1 = math.ceil((iW-oW)/2)
      local h1 = math.ceil((iH-oH)/2)
      out[1] = image.crop(im, w1, h1, w1+oW, h1+oW) -- center patch
      out[2] = image.hflip(out[1])
      h1 = 1; w1 = 1;
      out[3] = image.crop(im, w1, h1, w1+oW, h1+oW)  -- top-left
      out[4] = image.hflip(out[3])
      h1 = 1; w1 = iW-oW;
      out[5] = image.crop(im, w1, h1, w1+oW, h1+oW)  -- top-right
      out[6] = image.hflip(out[5])
      h1 = iH-oH; w1 = 1;
      out[7] = image.crop(im, w1, h1, w1+oW, h1+oW)  -- bottom-left
      out[8] = image.hflip(out[7])
      h1 = iH-oH; w1 = iW-oW;
      out[9] = image.crop(im, w1, h1, w1+oW, h1+oW)  -- bottom-right
      out[10] = image.hflip(out[9])

      local out_rotated = torch.Tensor(30, 3, oW, oH)
      for i = 1,10 do
         out_rotated[i] = out[i]
         out_rotated[i + 10] = image.rotate(out[i], -8)
         out_rotated[i + 20] = image.rotate(out[i], 8)
      end
      return out_rotated
      -- return out
   else
      return im
   end
end

if paths.filep(validateCache) then
   print('Loading validate metadata from cache')
   validateLoader = torch.load(validateCache)
   validateLoader.sampleHookTest = validateHook
else
   print('Creating validate metadata')
   validateLoader = dataLoader{
      paths = {paths.concat(opt.data, 'val')},
      loadSize = {3, 64, 64},
      sampleSize = sampleSize,
      split = 0,
      verbose = true,
      forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and validateLoader
      -- forceClasses = classesList -- force consistent class indices between trainLoader and validateLoader
   }
   torch.save(validateCache, validateLoader)
   validateLoader.sampleHookTest = validateHook
end
collectgarbage()
-- End of validate loader section





paths.dofile('testDataset.lua')
if paths.filep(testCache) then
   print('Loading test metadata from cache')
   testLoader = torch.load(testCache)
   testLoader.sampleHookTest = validateHook
else
   print('Creating test metadata')
   testLoader = testDataLoader{
      paths = {paths.concat(opt.data, 'test')},
      loadSize = {3, 64, 64},
      sampleSize = sampleSize,
      verbose = true
   }
   torch.save(testCache, testLoader)
   testLoader.sampleHookTest = validateHook
end
collectgarbage()




-- Estimate the per-channel mean/std (so that the loaders can normalize appropriately)
if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
else
   local tm = torch.Timer()
   local nSamples = 10000
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local meanEstimate = {0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)
      for j=1,3 do
         meanEstimate[j] = meanEstimate[j] + img[j]:mean()
      end
   end
   for j=1,3 do
      meanEstimate[j] = meanEstimate[j] / nSamples
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local stdEstimate = {0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)
      for j=1,3 do
         stdEstimate[j] = stdEstimate[j] + img[j]:std()
      end
   end
   for j=1,3 do
      stdEstimate[j] = stdEstimate[j] / nSamples
   end
   std = stdEstimate

   local cache = {}
   cache.mean = mean
   cache.std = std
   torch.save(meanstdCache, cache)
   print('Time to estimate:', tm:time().real)
end
print('Mean: ', mean[1], mean[2], mean[3], 'Std:', std[1], std[2], std[3])

do -- just check if mean/std look good now
   local validateMean = 0
   local validateStd = 0
   for i=1,100 do
      local img = trainLoader:sample(1)
      validateMean = validateMean + img:mean()
      validateStd  = validateStd + img:std()
   end
   print('Stats of 100 randomly sampled images after normalizing. Mean: ' .. validateMean/100 .. ' Std: ' .. validateStd/100)
end
