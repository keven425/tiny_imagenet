--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local testDataIterator = function()
   testLoader:reset()
   return function() return testLoader:get_batch(false) end
end

local batchNumber
local top1_center, top5_center, loss
local top1_10crop, top5_10crop
local timer = torch.Timer()

local outputHandle = assert(io.open('kvw.txt', 'w'))

function test()
   print('==> doing epoch on test data:')

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center = 0; top5_center = 0
   top1_10crop = 0; top5_10crop = 0
   loss = 0
   print('nTest: ' .. nTest)
   print('testBatchSize: ' .. opt.testBatchSize)
   for i=1, nTest/opt.testBatchSize do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.testBatchSize + 1
      local indexEnd = (indexStart + opt.testBatchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, names = testLoader:get(indexStart, indexEnd)
            -- local i_stg = tonumber(ffi.cast('intptr_t', torch.pointer(inputs:storage())))
            -- local l_stg =  torch.pointer(names:storage())
            -- inputs:cdata().storage = nil
            return sendTensor(inputs), names
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
      donkeys:synchronize()
      collectgarbage()
   end

   donkeys:synchronize()
   cutorch.synchronize()

   for i = 0,9999 do
    local filename = 'test_' .. i .. '.JPEG'
    local label = filename2label[filename] or 0
    outputHandle:write(filename .. ' ' .. label .. '\n')
   end

   print('test finished\n')


end -- of test()
-----------------------------------------------------------------------------
local inputsCPU = torch.Tensor()
-- local namesCPU = torch.CharTensor(opt.validateBatchSize)
local inputs = torch.CudaTensor()
-- local names = torch.CudaTensor(opt.testBatchSize)

filename2label = {}

function testBatch(inputsThread, names)
  cutorch.synchronize()
   batchNumber = batchNumber + opt.testBatchSize
  
   receiveTensor(inputsThread, inputsCPU)
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   
   local outputs = model:forward(inputs)
   cutorch.synchronize()
   
   local pred = outputs:float()

   local function updateTestResult(prob, name)
        _,p = prob:sort(1, true)
        local predictedLabel = classes[p[1]]
        print('predictedLabel index: ' .. p[1])
        _, filename = string.match(name, "(.-)([^//]-([^%.]+))$")
        filename2label[filename] = predictedLabel
   end

    if opt.augment ~= 'none' then
        for i=1,pred:size(1),30 do
            local porbs = pred[{{i, i+29}, {}}]  
            local tencrop = porbs:sum(1)[1]
            updateTestResult(tencrop, names[i])
        end
    else
        for i=1,pred:size(1) do
            updateTestResult(pred[i], names[i])
        end
    end
end
