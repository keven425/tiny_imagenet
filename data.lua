--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local ffi = require 'ffi'
local Threads = require 'threads'

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------

nClasses = nil
classes = nil
nValidate = 0
nTest = 0

loadSize   = {3, 64, 64}
sampleSize = {3, 64, 64}
if opt.augment ~= 'none' then
   sampleSize = {3, 56, 56}
end

do -- start K datathreads (donkeys)
   if opt.nDonkeys > 0 then
      local options = opt -- make an upvalue to serialize over to donkey threads
      donkeys = Threads(
         opt.nDonkeys,
         function()
            gsdl = require 'sdl2'
            require 'torch'
         end,
         function(idx)
            opt = options -- pass to all donkeys via upvalue
            tid = idx
            local seed = opt.manualSeed + idx
            torch.manualSeed(seed)
            print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            paths.dofile('donkey.lua')
         end
      );
   else -- single threaded data loading. useful for debugging
      paths.dofile('donkey.lua')
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end

donkeys:synchronize()

donkeys:addjob(function() return testLoader:size() end, function(c) nTest = c end)
print('nTest: ' .. nTest)
donkeys:synchronize()

donkeys:addjob(function() return trainLoader.classes end, function(c) classes = c end)
donkeys:synchronize()
nClasses = #classes
assert(nClasses, "Failed to get nClasses")
print('nClasses: ', nClasses)
torch.save(paths.concat(opt.save, 'classes.t7'), classes)

donkeys:addjob(function() return validateLoader:sizeValidate() end, function(c) nValidate = c end)
donkeys:synchronize()
assert(nValidate > 0, "Failed to get nValidate")
print('nValidate: ', nValidate)
-- end
