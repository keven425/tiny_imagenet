--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local defaultDir = paths.concat(os.getenv('HOME'), 'fbcunn_imagenet')

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache',
               defaultDir ..'/imagenet_runs',
               'subdirectory in which to save/log experiments')
    cmd:option('-data',
               defaultDir .. '/imagenet_raw_images/256',
               'Home of ImageNet dataset')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | fbcunn | cunn')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        2, 'number of donkeys to initialize (data loading threads)')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         55,    'Number of total epochs to run')
    cmd:option('-epochSize',       100000 / 256, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       256,   'mini-batch size (1 = pure stochastic)')
    cmd:option('-validateBatchSize',    16,   'mini-batch size for validating')
    cmd:option('-testBatchSize',    16,   'mini-batch size for testing')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.01, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-LRD',    0.05, 'learning rate decay')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     0, 'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'kevenet', 'Options: alexnet | overfeat | kevenet')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:option('-testMode', 'none', 'boolean flag. if true, output <filename> <predicted_label> on each test image into predicted.txt')
    cmd:option('-augment',  'none', 'augment training samples?')
    cmd:option('-visualize',  'none', 'visualize weights?')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.version = cmd:string('kevenet', opt, {retrain=true, optimState=true, cache=true, data=true})
    opt.save = paths.concat(opt.cache, opt.version)
    -- add date/time
    opt.save = paths.concat(opt.save, ',' .. os.date():gsub(' ',''))
    return opt
end

return M
