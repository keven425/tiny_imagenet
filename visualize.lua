if opt.visualize ~= 'visualize' then
	require 'image'
	imageTensor = image.toDisplayTensor{ 
						image=model:get(2).get(1).get(1).weight, zoom=4, nrow=10,
                                min=-1, max=1, legend='layer 1: weights', padding=1 }
    image.save('weights.png', imageTensor)
    os.exit()
end
