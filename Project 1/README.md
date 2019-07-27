# Assignment 1B 

## 1. What are Channels and Kernels (according to EVA)?

**Channels**: 

A Channel represents different components of an image or feature maps. For example: for a normal coloured image, there are three channels, R (Red), G (Green) and B (Blue). And suppose we have convolved this image a (3x3x3)x32 kernels, then, the resultant image or feature map would have 32 channels. Also, the   each (3x3x3)  kernel has 3 channels one for each channel in the image. Therefore, in the next convolution layer, each kernel should have 32 channels as the convolved image has 32 channels.

**Kernels:**

A kernel is an MxMxN dimensional matrix which extracts information from the image. N depends on the number of channels in the input image for that convolutional layer. If we are considering the kernel for first convolution layer whose input is the original image itself, then, that kernel should have 3 channels. Each channel in a kernel extracts different property of the image. For example, one channel might extract the vertical edges in the image, other might extract horizontal edges, etc. Also, mostly the value of M used is 3 for the reasons mentioned in the answer below for question 2. 

## 2. Why should we only (well mostly) use 3x3 Kernels?

1. 3x3 kernels leads to lesser number of parameters to be trained for the model. For instance, if we are using 5x5x3 kernel (for first stage where the original coloured image is the input), then, we will have 75 parameters. However, we can have the same receptive field as that of 5x5x3 kernel by using two 3x3x3 kernel which would require training of 54 parameters which is significantly lesser as compared to 5x5x3 kernel.

2. GPUs are optimised to compute faster for 3x3 kernels as compared to the other sizes. So, again the computation will be faster as compared to other kernels.

3. Output from using 3x3 kernel is symmetric in the sense that pixel obtained after convolution can be attributed to a particular sub-matrix in the image as shown in the image below. The output -3 corresponds to that green shaded area in the input image and all the pixels in the input image are symmetrically arranged around the output pixel. Whereas in the case of 2x2 or other even sized filters this is not possible which leads to distortion across the layers.  

   ![sym_3x3](sym_3x3.png)

## 3. How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)

Mx corresponds to the size of the image after xth convolution layer.

Step 1. M1  = (199 - 3 )/1 +1 = 197

Step 2. M2 = 197 - 3 + 1 = 195

Basically, this is an arithmetic progression. So, we will find the number of step required using the below formula:

​	an = a1 + (n-1) * d

​	Here, an = 199,

​		a1 = 1, 

​		and, d  = 2  

N = (199 - 1)/2 +1 = 100

So, we need to perform 100 steps to reach 1x1 from 199x199.