# Assignment 12

1. ### How to train your ResNet by David Page

   Original Submission by Ben Johnson to DAWNBench dataset, which was state-of-the-art at that time, had following innovations:

    	1. Mixed-precision training
    	2. Smaller ResNet Network (ResNet-18)
    	3. Higher learning  rates as suggested by Leslie Smith
    	4. One  Cycle Learning Rate policy

   Ben reached 94% accuracy in under 6 minutes (341s) on a single V100 GPU. David explains that at 100% compute efficiency, training should complete in 40 seconds which leaves lot of room to improve the 341s state-of-the-art. 

   #### **Post 1: Baseline**

   In the **Baseline** post, he mentions to include the following changes to the existing code by Ben to reduce the training time from 341s to 297s: 

   1. Remove the two consecutive BN-RELU groups after the first convolution.
   2. Remove the strange kink in the learning rate at epoch 15. With these two changes only the training time reduced to 323s
   3. Common image pre-processing tasks like padding, normalisation, transposition, etc. which are required for every epoch can be done once before starting training which further reduces the training time by 15s
   4. For selecting data augmentation strategies during training, millions of calls are made to the random number generator and by combining these into a small number of bulk calls at the start of the epoch, he is able to save another 7s. 
   5. By using only the main thread for data augmentation can save off another 4s leading to final time of 297s



   #### **Post 2**: Mini-Batches

    	1. Current training batch size is 128. In DavidNet, they increased batch size to 512s, but hyperparameters need to be adjusted for this increase in batch size
    	2. There is some heavy optimisation jargon there and the key takeaways are :
         - If you want to train neural network faster for a larger datatset, then, you can go for higher batch sizes (to depress the forgetfulness effect)  and small learning rates, but at very large batch sizes, learning would become unstable
         - If you want to train a neural network faster for a smaller dataset, then, you can go for higher learning rates with small batch size but you will need to mitigate the forgetfullness effects there. At higher batch sizes, you will need lower learning rates but curvature effects needs to be mitigated there. 
    	3. For CIFAR-10 dataset, david employed 512 batch size with 10% increase in learning rate which completed the training in 256s reaching 94% accuracy.

   #### Post 3: Regularisation

      1. David computes process times for each process in the network which you can see in the chart below. It is clear that BN takes a lot of computation time and also convolution is taking considerable more time. Dataloader and optimisation steps aren't taking much time.![](https://github.com/sukant16/DS_utils/blob/master/EVA/Process%20Times.png?raw=true)
      2. Converting batch norm wrights back to single precision takes care of the long times for BN. This was because default method of converting a model to half precision in PyTorch (as of version 0.4) triggers a slow code path which doesn’t use the optimized CuDNN routine. Now, it takes 186s to reach 94% accuracy in 35 epochs
      3. Adding Cutout regularisation (8x8 square pixels area) along with other data augmentations (padding, clipping and horizontal flipping) led to 94.3 % median accuracy in 5 runs in 35 epochs. 
      4. Making the LR reaching the peak at 8th instead of the 12th epoch and having linear decay in LR bring the median run accuracy to 94.5%. If number of epochs are reduced to 30, 4/5 of runs reach 94% accuracy.
      5. Pushing batch size to 768 with number of epochs at 30, training time reduces to 154s

### Post 4: Architecture

1. ![](https://github.com/sukant16/DS_utils/blob/master/EVA/Post4.png?raw=true)

Above table shows sequential growth of architecture where he as stripped the original network of residual connections. Later on, he runs more experiments with and without residual connections and find the below architecture as the best one which upon training for 24 epochs gave 94% accuracy in 7 out of 10 runs in 79s.

![img](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/residualL1L3.svg)

### Post 5: Hyperparameters Tuning

This post talks about tuning momentum and weight decay. I am not sure I understood what exactly David wants to convey. I understood the part about weight decay somewhat. But, the approach for tuning momentum is not clear. He says to keep lr/(1-momentum) constant while changing the two, but, then he says it won't bring much effect. Then, how should one proceed for optimising this hyperparameter. 

For weights with a scaling symmetry – which includes all the convolutional layers of our network because of subsequent batch normalisation – gradients are orthogonal to weights. As a result, gradient updates lead to an increase in weight norm whilst weight decay leads to a decrease.

For small weights, the growth term dominates and vice versa for large weights. This leads to a stable control mechanism whereby the weight norms approach a fixed point (for fixed learning rate and other hyperparameters) such that the first order shrinking effect of weight decay balances the second order growth from orthogonal gradient updates.

### Post 6: Weight Decay

This post seems to be out of scope as of now. I can't write any summary as I didn't understand what exactly is happening.

### Post 7: Batch Norm

Why Batch norm?

1. stabilises optimisation allowing much higher learning rates and faster training.
2. improving generalisation by injecting noise
3. reduces sensitivity to weight initialisation
4. interacts with weight decay to control the learning rate dynamics

Point 1 is the major benefit as rest can be achieved using better techniques.  

What's bad about Batch Norm?

1. slow
2. different at training and test time and therefore fragile
3. ineffective for small batches and various layer types
4. multiple interaction effects which are hard to seperate

However, BN is required to train deeper networks and there is nothing which can replace it. Speed issue can be taken care of by using a good compiler which computes the statistics in the previous layer and applies them in the next avoiding unnecessary trips to memory and removing almost all overhead.

- Batch norm acts on *histograms* of *per channel* activations (by shifting means and rescaling variances). ![](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/before_after_0.svg)

   ![](https://296830-909578-raikfcquaxqncofqfm.stackpathdns.com/wp-content/uploads/2019/06/before_after_1-1.svg)

- The diagram above shows histograms of activation values, across pixels and examples in a batch, before and after a batch norm layer. Different channels are represented by different colours; the per channel mean is shown underneath; and minimum/maximum values per channel are indicated by the vertical ticks

### Post-8: Bag of tricks

1. ###### Preprocessing on GPUs

   - Right now,  normalising, transposing and padding the dataset before training is being done on CPU and then images are transferred to GPUs. Instead, move the whole dataset (in uint8 format) to the GPU, which only takes 40 ms. Do preprocessing there which takes 15 ms. Applying augmentations to individual training examples, as on the CPU, we incur substantial overhead launching multiple GPU kernels to process each item. We can avoid this by applying the same augmentation to groups of examples and we can preserve randomness by shuffling the data beforehand. Here, we are relying on the fact that the dataset is small enough to store and manipulate as a whole in GPU memory. With this, training time drops to 70s.

2. ###### Moving Max-pool layers

   - Moving max pool layer before ReLU activation leads to more efficient computation and if it moved before BN, then, it leads to 5s improvement with a slight dent in accuracy which David compensated by training for one more epoch. The net effect bring time to 64s.

3. ###### Label Smoothing

   - It involves blending the one-hot target probabilities with a uniform distribution over class labels inside the cross entropy loss. This helps to stabilise the output distribution and prevents the network from making overconfident predictions which might inhibit further training. 
   - Test accuracy improves to 94.2% (mean of 50 runs.). As a rule of thumb, he drops one epoch for each 0.1% improvement in test accuracy.
   - Accuracy for 23 epochs of training is 94.1% and training time dips under a minute!

4. ###### CELU Activations

   - Continuously Differentiable Exponential Linear Unit
   - Smooth activation function
   - Improvement to 94.3% test accuracy (mean of 50 runs) allowing a further 3 epoch reduction in training and a 20 epoch time of 52s for 94.1% accuracy.Ghost Batch Norm

5. ###### Ghost Batch Norm

   - Batch norm seems to work best with batch size of around 32. The reasons presumably have to do with noise in the batch statistics and specifically a balance between a beneficial regularising effect at intermediate batch sizes and an excess of noise at small batches.
   -  Reducing batch size will give a serious hit on training times
   - David says to apply batch norm separately to subsets of a training batch. But, I think we already do that, isn't it?

6. ###### Exponential Moving Averages

   - High learning rates are necessary for rapid training and they need to be annealed later on to enable optimisation along the steeper and noisier directions in parameter space. Parameter averaging methods allow training to continue at a higher rate whilst potentially approaching minima along noisy or oscillatory directions by averaging over multiple iterates.
   -  Need a new learning rate schedule with higher learning rates towards the end of training, and a momentum for the moving average. For the learning rate, a simple choice is to stick with the piecewise linear schedule that's being used throughout, floored at a low fixed value for the last 2 epochs and choose a momentum of 0.99 so that averaging takes place over a timescale of roughly the last epoch.

7. ###### Test Time Augmentations

   - Make inference on the actual as well as the augmented images and  come to a consensus by averaging network outputs for the augmented images and thus guaranteeing invariance. For example, we would like our network to classify images the same way under horizontal flips of the input.
   - Adding horizontal flip TTA with 13 epoch training setup, test accuracy rises to 94.6%.

8. ###### Training to convergence

   - Training for longer with a simpler model structure,  without all these techniques used to reach 94%  accuracy as fast as possible, lead to higher accuracy after convergence implying there is further scope in improving accuracy faster.  s

### Fenwicks Library

- I tried running the fenwicks library but ran into erros, probably because of changes in version of tensorflow which brings the point of the usefulness of such a library. It only makes sense for my personal use, but I don't think I will use library like fenwicks (I meant developed by somebody else) unless it does some optimisation of the code rather than just representing a higher level representation of other libraries as I would have to keep track of more libraries as the base libraries changes. Whereas if I have created the library, I will know the ins and out of the library and I will know where to make necessary changes if the base libraries changes. So, yes, I understand the usefulness of having such a library so that I don't have to write boilerplate code again and again and instead use such a library. And thanks to you, I have started making such a library.