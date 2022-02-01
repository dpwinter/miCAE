MICAE is a multiple input convolutional autoencoder neural network that consists of multiple convolutional encoders in the input stage, a merged (e.g. concatenated) latent space and a convolutional decoder in the output stage.
The decoder part of the network in this architecture is scaled up with the user-specified number of used input encoders. 

## Findings
1) Elementary encoding units can be trained in solitary and their latent spaces can be concatenated.
2) The concatenated latent space can be used as input to a decoder that can be trained to reproduce the concanation of the elementary encoding units' input images.
3) Training the decoder with the same set of preloaded and fixed weights for all elementary encoding units results in a model with similar performance to training all encoders and the decoder of the model together from scratch. 
4) This model can also be trained from scratch using tied encoder weights, yielding similar performance.
5) Larger latent dimension does not mean better performance. (maxing out at latent_dim~64)
6) Neither CAE nor MICAE seem to overfit.

## Research Objectives
1) Probing latent space: Better understand the compressed representation of the data after training.
2) Latent space concatenation: Understand if and how latent spaces of several models can be combined.

## Boundary Conditions
1) The encoder models must share the same weights. (translational invariance of input & not feasible to train ECONs separately)
2) The encoder architecture must be relatively shallow. (must fit on ECON chip)
3) The compression factor (input/latent) must be 3 or larger. (ratio input/output links (48x8b/16x9b) in the first stage)
4) The encoder architecture of the real problem is as good as fixed. (Input->Conv2D->Flatten->Dense)
