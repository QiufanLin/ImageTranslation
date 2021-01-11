# Galaxy Image Translation with Semi-supervised Noise-reconstructed Generative Adversarial Networks

In this work, we propose a two-way galaxy image translation model for that exploits both paired and unpaired images in a semi-supervised manner, and introduce a noise emulating module that is able to learn and reconstruct noise characterized by high-frequency features. We show that our model recovers global and local properties effectively and outperforms benchmark image translation models. To our best knowledge, this work is the first attempt to apply semi-supervised methods and noise reconstruction techniques in astrophysical studies.

The code is tested using: 
- Python 2.7.15
- TensorFlow 1.12.0
- CPU: Intel(R) Core(TM) i9-7920X
- GPU: Titan V / GeForce RTX 2080 Ti

## Results
Our model can properly recover both galaxy shapes and noise characteristics, whereas there are shortcomings for other methods with no identity constrant through paired images or no noise reconstructing mechanism.

![image](https://github.com/QiufanLin/ImageTranslation/blob/main/Figures/image_examples.png)

![image](https://github.com/QiufanLin/ImageTranslation/blob/main/Figures/variant_analysis.png)

## Data
In our experiments, we use multi-band galaxy images from the Sloan Digital Sky Survey (SDSS; Alam et al. 2015) and the Canada France Hawaii Telescope Legacy Survey (CFHT; Gwyn et al. 2012). Each image contains a galaxy at the center and covers five photometric passbands (*u*, *g*, *r*, *i*, *z*). To ensure the same angular scale, we make SDSS cutout images of 64×64 pixels in size and CFHT cutout images of 136×136 pixels.

The pixel intensity of raw CFHT images is reduced by a factor of 1,000 in order to match the intensity level of SDSS images. Then the pixel intensity of all images is rescaled with the following equation which increases the noise amplitude and thus facilitates noise reconstruction.

I = -\sqrt{-I_0 + 1.0} + 1.0 if I_0 < 0

I = \sqrt{I_0 + 1.0} - 1.0 if I_0 > 0

I and I_0 denote the rescaled intensity and the original intensity, respectively. (Negative fluxes are due to image pre-processing.) The images after this rescaling opertion are saved as *rescaled* images in "./examples/img_test_examples.npz" and taken as inputs to the networks.

We also create a sample of CFHT images of size 64×64 pixels as SDSS images by regridding the rescaled CFHT images with the Bilinear Interpolation. These are saved as *regrided* images in "./examples/img_test_examples.npz" and used in Experiments (g) CycleGAN and (h) AugCGAN.

See below for access to trained models and image examples.

![image](https://github.com/QiufanLin/ImageTranslation/blob/main/Figures/translation.png)

## Model description
We develop a two-step training scheme. Step 1: (i) update the Autoencoders *A_X*, *A_Y* with the original images *x*, *y* from the two domains *X*, *Y*, respectively; (ii) adversarially update the Noise Emulators *NE_X*, *NE_Y* and the Discriminators D_X, D_Y while keeping *A_X*, *A_Y* fixed and taking Gaussian random seeds *z1*, *z2* as inputs to *NE_X*, *NE_Y* to produce noise. Step 2: update the Generators *G_(X→Y)*, *G_(Y→X)*, using noise produced by *NE_X*, *NE_Y*. 

![image](https://github.com/QiufanLin/ImageTranslation/blob/main/Figures/graph.png)

![image](https://github.com/QiufanLin/ImageTranslation/blob/main/Figures/architecture.png)

## Train
> python model.py --method=? --phase=train

Trained models and image examples can be downloaded from
https://drive.google.com/drive/folders/1lxkZFvLgHLsHrTF4lJwXs7shR24TZyjH?usp=sharing

"./examples" only contains a small sample of 32 galaxies. For access to the full SDSS and CFHT datasets, please refer to Alam et al. (2015) and Gwyn et al. (2012).

## Test/reload a trained model
> python model.py --method=? --phase=test

("?" stands for a method. Method 1 and Method 2 have to be executed consecutively. "./examples" contains saved reconstructed images from each method.)

method == 1: Step 1 of our model.

method == 2: Step 2 of our model.

method == 3: (a. Ad.pix2pix or “Adapted pix2pix”) One-way translation using our networks with the identity loss but not the Autoencoders or the Noise Emulators, similar to pix2pix (Isola et al. 2017).

method == 4: (b. Ad.CycleGAN or “Adapted CycleGAN”) Two-way translation using our networks with the cycle-consistency loss but not the Autoencoders or the Noise Emulators, similar to CycleGAN (Zhu et al. 2017).

method == 5: (c. Ad.CycleGAN+PID) Same as Case (b), except adding the pseudo-identity loss (PID).

method == 6: (d. Ad.CycleGAN+ID) Same as Case (b), except adding the identity loss (ID).

method == 7: (e. Ad.CycleGAN+ID+PS) Same as Case (d), except using the Pixel Shuffle units (PS) in the Generators.

method == 8: (f. Ours–Auto) Same as our model, except trained in one step without the Autoencoders.

method == 9: (g. CycleGAN) Two-way translation with only the cycle-consistency loss, using the 6-residual-block CycleGAN architecture as presented in Zhu et al. (2017).

method == 10: (h. AugCGAN) The semi-supervised two-way translation setting of Augmented CycleGAN as presented in Almahairi et al. (2018), based on the CycleGAN architecture with the identity loss, having random seeds injected to the Generators to enable stochastic mappings.
