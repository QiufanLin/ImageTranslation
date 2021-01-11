# -*- coding: utf-8 -*-
# From "Galaxy Image Translation with Semi-supervised Noise-reconstructed Generative Adversarial Networks" (Q.Lin et al. 2021)


import numpy as np
import tensorflow as tf
import argparse
import time
from networks import *


parser = argparse.ArgumentParser()
parser.add_argument('--method', help='Select an image translation method.', type=int)
parser.add_argument('--phase', help='To train or test/reload a trained model.', type=str)
args = parser.parse_args()
method = args.method
phase = args.phase

############################################################
### method == 1: Step 1 of our model
### method == 2: Step 2 of our model (executed after Step 1)
### method == 3: (a) Ad.pix2pix
### method == 4: (b) Ad.CycleGAN
### method == 5: (c) Ad.CycleGAN+PID
### method == 6: (d) Ad.CycleGAN+ID
### method == 7: (e) Ad.CycleGAN+ID+PS
### method == 8: (f) Ours–Auto
### method == 9: (g) CycleGAN (Zhu et al. 2017)
### method == 10: (h) AugCGAN (Almahairi et al. 2018)
############################################################


model_savepath = './models/model_method' + str(method) + '/'
img_savepath = './examples/img_test_method' + str(method)

iterations = 60000
learning_rate_ini = 0.0001
learning_rate_reduce_step = 5  #20000
learning_rate_reduce_factor = 5.0

batch_train = 24
batch_test = 24

if method <= 8: img_size0 = 136   # CFHT with the original pixel scale
else: img_size0 = 64              # Regrided CFHT
img_size1 = 64                    # SDSS
channels = 5                      # Number of passbands
ratio_pixel = 0.396 / 0.187       # The ratio between the pixel scales of SDSS and CFHT
if method == 10: num_latent = 32  # The dimension of the latent code used in Method 10
clip_grad = True                  # Gradient clipping



#################
### Load data ###
#################

####################################################################################
### Please refer to Alam et al. (2015) and Gwyn et al. (2012) for the full datasets.
####################################################################################

fi = np.load('./examples/img_test_examples.npz')       # 32 SDSS & 32 CFHT images
img_test_CFHT_rescaled = fi['img_test_CFHT_rescaled']
img_test_CFHT_regrided = fi['img_test_CFHT_regrided']
img_test_SDSS_rescaled = fi['img_test_SDSS_rescaled']
# print (img_test_CFHT_rescaled.shape, img_test_CFHT_regrided.shape, img_test_SDSS_rescaled.shape)

if method <= 8:
    img_test_CFHT_input = img_test_CFHT_rescaled
else:
    img_test_CFHT_input = img_test_CFHT_regrided        
img_test_SDSS_input = img_test_SDSS_rescaled
    

def img_reshape_single(a):    
    mode = np.random.random()
    if mode < 0.25: a = np.rot90(a, 1, axes=(1, 2))
    elif mode < 0.50: a = np.rot90(a, 2, axes=(1, 2))
    elif mode < 0.75: a = np.rot90(a, 3, axes=(1, 2))
    else: pass       
    mode = np.random.random()
    if mode < 1 / 3.0: a = np.flip(a, 1)
    elif mode < 2 / 3.0: a = np.flip(a, 2)
    else: pass
    return a


def img_reshape_pair(a, b):    
    mode = np.random.random()
    if mode < 0.25:
        a = np.rot90(a, 1, axes=(1, 2))
        b = np.rot90(b, 1, axes=(1, 2))
    elif mode < 0.50:
        a = np.rot90(a, 2, axes=(1, 2))
        b = np.rot90(b, 2, axes=(1, 2))
    elif mode < 0.75: 
        a = np.rot90(a, 3, axes=(1, 2))
        b = np.rot90(b, 3, axes=(1, 2))
    else: pass        
    mode = np.random.random()
    if mode < 1 / 3.0: 
        a = np.flip(a, 1)
        b = np.flip(b, 1)
    elif mode < 2 / 3.0: 
        a = np.flip(a, 2)
        b = np.flip(b, 2)
    else: pass
    return a, b


def get_next_batch(batch):
    ### x0_: CFHT, unpaired. x1_: SDSS, unpaired. x2_: SDSS, paired; x3_: CFHT, paired
    ### Please refer to Alam et al. (2015) and Gwyn et al. (2012) for the full datasets.
    #######
    x0_ = 0
    x1_ = 0
    x2_ = 0
    x3_ = 0
    ####### 
    
    x0_ = img_reshape_single(x0_)
    x1_ = img_reshape_single(x1_)
    x2_, x3_ = img_reshape_pair(x2_, x3_)
    return x0_, x1_, x2_, x3_



#######################
### Define networks ###
#######################

    
### x0: CFHT, unpaired. x1: SDSS, unpaired. x2: SDSS, paired; x3: CFHT, paired
x0 = tf.placeholder(tf.float32, shape=[None, img_size0, img_size0, channels], name="x0")
x1 = tf.placeholder(tf.float32, shape=[None, img_size1, img_size1, channels], name="x1")
x2 = tf.placeholder(tf.float32, shape=[None, img_size1, img_size1, channels], name="x2")
x3 = tf.placeholder(tf.float32, shape=[None, img_size0, img_size0, channels], name="x3")
lr = tf.placeholder(tf.float32, shape=[], name="lr")
batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")


### Noise Emulators
if method <= 2 or method == 8:
    x0_noise = noise_emulator(input=x0, name='noise0')
    x1_noise = noise_emulator(input=x1, name='noise1')
        
### Autoencoders
if method == 1:
     use_deconv = False     
     ### Obtain the non-noise component
     x00_recon_sm = autoencoder(input=x0, batch_size=batch_size, use_deconv=use_deconv, name='auto0')
     x11_recon_sm = autoencoder(input=x1, batch_size=batch_size, use_deconv=use_deconv, name='auto1')
     ### Add noise
     x0_recon = x00_recon_sm + x0_noise
     x1_recon = x11_recon_sm + x1_noise
            
### Generators
if method >= 2 and method <= 8:
    use_deconv = False
    if method == 7: use_PixelShuffle = True
    else: use_PixelShuffle = False
    
    ### The first half-cycle with unpaired data
    ### Obtain the non-noise component
    x01_recon_sm = generator(input=x0, img_size0=img_size0, img_size1=img_size1, batch_size=batch_size, direction='01', use_deconv=use_deconv, use_PixelShuffle=use_PixelShuffle, name='generator01')
    x10_recon_sm = generator(input=x1, img_size0=img_size0, img_size1=img_size1, batch_size=batch_size, direction='10', use_deconv=use_deconv, use_PixelShuffle=use_PixelShuffle, name='generator10')
    ### Add noise
    if method == 2 or method == 8:
         x0_recon = x10_recon_sm + x0_noise
         x1_recon = x01_recon_sm + x1_noise
    else:
         x0_recon = x10_recon_sm
         x1_recon = x01_recon_sm

    ### The second half-cycle with unpaired data
    ### Obtain the non-noise component            
    x010_recon_sm = generator(input=x1_recon, img_size0=img_size0, img_size1=img_size1, batch_size=batch_size, direction='10', use_deconv=use_deconv, use_PixelShuffle=use_PixelShuffle, name='generator10', reuse=True)
    x101_recon_sm = generator(input=x0_recon, img_size0=img_size0, img_size1=img_size1, batch_size=batch_size, direction='01', use_deconv=use_deconv, use_PixelShuffle=use_PixelShuffle, name='generator01', reuse=True)
     
    ### The first half-cycle with paired data
    ### Obtain the non-noise component
    x32_recon_sm = generator(input=x3, img_size0=img_size0, img_size1=img_size1, batch_size=batch_size, direction='01', use_deconv=use_deconv, use_PixelShuffle=use_PixelShuffle, name='generator01', reuse=True)
    x23_recon_sm = generator(input=x2, img_size0=img_size0, img_size1=img_size1, batch_size=batch_size, direction='10', use_deconv=use_deconv, use_PixelShuffle=use_PixelShuffle, name='generator10', reuse=True)

### Discriminators
if method == 1 or (method >= 3 and method <= 8):                            
    D_real0 = discriminator(input=x0, name='discriminator0')
    D_real1 = discriminator(input=x1, name='discriminator1')
    D_gen0 = discriminator(input=x0_recon, name='discriminator0', reuse=True)
    D_gen1 = discriminator(input=x1_recon, name='discriminator1', reuse=True)


### (Aug)CycleGAN Generators
if method >= 9:
    ### Latent code in Method 10
    if method == 9: 
        use_CondIN = False
        latent0 = None
        latent1 = None
        latent2 = None
        latent3 = None
    if method == 10: 
        use_CondIN = True
        latent0 = tf.placeholder(tf.float32, shape=[None, num_latent], name="latent0")
        latent1 = tf.placeholder(tf.float32, shape=[None, num_latent], name="latent1")
        latent2 = tf.placeholder(tf.float32, shape=[None, num_latent], name="latent2")
        latent3 = tf.placeholder(tf.float32, shape=[None, num_latent], name="latent3")
    
    ### The first half-cycle with unpaired data
    ### Obtain the reconstructed images
    x1_recon = cyclegan_generator(input=x0, batch_size=batch_size, use_CondIN=use_CondIN, noise_seed=latent1, name='generator01')
    x0_recon = cyclegan_generator(input=x1, batch_size=batch_size, use_CondIN=use_CondIN, noise_seed=latent0, name='generator10')
    
    ### Obtain the latent code
    if method == 9:
        latent0_recon = None
        latent1_recon = None
        latent0_recon_rev = None
        latent1_recon_rev = None
    if method == 10:
        latent0_recon = encoder_latent(x0, x1_recon, num_latent=num_latent, name='generator_latent0')
        latent1_recon = encoder_latent(x1, x0_recon, num_latent=num_latent, name='generator_latent1')
        latent0_recon_rev = encoder_latent(x0_recon, x1, num_latent=num_latent, name='generator_latent0', reuse=True)
        latent1_recon_rev = encoder_latent(x1_recon, x0, num_latent=num_latent, name='generator_latent1', reuse=True)
    
    ### The second half-cycle with unpaired data
    ### Obtain the reconstructed images
    x010_recon_sm = cyclegan_generator(input=x1_recon, batch_size=batch_size, use_CondIN=use_CondIN, noise_seed=latent0_recon, name='generator10', reuse=True)
    x101_recon_sm = cyclegan_generator(input=x0_recon, batch_size=batch_size, use_CondIN=use_CondIN, noise_seed=latent1_recon, name='generator01', reuse=True)       
    x010_recon = x010_recon_sm
    x101_recon = x101_recon_sm
    
    ### The first half-cycle with paired data
    ### Obtain the reconstructed images & latent code
    if method == 10:
        x32_recon_sm = cyclegan_generator(input=x3, batch_size=batch_size, use_CondIN=use_CondIN, noise_seed=latent2, name='generator01', reuse=True)
        x23_recon_sm = cyclegan_generator(input=x2, batch_size=batch_size, use_CondIN=use_CondIN, noise_seed=latent3, name='generator10', reuse=True)
        x32_recon = x32_recon_sm
        x23_recon = x23_recon_sm
        latent2_recon = encoder_latent(x2, x3, num_latent=num_latent, name='generator_latent1', reuse=True)
        latent3_recon = encoder_latent(x3, x2, num_latent=num_latent, name='generator_latent0', reuse=True)
            
### (Aug)CycleGAN Discriminators
if method >= 9:
    D_real0 = cyclegan_discriminator(input=x0, name='discriminator0')
    D_real1 = cyclegan_discriminator(input=x1, name='discriminator1')
    D_gen0 = cyclegan_discriminator(input=x0_recon, name='discriminator0', reuse=True)
    D_gen1 = cyclegan_discriminator(input=x1_recon, name='discriminator1', reuse=True)
    ### Latent code
    if method == 10:
        ### Unpaired
        D_real_latent0 = cyclegan_discriminator_latent(input=latent0, name='discriminator_latent0')
        D_real_latent1 = cyclegan_discriminator_latent(input=latent1, name='discriminator_latent1')
        D_gen_latent0 = cyclegan_discriminator_latent(input=latent0_recon, name='discriminator_latent0', reuse=True)
        D_gen_latent1 = cyclegan_discriminator_latent(input=latent1_recon, name='discriminator_latent1', reuse=True)
        ### Paired
        D_real_latent2 = cyclegan_discriminator_latent(input=latent2, name='discriminator_latent1', reuse=True)
        D_real_latent3 = cyclegan_discriminator_latent(input=latent3, name='discriminator_latent0', reuse=True)
        D_gen_latent2 = cyclegan_discriminator_latent(input=latent2_recon, name='discriminator_latent1', reuse=True)
        D_gen_latent3 = cyclegan_discriminator_latent(input=latent3_recon, name='discriminator_latent0', reuse=True)



#############################
### Define loss functions ###
#############################
    

### Auto
if method == 1:
    loss_auto0 = tf.reduce_mean(tf.pow(x00_recon_sm - x0, 2))
    loss_auto1 = tf.reduce_mean(tf.pow(x11_recon_sm - x1, 2))
    loss_auto = loss_auto0 + loss_auto1
    
### Identity
if method == 2 or method == 3 or method == 6 or method == 7 or method == 8 or method == 10:
    loss_id2 = tf.reduce_mean(tf.pow(x32_recon_sm - x2, 2))
    loss_id3 = tf.reduce_mean(tf.pow(x23_recon_sm - x3, 2))
    loss_id = loss_id2 + loss_id3

### Pseudo-identity
if method == 5:
    ### Resize CFHT (136 -> 64)
    ### Multiplied by ratio_pixel due to conservation of integraded rescaled fluxs
    x0_resize = ratio_pixel * tf.image.resize_images(x0, size=[img_size1, img_size1], method=tf.image.ResizeMethod.BILINEAR)
    x10_recon_sm_resize = ratio_pixel * tf.image.resize_images(x10_recon_sm, size=[img_size1, img_size1], method=tf.image.ResizeMethod.BILINEAR)    
    loss_pid0 = tf.reduce_mean(tf.pow(x10_recon_sm_resize - x1, 2))
    loss_pid1 = tf.reduce_mean(tf.pow(x01_recon_sm - x0_resize, 2))
    loss_pid = loss_pid0 + loss_pid1
        
### Cycle-consistency
if method == 2 or method >= 4:    
    loss_cyc0 = tf.reduce_mean(tf.pow(x010_recon_sm - x0, 2))
    loss_cyc1 = tf.reduce_mean(tf.pow(x101_recon_sm - x1, 2))
    loss_cyc = loss_cyc0 + loss_cyc1
    ### Latent cycle-consistency
    if method == 10:
        loss_cyc_latent0 = tf.reduce_mean(tf.pow(latent0_recon_rev - latent0, 2))
        loss_cyc_latent1 = tf.reduce_mean(tf.pow(latent1_recon_rev - latent1, 2))
        loss_cyc_latent = loss_cyc_latent0 + loss_cyc_latent1
        
### Adversarial
if method == 1 or method >= 3:
    labels1 = tf.ones_like(D_real0)
    labels0 = tf.zeros_like(D_real0)        
    loss_D_real0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real0, labels = labels1))
    loss_D_real1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real1, labels = labels1))
    loss_D_gen0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gen0, labels = labels0))
    loss_D_gen1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gen1, labels = labels0))
    loss_G_gen0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gen0, labels = labels1))
    loss_G_gen1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gen1, labels = labels1))
    loss_D_adv = loss_D_real0 + loss_D_real1 + loss_D_gen0 + loss_D_gen1
    loss_G_adv = loss_G_gen0 + loss_G_gen1
    ### Latent code
    if method == 10:
        ### Unpaired
        labels_latent1 = tf.ones_like(D_real_latent0)
        labels_latent0 = tf.zeros_like(D_real_latent0)
        loss_D_real_latent0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_latent0, labels = labels_latent1))
        loss_D_real_latent1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_latent1, labels = labels_latent1))
        loss_D_gen_latent0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gen_latent0, labels = labels_latent0))
        loss_D_gen_latent1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gen_latent1, labels = labels_latent0))
        loss_G_gen_latent0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gen_latent0, labels = labels_latent1))
        loss_G_gen_latent1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gen_latent1, labels = labels_latent1))
        loss_D_adv = loss_D_adv + loss_D_real_latent0 + loss_D_real_latent1 + loss_D_gen_latent0 + loss_D_gen_latent1
        loss_G_adv = loss_G_adv + loss_G_gen_latent0 + loss_G_gen_latent1
        ### Paired
        loss_D_real_latent2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_latent2, labels = labels_latent1))
        loss_D_real_latent3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_latent3, labels = labels_latent1))
        loss_D_gen_latent2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gen_latent2, labels = labels_latent0))
        loss_D_gen_latent3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gen_latent3, labels = labels_latent0))
        loss_G_gen_latent2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gen_latent2, labels = labels_latent1))
        loss_G_gen_latent3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gen_latent3, labels = labels_latent1))
        loss_D_adv = loss_D_adv + loss_D_real_latent2 + loss_D_real_latent3 + loss_D_gen_latent2 + loss_D_gen_latent3
        loss_G_adv = loss_G_adv + loss_G_gen_latent2 + loss_G_gen_latent3


### Total loss    
if method == 1:
    loss_G_sm = loss_auto
if method == 2 or method == 8:
    loss_G_sm = loss_cyc + loss_id
if method == 3:
    loss_G_combine = 1000 * loss_id + loss_G_adv
if method == 4 or method == 9:
    loss_G_combine = 1000 * loss_cyc + loss_G_adv
if method == 5:
    loss_G_combine = 1000 * (loss_cyc + loss_pid) + loss_G_adv
if method == 6 or method == 7:
    loss_G_combine = 1000 * (loss_cyc + loss_id) + loss_G_adv
if method == 10:
    loss_G_combine = 1000 * (loss_cyc + loss_id) + loss_cyc_latent + loss_G_adv

#############
### Train ###
#############


def Train():    
    tvars = tf.trainable_variables()
    tvars_D = [var for var in tvars if ('discriminator' in var.name)]
    tvars_G_combine = [var for var in tvars if ('generator' in var.name or 'auto' in var.name or 'noise' in var.name)]
    tvars_G_sm = [var for var in tvars if ('generator' in var.name or 'auto' in var.name)]
    tvars_G_noise = [var for var in tvars if ('noise' in var.name)]
        
    optimizer1 = tf.train.AdamOptimizer(learning_rate=lr)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=lr)
    if clip_grad:
        optimizer1 = tf.contrib.estimator.clip_gradients_by_norm(optimizer1, clip_norm=5.0)
        optimizer2 = tf.contrib.estimator.clip_gradients_by_norm(optimizer2, clip_norm=5.0)          
    if method == 1 or method >= 3:
        optimizer_D = optimizer1.minimize(loss_D_adv, var_list=tvars_D)
    if method == 1 or method == 2 or method == 8:
        optimizer_G_sm = optimizer2.minimize(loss_G_sm, var_list=tvars_G_sm)
    if method == 1 or method == 8: 
        optimizer_G_noise = optimizer2.minimize(loss_G_adv, var_list=tvars_G_noise)
    if (method >= 3 and method <= 7) or method == 9 or method == 10: 
        optimizer_G_combine = optimizer2.minimize(loss_G_combine, var_list=tvars_G_combine)
    
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=session_conf)
    session.run(tf.global_variables_initializer())
    
    ### Load the pre-trained model from Method 1    
    if method == 2:    
        var_list = [var for var in tvars if 'noise' in var.name]
        saver = tf.train.Saver(var_list = var_list)
        saver.restore(session, tf.train.latest_checkpoint('./models/model_method1/'))


    print ('Start training.')
    start = time.time()
    learning_rate = learning_rate_ini       
    for i in range(iterations):
        if i != 0 and i % learning_rate_reduce_step == 0: 
            learning_rate = learning_rate / learning_rate_reduce_factor
            print ('iteration:', i+1, 'learning_rate:', learning_rate, 'time:', str((time.time() - start) / 60) + ' minutes')
        ### Feed images
        x0_, x1_, x2_, x3_ = get_next_batch(batch_train)
        feed_dict = {x0:x0_, x1:x1_, x2:x2_, x3:x3_, lr:learning_rate, batch_size:batch_train}
        ### Feed random noises as latent code
        if method == 10:
            feed_dict[latent0] = np.random.normal(0, 1, (batch_train, num_latent))
            feed_dict[latent1] = np.random.normal(0, 1, (batch_train, num_latent))
            feed_dict[latent2] = np.random.normal(0, 1, (batch_train, num_latent))
            feed_dict[latent3] = np.random.normal(0, 1, (batch_train, num_latent))

        ### One step
        if method == 2:
            session.run(optimizer_G_sm, feed_dict = feed_dict)
        ### Three steps
        elif method == 1 or method == 8:                       
            session.run(optimizer_G_sm, feed_dict = feed_dict)
            session.run(optimizer_D, feed_dict = feed_dict)
            session.run(optimizer_G_noise, feed_dict = feed_dict)
        ### Two steps
        else:
            session.run(optimizer_D, feed_dict = feed_dict)
            session.run(optimizer_G_combine, feed_dict = feed_dict)

    saver = tf.train.Saver()
    saver.save(session, model_savepath, iterations)                    
                    
                

############
### Test ###
############                    


def Test():
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
    session = tf.InteractiveSession(config=session_conf)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, tf.train.latest_checkpoint(model_savepath))

    if method <= 8:
        img_test_CFHT_input = img_test_CFHT_rescaled
    else:
        img_test_CFHT_input = img_test_CFHT_regrided        
    img_test_SDSS_input = img_test_SDSS_rescaled
    
    img_test_CFHT_recon = np.zeros(img_test_CFHT_input.shape)
    img_test_SDSS_recon = np.zeros(img_test_SDSS_input.shape)

    for i in range(1 + int(len(img_test_CFHT_input) / batch_test)):
        i_min = i*batch_test
        i_max = min(len(img_test_CFHT_input), (i+1)*batch_test)
        ### Feed images
        feed_dict = {x0:img_test_CFHT_input[i_min: i_max], x1:img_test_SDSS_input[i_min: i_max], x2:img_test_SDSS_input[i_min: i_max], x3:img_test_CFHT_input[i_min: i_max], batch_size:i_max-i_min}
        ### Feed random noises as latent code
        if method == 10:
            feed_dict[latent0] = np.random.normal(0, 1, (i_max - i_min, num_latent))
            feed_dict[latent1] = np.random.normal(0, 1, (i_max - i_min, num_latent))
            feed_dict[latent2] = np.random.normal(0, 1, (i_max - i_min, num_latent))
            feed_dict[latent3] = np.random.normal(0, 1, (i_max - i_min, num_latent))
        img_test_CFHT_recon[i_min: i_max], img_test_SDSS_recon[i_min: i_max] = session.run([x0_recon, x1_recon], feed_dict = feed_dict)
    np.savez(img_savepath, img_test_CFHT_recon=img_test_CFHT_recon, img_test_SDSS_recon=img_test_SDSS_recon)



#####################################################

if phase == 'train':
    Train()
if phase == 'test':
    Test()
                    