# -*- coding: utf-8 -*-
# From "Galaxy Image Translation with Semi-supervised Noise-reconstructed Generative Adversarial Networks" (Q.Lin et al. 2021)


import tensorflow as tf


def conv2d(input, name, num_output_channels=None, kernel_size=3, strides=[1,1,1,1], padding='SAME', use_biases=True, act='leakyrelu', use_input_k_b=False, input_kernel=None, input_biases=None, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if padding == 'SYMMETRIC':
            padding_size = int(kernel_size / 2)
            input = tf.pad(input, paddings=tf.constant([[0,0], [padding_size,padding_size], [padding_size,padding_size], [0,0]]), mode='SYMMETRIC')
            padding = 'VALID'
        if padding == 'REFLECT':
            padding_size = int(kernel_size / 2)
            input = tf.pad(input, paddings=tf.constant([[0,0], [padding_size,padding_size], [padding_size,padding_size], [0,0]]), mode='REFLECT')
            padding = 'VALID'
            
        if use_input_k_b:
            kernel = input_kernel
            biases = input_biases
        else:
            num_in_channels = input.get_shape()[-1].value
            kernel_shape = [kernel_size, kernel_size, num_in_channels, num_output_channels] 
            biases = tf.get_variable('biases', shape=[num_output_channels], initializer=tf.constant_initializer(0.1))
            kernel =  tf.get_variable('weights', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
                        
        output = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)
        if use_biases: output = tf.nn.bias_add(output, biases)
        
        if act == 'leakyrelu': output = tf.nn.leaky_relu(output)
        elif act == 'relu': output = tf.nn.relu(output)
        elif act == 'tanh': output = tf.nn.tanh(output)
        elif act == 'sigmoid': output = tf.sigmoid(output)
        elif act == None: pass        
        return output


    
def deconv2d(input, num_output_channels, kernel_size, name, output_size, batch_size, strides=[1,2,2,1], padding='SAME', act='leakyrelu', reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        num_input_channels = input.get_shape()[-1].value
        kernel_shape = [kernel_size, kernel_size, num_output_channels, num_input_channels]
        output_shape = [batch_size, output_size[0], output_size[1], num_output_channels]
            
        biases = tf.get_variable('biases', shape=[num_output_channels], initializer=tf.constant_initializer(0.1))
        kernel = tf.get_variable('weights', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
                        
        output = tf.nn.conv2d_transpose(input, kernel, output_shape=output_shape, strides=strides, padding=padding)
        output = tf.nn.bias_add(output, biases)
        
        if act == 'leakyrelu': output = tf.nn.leaky_relu(output)
        elif act == 'relu': output = tf.nn.relu(output)
        elif act == 'tanh': output = tf.nn.tanh(output)
        elif act == 'sigmoid': output = tf.sigmoid(output)
        elif act == None: pass        
        return output



def pool2d(input, kernel_size, stride, name, padding='SAME', use_avg=True, reuse=False):
    with tf.variable_scope(name):
        if reuse: tf.get_variable_scope().reuse_variables()
        
        if use_avg: 
            return tf.nn.avg_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)
        else: 
            return tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)



def fully_connected(input, num_output, name, act='leakyrelu', reuse=False):           
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        num_input_units = input.get_shape()[-1].value
        kernel_shape = [num_input_units, num_output]
        kernel = tf.get_variable('weights', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[num_output], initializer=tf.constant_initializer(0.1))
        
        output = tf.matmul(input, kernel)
        output = tf.nn.bias_add(output, biases)

        if act == 'leakyrelu': output = tf.nn.leaky_relu(output)
        elif act == 'relu': output = tf.nn.relu(output)
        elif act == 'tanh': output = tf.nn.tanh(output)
        elif act == 'sigmoid': output = tf.sigmoid(output)
        elif act == None: pass         
        return output



def autoencoder(input, batch_size, name, use_deconv=False, reuse=False):  
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        conv1 = conv2d(input=input, num_output_channels=64, kernel_size=3, name='conv1', act='leakyrelu')
        p1 = pool2d(input=conv1, kernel_size=2, stride=2, name='p1', use_avg=True)
        conv2 = conv2d(input=p1, num_output_channels=128, kernel_size=3, name='conv2', act='leakyrelu')
        p2 = pool2d(input=conv2, kernel_size=2, stride=2, name='p2', use_avg=True)
        conv3 = conv2d(input=p2, num_output_channels=256, kernel_size=3, name='conv3', act='leakyrelu')
        p3 = pool2d(input=conv3, kernel_size=2, stride=2, name='p3', use_avg=True)
        conv4 = conv2d(input=p3, num_output_channels=256, kernel_size=3, name='conv4', act='leakyrelu')

        if use_deconv:
            up1 = deconv2d(input=conv4, num_output_channels=256, kernel_size=5, output_size=[conv4.get_shape()[1].value*2, conv4.get_shape()[2].value*2], batch_size=batch_size, name='up1', act='leakyrelu')
        else:
            up1 = tf.image.resize_images(conv4, size=[conv4.get_shape()[1].value*2, conv4.get_shape()[2].value*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #    print (up1)        
        conv5 = conv2d(input=up1, num_output_channels=256, kernel_size=3, name='conv5', act='leakyrelu')

        if use_deconv:
            up2 = deconv2d(input=conv5, num_output_channels=128, kernel_size=9, output_size=[conv5.get_shape()[1].value*2, conv5.get_shape()[2].value*2], batch_size=batch_size, name='up2', act='leakyrelu')
        else:
            up2 = tf.image.resize_images(conv5, size=[conv5.get_shape()[1].value*2, conv5.get_shape()[2].value*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #    print (up2)                    
        conv6 = conv2d(input=up2, num_output_channels=128, kernel_size=3, name='conv6', act='leakyrelu')

        if use_deconv:
            up3 = deconv2d(input=conv6, num_output_channels=64, kernel_size=13, output_size=[conv6.get_shape()[1].value*2, conv6.get_shape()[2].value*2], batch_size=batch_size, name='up3', act='leakyrelu')
        else:
            up3 = tf.image.resize_images(conv6, size=[conv6.get_shape()[1].value*2, conv6.get_shape()[2].value*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #    print (up3)                    
        conv7 = conv2d(input=up3, num_output_channels=64, kernel_size=5, name='conv7', act='leakyrelu')
        conv8 = conv2d(input=conv7, num_output_channels=64, kernel_size=5, name='conv8', act='leakyrelu')        
        conv9 = conv2d(input=conv8, num_output_channels=input.get_shape()[-1].value, kernel_size=5, name='conv9', act='leakyrelu')            
        return conv9



def noise_emulator(input, name, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        num_input_channels = input.get_shape()[-1].value
        amplitude_seed = tf.random_normal(tf.shape(tf.reduce_mean(input, (1,2))), 0, 1)
        amplitude_seed_split = tf.split(amplitude_seed, num_input_channels, -1)

        amplitude = []
        for i in range(num_input_channels):        
            fc1 = fully_connected(input=amplitude_seed_split[i], num_output=64, name='fc1'+str(i), act='leakyrelu')
            fc2 = fully_connected(input=fc1, num_output=64, name='fc2'+str(i), act='leakyrelu')
            fc3 = fully_connected(input=fc2, num_output=1, name='fc3'+str(i), act=None)
            amplitude.append(tf.nn.softplus(fc3))
        amplitude = tf.concat([x for x in amplitude], -1)

        noise_2dmap = tf.random_normal(tf.shape(tf.pad(input, paddings=tf.constant([[0,0], [3,3], [3,3], [0,0]]), mode='SYMMETRIC')), 0, 1)
        noise = noise_2dmap * tf.expand_dims(tf.expand_dims(amplitude, 1), 1)
        noise_split = tf.split(noise, num_input_channels, -1)
        
        kernel = tf.get_variable('kernel', shape=[1, 7, 7, num_input_channels], initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        kernel = (kernel + tf.image.rot90(kernel, k=1) + tf.image.rot90(kernel, k=2) + tf.image.rot90(kernel, k=3)) / 4.0
        kernel = (kernel + tf.image.flip_left_right(kernel)) / 2.0
        kernel = tf.transpose(kernel, [1,2,0,3])
        kernel_split = tf.split(kernel, num_input_channels, -1)

        output_noise = []
        for i in range(num_input_channels):
            output_noise.append(conv2d(input=noise_split[i], use_input_k_b=True, num_output_channels=1, kernel_size=7, padding='VALID', input_kernel=kernel_split[i], use_biases=False, name='output_noise'+str(i), act=None))
        output_noise = tf.concat([x for x in output_noise], -1)
        return output_noise



def discriminator(input, name, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        num_input_channels = input.get_shape()[-1].value
        filt = tf.expand_dims(tf.expand_dims(tf.constant([[-1.0, 3.0, -4.0, 3.0, -1.0], [3.0, -8.0, 10.0, -8.0, 3.0], [-4.0, 10.0, -12.0, 10.0, -4.0], [3.0, -8.0, 10.0, -8.0, 3.0], [-1.0, 3.0, -4.0, 3.0, -1.0]], dtype=tf.float32), -1), -1)
        input = [tf.nn.conv2d(tf.expand_dims(input[:,:,:,i], -1), filt, strides=[1,1,1,1], padding='VALID') for i in range(num_input_channels)]
        input = tf.concat([x for x in input], -1)   # [batch, img_size-4, img_size-4, 5]

        input2 = tf.transpose(tf.cast(input, dtype=tf.complex64), [0, 3, 1, 2])
        input2 = tf.transpose(tf.fft2d(input2), [0, 2, 3, 1])
        input2 = tf.concat([tf.expand_dims(tf.real(input2), -1), tf.expand_dims(tf.imag(input2), -1)], -1)
    #    print(input2)   # [batch, img_size-4, img_size-4, 5, 2]

        input_split = tf.split(input, num_input_channels, -1)
        input2_split = tf.split(input2, num_input_channels, -2)
        output = []

        for i in range(num_input_channels):
            ### Fourier space
            conv1 = conv2d(input=tf.squeeze(input2_split[i], -2), num_output_channels=64, kernel_size=7, name='fft_conv1'+str(i), act='leakyrelu')
            p1 = pool2d(input=conv1, kernel_size=2, stride=2, name='fft_p1'+str(i), use_avg=True)

            conv2 = conv2d(input=p1, num_output_channels=64, kernel_size=7, name='fft_conv2'+str(i), act='leakyrelu')
            a_conv2 = fully_connected(input=tf.reduce_mean(conv2, (1,2)), num_output=64, name='attention_fft_conv2'+str(i), act='sigmoid')
            conv2 = conv2 * tf.expand_dims(tf.expand_dims(a_conv2, 1), 1)
            p2 = pool2d(input=conv2, kernel_size=2, stride=2, name='fft_p2'+str(i), use_avg=True)

            conv3 = conv2d(input=p2, num_output_channels=64, kernel_size=7, name='fft_conv3'+str(i), act='leakyrelu')
            a_conv3 = fully_connected(input=tf.reduce_mean(conv3, (1,2)), num_output=64, name='attention_fft_conv3'+str(i), act='sigmoid')
            conv3 = conv3 * tf.expand_dims(tf.expand_dims(a_conv3, 1), 1)            
            p3 = pool2d(input=conv3, kernel_size=2, stride=2, name='fft_p3'+str(i), use_avg=True)

            conv4 = conv2d(input=p3, num_output_channels=128, kernel_size=3, name='fft_conv4'+str(i), act='leakyrelu')
            a_conv4 = fully_connected(input=tf.reduce_mean(conv4, (1,2)), num_output=128, name='attention_fft_conv4'+str(i), act='sigmoid')
            conv4 = conv4 * tf.expand_dims(tf.expand_dims(a_conv4, 1), 1)
            p4 = pool2d(input=conv4, kernel_size=2, stride=2, name='fft_p4'+str(i), use_avg=True)

            conv5 = conv2d(input=p4, num_output_channels=128, kernel_size=3, name='fft_conv5'+str(i), act='leakyrelu')    
            a_conv5 = fully_connected(input=tf.reduce_mean(conv5, (1,2)), num_output=128, name='attention_fft_conv5'+str(i), act='sigmoid')
            conv5 = conv5 * tf.expand_dims(tf.expand_dims(a_conv5, 1), 1)       
            concat = tf.concat([tf.reduce_mean(conv5, (1,2)), tf.reduce_max(conv5, (1,2)), tf.reduce_min(conv5, (1,2))], -1)

            ### Real space                       
            conv1 = conv2d(input=input_split[i], num_output_channels=64, kernel_size=3, name='conv1'+str(i), act='leakyrelu')
            p1 = pool2d(input=conv1, kernel_size=2, stride=2, name='p1'+str(i), use_avg=True)

            conv2 = conv2d(input=p1, num_output_channels=64, kernel_size=3, name='conv2'+str(i), act='leakyrelu')
            a_conv2 = fully_connected(input=tf.reduce_mean(conv2, (1,2)), num_output=64, name='attention_conv2'+str(i), act='sigmoid')
            conv2 = conv2 * tf.expand_dims(tf.expand_dims(a_conv2, 1), 1)
            p2 = pool2d(input=conv2, kernel_size=2, stride=2, name='p2'+str(i), use_avg=True)

            conv3 = conv2d(input=p2, num_output_channels=64, kernel_size=3, name='conv3'+str(i), act='leakyrelu')
            a_conv3 = fully_connected(input=tf.reduce_mean(conv3, (1,2)), num_output=64, name='attention_conv3'+str(i), act='sigmoid')
            conv3 = conv3 * tf.expand_dims(tf.expand_dims(a_conv3, 1), 1)            
            concat = tf.concat([concat, tf.reduce_mean(conv3, (1,2)), tf.reduce_max(conv3, (1,2)), tf.reduce_min(conv3, (1,2))], -1)
      
            fc1 = fully_connected(input=concat, num_output=32, name='fc1'+str(i), act='leakyrelu')
            fc2 = fully_connected(input=fc1, num_output=1, name='fc2'+str(i), act=None)
            output.append(fc2)                            
        return tf.concat([x for x in output], -1)



def generator(input, img_size0, img_size1, batch_size, name, direction='01', use_deconv=False, use_PixelShuffle=False, reuse=False):   
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
                
        def PixelShuffle(input):
            _, d1, d2, c = input.get_shape().as_list()
            input_split = tf.split(input, int(c/4), -1)  # c/4, [batch, d1, d2, 4]
            output = []
            for x in input_split:
                x = tf.reshape(x, [-1, d1, d2, 2, 2])
                x = tf.split(x, d1, 1)  # d1, [batch, 1, d2, 2, 2]
                x = tf.concat([tf.squeeze(y, 1) for y in x], 2)  # batch, d2, d1*2, 2
                x = tf.split(x, d2, 1)  # d2, [batch, 1, d1*2, 2]
                x = tf.concat([tf.squeeze(y, 1) for y in x], 2)  # batch, d1*2, d2*2
                output.append(tf.reshape(x, [-1, d1*2, d2*2, 1]))     # c/4, [batch, d1*2, d2*2, 1]
            return tf.concat([x for x in output], -1)        

        num_input_channels = input.get_shape()[-1].value
        if direction == '10':  # img_size0 == 136, img_size1 == 64
            conv1 = conv2d(input=input, num_output_channels=64, kernel_size=3, name='conv1', act='leakyrelu')
            p1 = pool2d(input=conv1, kernel_size=2, stride=2, name='p1', use_avg=True)
            conv2 = conv2d(input=p1, num_output_channels=128, kernel_size=3, name='conv2', act='leakyrelu')
            p2 = pool2d(input=conv2, kernel_size=2, stride=2, name='p2', use_avg=True)
            conv3 = conv2d(input=p2, num_output_channels=128, kernel_size=3, name='conv3', act='leakyrelu')
            p3 = pool2d(input=conv3, kernel_size=2, stride=2, name='p3', use_avg=True)
            conv4 = conv2d(input=p3, num_output_channels=256, kernel_size=3, name='conv4', act='leakyrelu')

            ### 8 -> 16
            if use_PixelShuffle:
                up1 = PixelShuffle(conv4)
            elif use_deconv:
                up1 = deconv2d(input=conv4, num_output_channels=128, kernel_size=5, output_size=[conv4.get_shape()[1].value*2, conv4.get_shape()[2].value*2], batch_size=batch_size, name='up1', act='leakyrelu')
            else:
                up1 = tf.image.resize_images(conv4, size=[conv4.get_shape()[1].value*2, conv4.get_shape()[2].value*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
     #       print (up1)
            conv5 = conv2d(input=up1, num_output_channels=128, kernel_size=3, name='conv5', act='leakyrelu')    
            
            ### 16 -> 32
            if use_PixelShuffle:
                up2 = PixelShuffle(conv5)
            elif use_deconv:
                up2 = deconv2d(input=conv5, num_output_channels=128, kernel_size=7, output_size=[conv5.get_shape()[1].value*2, conv5.get_shape()[2].value*2], batch_size=batch_size, name='up2', act='leakyrelu')
            else:
                up2 = tf.image.resize_images(conv5, size=[conv5.get_shape()[1].value*2, conv5.get_shape()[2].value*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
     #       print (up2)
            conv6 = conv2d(input=up2, num_output_channels=128, kernel_size=3, name='conv6', act='leakyrelu')    
            
            ### 32 -> 64
            if use_PixelShuffle:
                up3 = PixelShuffle(conv6)
            elif use_deconv:
                up3 = deconv2d(input=conv6, num_output_channels=128, kernel_size=9, output_size=[conv6.get_shape()[1].value*2, conv6.get_shape()[2].value*2], batch_size=batch_size, name='up3', act='leakyrelu')  
            else:
                up3 = tf.image.resize_images(conv6, size=[conv6.get_shape()[1].value*2, conv6.get_shape()[2].value*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #        print (up3)
            conv7 = conv2d(input=up3, num_output_channels=128, kernel_size=3, name='conv7', act='leakyrelu')
            
            ### 64 -> 68 -> 136
            up4 = tf.pad(conv7, paddings=tf.constant([[0,0], [2,2], [2,2], [0,0]]), mode='SYMMETRIC')
      #      print (up4)            
            if use_PixelShuffle:
                up5 = PixelShuffle(up4)
            elif use_deconv:
                up5 = deconv2d(input=up4, num_output_channels=64, kernel_size=13, output_size=[img_size0, img_size0], batch_size=batch_size, name='up5', act='leakyrelu')             
            else:
                up5 = tf.image.resize_images(up4, size=[img_size0, img_size0], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)            
       #     print (up5)
            
            conv8 = conv2d(input=up5, num_output_channels=64, kernel_size=5, name='conv8', act='leakyrelu')
            conv9 = conv2d(input=conv8, num_output_channels=64, kernel_size=5, name='conv9', act='leakyrelu')
            conv10 = conv2d(input=conv9, num_output_channels=num_input_channels, kernel_size=5, name='conv10', act='leakyrelu')
            output = conv10
            
        elif direction == '01':  # img_size0 == 136, img_size1 == 64
            conv1 = conv2d(input=input, num_output_channels=64, kernel_size=3, name='conv1', act='leakyrelu')
            p1 = pool2d(input=conv1, kernel_size=2, stride=2, name='p1', use_avg=True)
            conv2 = conv2d(input=p1, num_output_channels=64, kernel_size=3, name='conv2', act='leakyrelu')
            p2 = pool2d(input=conv2, kernel_size=2, stride=2, name='p2', use_avg=True)
            conv3 = conv2d(input=p2, num_output_channels=128, kernel_size=3, name='conv3', act='leakyrelu')
            p3 = pool2d(input=conv3, kernel_size=2, stride=2, name='p3', use_avg=True)
            conv4 = conv2d(input=p3, num_output_channels=256, kernel_size=3, name='conv4', act='leakyrelu')    
            
            ### 17 -> 34
            if use_PixelShuffle:
                up1 = PixelShuffle(conv4)
            elif use_deconv:
                up1 = deconv2d(input=conv4, num_output_channels=128, kernel_size=5, output_size=[conv4.get_shape()[1].value*2, conv4.get_shape()[2].value*2], batch_size=batch_size, name='up1', act='leakyrelu')
            else:
                up1 = tf.image.resize_images(conv4, size=[conv4.get_shape()[1].value*2, conv4.get_shape()[2].value*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
   #         print (up1)            
            conv5 = conv2d(input=up1, num_output_channels=128, kernel_size=3, name='conv5', act='leakyrelu')
            
            ### 34 -> 68 -> 66 -> 64
            if use_PixelShuffle:
                up2 = PixelShuffle(conv5)
            elif use_deconv:
                up2 = deconv2d(input=conv5, num_output_channels=128, kernel_size=9, output_size=[conv5.get_shape()[1].value*2, conv5.get_shape()[2].value*2], batch_size=batch_size, name='up2', act='leakyrelu')
            else:
                up2 = tf.image.resize_images(conv5, size=[conv5.get_shape()[1].value*2, conv5.get_shape()[2].value*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #        print (up2)            
            conv6 = conv2d(input=up2, num_output_channels=64, kernel_size=5, name='conv6', act='leakyrelu', padding='VALID')
            conv7 = conv2d(input=conv6, num_output_channels=64, kernel_size=5, name='conv7', act='leakyrelu')
            conv8 = conv2d(input=conv7, num_output_channels=num_input_channels, kernel_size=5, name='conv8', act='leakyrelu')
            output = conv8                
        return output



def encoder_latent(input1, input2, num_latent, name, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        epsilon = 1e-20
        conv1 = conv2d(input=tf.concat([input1, input2], -1), num_output_channels=64, kernel_size=3, strides=[1,2,2,1], name='conv1', act='leakyrelu')
        m_conv1, v_conv1 = tf.nn.moments(conv1, [1,2], keep_dims=True)
        conv1 = (conv1 - m_conv1) / tf.sqrt(v_conv1 + epsilon)

        conv2 = conv2d(input=conv1, num_output_channels=128, kernel_size=3, strides=[1,2,2,1], name='conv2', act='leakyrelu')
        m_conv2, v_conv2 = tf.nn.moments(conv2, [1,2], keep_dims=True)
        conv2 = (conv2 - m_conv2) / tf.sqrt(v_conv2 + epsilon)

        conv3 = conv2d(input=conv2, num_output_channels=256, kernel_size=3, strides=[1,2,2,1], name='conv3', act='leakyrelu')
        m_conv3, v_conv3 = tf.nn.moments(conv3, [1,2], keep_dims=True)
        conv3 = (conv3 - m_conv3) / tf.sqrt(v_conv3 + epsilon)

        conv4 = conv2d(input=conv3, num_output_channels=512, kernel_size=3, strides=[1,2,2,1], name='conv4', act='leakyrelu')
        m_conv4, v_conv4 = tf.nn.moments(conv4, [1,2], keep_dims=True)
        conv4 = (conv4 - m_conv4) / tf.sqrt(v_conv4 + epsilon)

        conv5 = conv2d(input=conv4, num_output_channels=512, kernel_size=3, strides=[1,2,2,1], name='conv5', act='leakyrelu')
        m_conv5, v_conv5 = tf.nn.moments(conv5, [1,2], keep_dims=True)
        conv5 = (conv5 - m_conv5) / tf.sqrt(v_conv5 + epsilon)

        conv6 = conv2d(input=conv5, num_output_channels=num_latent, kernel_size=1, name='conv6', act=None)
        return tf.reduce_mean(conv6, (1, 2))
    
    

def cyclegan_generator(input, name, batch_size, use_CondIN=False, noise_seed=None, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
 
        epsilon = 1e-20
        def get_CondIN(conv_input, noise_input, name):
            mean = fully_connected(input=noise_input, num_output=conv_input.get_shape()[-1].value, name=name+'_mean', act=None)
            std = fully_connected(input=noise_input, num_output=conv_input.get_shape()[-1].value, name=name+'_std', act=None)
            mean = tf.expand_dims(tf.expand_dims(mean, 1), 1)
            std = tf.expand_dims(tf.expand_dims(tf.nn.softplus(std), 1), 1)
            m_conv, v_conv = tf.nn.moments(conv_input, [1,2], keep_dims=True)
            return mean + (conv_input - m_conv) / tf.sqrt(v_conv + epsilon) * std
                    
        conv0 = conv2d(input=input, num_output_channels=64, kernel_size=7, name='conv0', padding='REFLECT', act='leakyrelu')
        if use_CondIN:
            conv0 = get_CondIN(conv0, noise_seed, name='conv0')
        else:
            m_conv0, v_conv0 = tf.nn.moments(conv0, [1,2], keep_dims=True)
            conv0 = (conv0 - m_conv0) / tf.sqrt(v_conv0 + epsilon)
  
        conv1 = conv2d(input=conv0, num_output_channels=128, kernel_size=3, strides=[1,2,2,1], name='conv1', padding='REFLECT', act='leakyrelu')
        if use_CondIN:
            conv1 = get_CondIN(conv1, noise_seed, name='conv1')
        else:
            m_conv1, v_conv1 = tf.nn.moments(conv1, [1,2], keep_dims=True)
            conv1 = (conv1 - m_conv1) / tf.sqrt(v_conv1 + epsilon)

        conv2 = conv2d(input=conv1, num_output_channels=256, kernel_size=3, strides=[1,2,2,1], name='conv2', padding='REFLECT', act='leakyrelu')
        if use_CondIN:
            conv2 = get_CondIN(conv2, noise_seed, name='conv2')
        else:
            m_conv2, v_conv2 = tf.nn.moments(conv2, [1,2], keep_dims=True)
            conv2 = (conv2 - m_conv2) / tf.sqrt(v_conv2 + epsilon)
        
        conv_res = conv2
        for i in range(6):
            conv_res_i1 = conv2d(input=conv_res, num_output_channels=256, kernel_size=3, name='conv_res_i1_'+str(i+1), padding='REFLECT', act='leakyrelu')
            if use_CondIN:
                conv_res_i1 = get_CondIN(conv_res_i1, noise_seed, name='conv_res_i1_'+str(i+1))
            
            conv_res_i2 = conv2d(input=conv_res_i1, num_output_channels=256, kernel_size=3, name='conv_res_i2_'+str(i+1), padding='REFLECT', act=None)
            if use_CondIN:
                conv_res_i2 = get_CondIN(conv_res_i2, noise_seed, name='conv_res_i2_'+str(i+1))
            
            conv_res = tf.nn.leaky_relu(conv_res + conv_res_i2)
        
        up1 = deconv2d(input=conv_res, num_output_channels=128, kernel_size=3, output_size=[conv_res.get_shape()[1].value*2, conv_res.get_shape()[2].value*2], batch_size=batch_size, name='up1', act='leakyrelu')
        if use_CondIN:
            up1 = get_CondIN(up1, noise_seed, name='up1')
        else:
            m_up1, v_up1 = tf.nn.moments(up1, [1,2], keep_dims=True)
            up1 = (up1 - m_up1) / tf.sqrt(v_up1 + epsilon)

        up2 = deconv2d(input=up1, num_output_channels=64, kernel_size=3, output_size=[up1.get_shape()[1].value*2, up1.get_shape()[2].value*2], batch_size=batch_size, name='up2', act='leakyrelu')
        if use_CondIN:
            up2 = get_CondIN(up2, noise_seed, name='up2')
        else:
            m_up2, v_up2 = tf.nn.moments(up2, [1,2], keep_dims=True)
            up2 = (up2 - m_up2) / tf.sqrt(v_up2 + epsilon)

        conv3 = conv2d(input=up2, num_output_channels=input.get_shape()[-1].value, kernel_size=7, name='conv3', padding='REFLECT', act='leakyrelu')
        return conv3



def cyclegan_discriminator(input, name, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
                
        epsilon = 1e-20
        conv0 = conv2d(input=input, num_output_channels=64, kernel_size=4, strides=[1,2,2,1], name='conv0', act='leakyrelu')

        conv1 = conv2d(input=conv0, num_output_channels=128, kernel_size=4, strides=[1,2,2,1], name='conv1', act='leakyrelu')
        m_conv1, v_conv1 = tf.nn.moments(conv1, [1,2], keep_dims=True)
        conv1 = (conv1 - m_conv1) / tf.sqrt(v_conv1 + epsilon)

        conv2 = conv2d(input=conv1, num_output_channels=256, kernel_size=4, strides=[1,2,2,1], name='conv2', act='leakyrelu')
        m_conv2, v_conv2 = tf.nn.moments(conv2, [1,2], keep_dims=True)
        conv2 = (conv2 - m_conv2) / tf.sqrt(v_conv2 + epsilon)

        conv3 = conv2d(input=conv2, num_output_channels=512, kernel_size=4, name='conv3', act='leakyrelu')
        m_conv3, v_conv3 = tf.nn.moments(conv3, [1,2], keep_dims=True)
        conv3 = (conv3 - m_conv3) / tf.sqrt(v_conv3 + epsilon)
        
        conv4 = conv2d(input=conv3, num_output_channels=1, kernel_size=4, name='conv4', act=None)           
        return conv4



def cyclegan_discriminator_latent(input, name, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        fc1 = fully_connected(input=input, num_output=64, name='fc1', act='leakyrelu')
        fc2 = fully_connected(input=fc1, num_output=64, name='fc2', act='leakyrelu')
        fc3 = fully_connected(input=fc2, num_output=64, name='fc3', act='leakyrelu')
        fc4 = fully_connected(input=fc3, num_output=1, name='fc4', act='leakyrelu')
        return fc4