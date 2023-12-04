Apologize first that as I do not have a coding habit, some parameters need to be changed manually. :)

# Testing
 For testing the trained models in different channel types and SNR situations, pls follow the listed steps:

 For AWGN channel:
 1. Go into the main/ folder
 2. check that in model_djscc_adaptive_length.py, line 606-610 are NOT commented out while line 611 IS commented out. It should look like this:

    result=model.test()  
    print(result["kp2d_mpjpe"])  
    print(result["kp3d_mpjpe"])  
    print(result["kp3d_mpjpe_aligned"])  
    print(result["bits"])  
    '# model.train()'
    
 3. Set SNR value in the line 22 and 302 of generator_djscc_adaptive_length.py  
 
    snr value can be set as -5, -10, -15, -20   
    For -20dB channel, it should look like this:  
    line 22: stddev = np.sqrt(10 ** (-(-20) / 10))
    line 302: self.channel = Channel("awgn", -20, name="channel_output")    
    
 4. Set the folder name where the trained model in given channel situation is stored in the line 32 of config.py.  

    For example,  
    the folder named as 'lsp-lspet-mpii-smpl-3D-True1-DJSCC4000-snr-20-1024-adaptive' means maximun 4000 symbols are used, snr=-20, awgn channel with adaptive rate 

    For -20dB channel, it should look like this:
    LOG_DIR = os.path.join(ROOT_PROJECT_DIR, 'logs', 'lsp-lspet-mpii-smpl-3D-True1-DJSCC4000-snr-20-1024-adaptive')  
    
      
 5. RUN the model_djscc_adaptive_length.py   

    The output should look like this:  
    tf.Tensor(0.18493506, shape=(), dtype=float32)  
    tf.Tensor(0.23401994, shape=(), dtype=float32)  # the metric used in the paper
    tf.Tensor(0.13981862, shape=(), dtype=float32)  
    tf.Tensor(1843.0127, shape=(), dtype=float32)  # the average number of used symbols


For Fading channel:
 1. Go into the main/ folder
 2. check that in model_djscc_adaptive_length_fading.py, line 621-625 are NOT commented out while line 626 IS commented out. It should look like this:
    
 3. Set SNR value in the line 48 and 328 of generator_djscc_adaptive_length_fading.py  
 
    snr value can be set as  -5, -10, -15, -20   
    For -20dB channel, it should look like this:  
    line 48: noise_stddev = np.sqrt(10 ** (-(-20) / 10))
    line 328: self.channel = Channel("fading", -20, name="channel_output")    
    
 4. Set the folder name where the trained model in given channel situation is stored in the line 32 of config_fading.py.  

    For example,  
    the folder named as 'lsp-lspet-mpii-smpl-3D-True1-DJSCC4000-snr-20-1024-adaptive-fading' means maximun 4000 symbols are used, snr=-20, fading channel with adaptive rate 

    For -20dB channel, it should look like this:
    LOG_DIR = os.path.join(ROOT_PROJECT_DIR, 'logs', 'lsp-lspet-mpii-smpl-3D-True1-DJSCC4000-snr-20-1024-adaptive-fading') 
    
      
 5. RUN the model_djscc_adaptive_length_fading.py   

    The output should look like this:  
    tf.Tensor(0.21619897, shape=(), dtype=float32)  
    tf.Tensor(0.281966, shape=(), dtype=float32)  # the metric used in the paper
    tf.Tensor(0.15074024, shape=(), dtype=float32)  
    tf.Tensor(1966.3492, shape=(), dtype=float32)  # the average number of used symbols
      
