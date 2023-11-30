Apologize first that as I do not have a coding habit, some parameters need to be changed manually. :)

# Testing
 For testing the trained models in different channel types and SNR situations, pls follow the listed steps:

 1. Go into the main/ folder
 2. check that in model_djscc.py, line 561-564 are NOT commented out while line 565 IS commented out. It should look like this:

    result=model.test()  
    print(result["kp2d_mpjpe"])  
    print(result["kp3d_mpjpe"])  #This is the value used in the paper  
    print(result["kp3d_mpjpe_aligned"])  
    '# model.train()'  
    
 3. Set the channel type and SNR value in the line 244 of generator_djscc.py  

    channel type has two options 'awgn' or 'fading'  
    snr value can be set as 0, -5, -10, -15, -20  
    For -20dB fading channel, it should look like this:  
    self.channel = Channel("fading", -20, name="channel_output")  
    
 4. Set the folder name where the trained model in given channel situation is stored in the line 32 of config.py.  

    For example,  
    the folder named as 'lsp-lspet-mpii-smpl-3D-True1-DJSCC2000-snr-20-1024-fading-hpc' means 2000 symbols are used, snr=-20, fading channel  
    the folder named as 'lsp-lspet-mpii-smpl-3D-True1-DJSCC2000-snr-20-1024-hpc' means 2000 symbols are used, snr=-20, awgn channel  

    For -20dB fading channel, it should look like this:
    LOG_DIR = os.path.join(ROOT_PROJECT_DIR, 'logs', 'lsp-lspet-mpii-smpl-3D-True1-DJSCC2000-snr-20-1024-fading-hpc')  
    
  
    
 5. RUN the model_djscc.py  

    The output should look like this:  
    tf.Tensor(0.23500995, shape=(), dtype=float32)  
    tf.Tensor(0.3161883, shape=(), dtype=float32)  
    tf.Tensor(0.15762103, shape=(), dtype=float32)  
    The second value is the metric used in the paper  
