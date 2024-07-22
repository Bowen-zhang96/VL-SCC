## Testing 
To test different pre-trained models under /tmp, pls change the dictionary path in line 1644/1650 to different paths.
<br>
Suppose c is the value of 'bit_mean' in the printed results. The SPP for data is calculated by c/(16x2).
<br>
Note that the BPP for mask is 4 bit quantization /(16x16)=0.015625, therefore the SPP for mask in 10dB AWGN channel is 0.015625/log2(1+10)=0.004516.
<br>
The total SPP is c/(16x2)+0.004516.

## Training details
The image transmission experiments can be run from scratch. To enable the training mode, pls make the following changes:
1. Change the default value in line 1703 to 'False'.
2. Change the default value in line 1759 to 'None'.
3. Adjust the parameter in line 1237 to adjust the average symbol length.

<b>Getting one point</b>
In my experiment, I first set the parameter in line 1237 to a large value (maybe 2.1) to obtain spp=0.125. 
During this process, I first train for 600 epochs, and then I reduce the learning rate of RAN to 1e-6 and train for another 600 epochs. 
<br>
To reduce the learning rate, change line 1394 to learning_rate=0.01*args.learn_rate. 
<br>
To continue training for a reduced learning rate, simply pass the path of the pre-trained model to line 1753.

After obtaining spp=0.125, change the parameter in line 1237 to adjust the average symbol length. 
<br>
To continue training for changed parameters, simply pass the path of the pre-trained model to line 1753. 

Note that it is important to reduce the learning rate for RAN. 
Note that 600 epochs might not be necessary. Maybe 300 is enough.
