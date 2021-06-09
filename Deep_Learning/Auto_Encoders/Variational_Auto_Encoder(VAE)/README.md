# Variational Auto Encoder(VAE)
[`Variational Auto Encoder`](https://www.youtube.com/watch?v=YV9D3TWY5Zo) models tend to make strong assumptions related to the distribution of latent variables. They use a variational approach for latent representation learning,  which results in an additional loss component and a specific estimator for the training algorithm called the `Stochastic Gradient Variational Bayes` estimator. The probability distribution of the latent vector of a variational autoencoder typically matches the training data much closer than a standard autoencoder. As `VAE` are much more flexible and customisable in their generation behaviour than GANs, they are suitable for art generation of any kind.

## code 
[`python3 sample_keras.py`](./sample_keras.py)  
[`python3 sample_pytorch.py`](./sample_pytorch.py)  
[`python3 sample_scratch.py`](./sample_scratch.py)  

<p align="center">
  <img src="https://miro.medium.com/max/576/1*79AzftDm7WcQ9OfRH5Y-6g@2x.png">
</p>
<p align="center">
  <img src="https://www.tensorflow.org/tutorials/generative/images/cvae_latent_space.jpg">
</p>

## Usefull Resources:
+ https://github.com/FaustineLi/Variational-Autoencoders  
+ https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf  
+ https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73  
+ https://towardsdatascience.com/variational-autoencoders-as-generative-models-with-keras-e0c79415a7eb  (use incognito)  
+ https://keras.io/examples/generative/vae/  
+ https://www.tensorflow.org/tutorials/generative/cvae  
+ https://ml.informatik.uni-freiburg.de/papers/15-NIPS-auto-sklearn-preprint.pdf
+ https://hanxiao.io/2018/06/24/4-Encoding-Blocks-You-Need-to-Know-Besides-LSTM-RNN-in-Tensorflow/
+ https://www.mygreatlearning.com/blog/autoencoder/  
+ https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial  
+ https://pythonmachinelearning.pro/all-about-autoencoders/  