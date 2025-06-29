import tensorflow as tf
from Network import Generator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow.keras.backend as K
from Custom_loss import custom_loss
from numpy.random import randint

image_shape_hr = (104, 240, 6)
image_shape_lr=(26, 60, 6)
downscale_factor = 4

# select a batch of random samples, returns images and target 
def generate_batch_samples(data_gcm, data_obs, n_samples):
	# choose random instances
	ix = randint(0, data_gcm.shape[0], n_samples)
	# retrieve selected images
	gcm = data_gcm[ix]
	obs = data_obs[ix]   
	return gcm, obs

def save_models(step, model):
	# save the first generator model
	filename = 'generator_%03d.h5' % (step)
	model.save('/save_model/%s' %(filename))

generator=Generator(image_shape_lr).generator()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(data_gcm, data_obs,mean_pr, std_pr):
	# persistent is set to True because the tape is used more than
	# once to calculate the gradients.
	with tf.GradientTape(persistent=True) as tape:
		hr_fake=generator(data_gcm, training=True)
		loss= custom_loss(mean_pr, std_pr)
		loss_value=loss(data_obs, hr_fake)
    # Calculate the gradients for generator
	generator_gradients = tape.gradient(loss_value, 
                                        generator.trainable_variables)
	# Apply the gradients to the optimizer
	generator_optimizer.apply_gradients(zip(generator_gradients, 
                                            generator.trainable_variables))
	return loss_value

def train(train_gcm, train_obs, epochs, batch_size):
	# define properties of the training run
	n_epochs, n_batch, = epochs, batch_size
	bat_per_epo = int(len(train_gcm) / n_batch)
	n_steps = bat_per_epo * n_epochs
	for i in range(n_steps):
		batch_gcm, batch_obs=generate_batch_samples(train_gcm, train_obs,batch_size)
		loss_value=train_step(batch_gcm, batch_obs,mean_pr,std_pr)
        
		#print('Iteration>%d, loss=%.3f' % (i+1, loss_value))
		loss_file = open('losses.txt' , 'a')
		loss_file.write('Iteration>%d, loss=%.3f\n' % (i+1, loss_value))
		loss_file.close()
		if (i+1) % 741 == 0:
			# save the models
			save_models((i+1) // 741, generator)


mean_pr=np.load('/data/ERA5_mean_train.npy', mmap_mode='c')[:,:,5]
std_pr=np.load('/data/ERA5_std_train.npy', mmap_mode='c')[:,:,5]
# load low resolution data for training
predictors=np.load('/data/predictors_train_mean_std_separate.npy', mmap_mode='c')
# load high resolution data for training
obs=np.load('/data/obs_train_mean_std.npy',mmap_mode='c')

train(predictors, obs,160, 64)
