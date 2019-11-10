
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pdb
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def set_axis_prop(ax, grid_on, ticks_on, axis_on):
    ax.grid(grid_on)
    if not ticks_on:
        if hasattr(ax, 'set_xticks'): ax.set_xticks([])
        if hasattr(ax, 'set_yticks'): ax.set_yticks([])
        if hasattr(ax, 'set_zticks'): ax.set_zticks([])
    if not axis_on: ax.axis('off')

def get_full_grid_samples(resolution=100, range_min=-1, range_max=1):
    x0_range = np.linspace(range_min, range_max, 2*resolution+1)
    x1_range = np.linspace(range_min, range_max, 2*resolution+1)
    x0v, x1v = np.meshgrid(x0_range, x1_range)
    grid_flat = np.concatenate([x0v.flatten()[:, np.newaxis], x1v.flatten()[:, np.newaxis]], axis=1)
    grid = np.concatenate([x0v[:,:,np.newaxis], x1v[:,:,np.newaxis]], axis=2)
    return grid_flat, grid, x0_range, x1_range

def get_sparse_grid_samples(resolution=100, subsample_rate=10, range_min=-1, range_max=1):
    full_grid_flat, full_grid, _, _ = get_full_grid_samples(resolution=resolution, range_min=range_min, range_max=range_max)
    index_x0v, index_x1v = np.meshgrid(np.arange(full_grid.shape[0]), np.arange(full_grid.shape[1]))
    index_grid_flat = np.concatenate([index_x0v.flatten()[:, np.newaxis], index_x1v.flatten()[:, np.newaxis]], axis=1)
    index_grid = np.concatenate([index_x0v[:,:,np.newaxis], index_x1v[:,:,np.newaxis]], axis=2)
    subsample_mask = ((index_grid_flat[:, 0]%subsample_rate == 0)+ (index_grid_flat[:, 1]%subsample_rate == 0))>0
    return full_grid_flat[subsample_mask,:]

class GaussianDistribution():
    def __init__(self, mean_vec, cov_matrix):   
        self.mean_vec = np.asarray(mean_vec)
        self.cov_matrix = np.asarray(cov_matrix)
        self.mult_gaussian = multivariate_normal(mean=self.mean_vec, cov=self.cov_matrix)
    
    def sample(self, n_samples):
        return self.mult_gaussian.rvs(n_samples)

    def pdf(self, input_x):
        return self.mult_gaussian.pdf(input_x)

    def log_pdf(self, input_x):
        return np.log(self.pdf(input_x))

####################################################################################
##############################    BASE DISTRIBUTION   ##############################                           
####################################################################################
tf.compat.v1.enable_resource_variables()

my_dpi = 350
grid_on, ticks_on, axis_on = False, False, True

dim = 2 # dim = 3 is for quantitive eval, dim = 2 is for visualizations.
model_id = 2 # model 2 is more complex.
optimization_mode = 'sgld'
assert (optimization_mode == 'sgd' or optimization_mode == 'sgld')

if model_id == 1:
    if dim == 3:
        model_mean_vector = [0, 0, 0]
        model_cov_matrix= [[4, 0, 0], [0, 4, 0], [0, 0, 4.]]
    elif dim == 2:
        model_mean_vector = [0, 0]
        model_cov_matrix= [[4, 0], [0, 4]]
elif model_id == 2: 
    if dim == 3:
        model_mean_vector = [3, 0.3, -1]
        model_cov_matrix= [[4, 1, 1], [1, 4, 1], [1, 0.25, 4]]
    elif dim == 2:
        model_mean_vector = [3, 0.3]
        model_cov_matrix= [[4, 0.25], [0.25, 4]]

if dim == 3: 
    init_var_vector = [4., 2., 1.]
    training_steps = int(1e6)
    vis_n_steps = int(1e5)
elif dim == 2: 
    init_var_vector = [4., -7.]
    training_steps = int(2e3)
    vis_n_steps = int(1e2) 

if dim == 2: 
    plt.close('all')
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10))

    plt.scatter(init_var_vector[0], init_var_vector[1], color='blue')

    resolution = 300
    range_1_min = -10 
    range_1_max = 10

    dist = GaussianDistribution(mean_vec=model_mean_vector, cov_matrix=model_cov_matrix)
    dist_samples = dist.sample(10000)
    dist_samples_pdf = dist.pdf(dist_samples)
    full_grid_samples, full_grid, full_x0_range, full_x1_range = get_full_grid_samples(resolution=resolution, range_min=range_1_min, range_max=range_1_max)
    full_grid_samples_pdf = dist.pdf(full_grid_samples)
    full_grid_pdf = full_grid_samples_pdf.reshape(full_x0_range.shape[0], full_x1_range.shape[0])

    ax.contour(full_x0_range, full_x1_range, full_grid_pdf, levels=np.arange(np.min(full_grid_pdf), np.max(full_grid_pdf), (np.max(full_grid_pdf)-(np.min(full_grid_pdf)))/10))
    ax.axis([range_1_min, range_1_max, range_1_min, range_1_max])
    set_axis_prop(ax, grid_on, ticks_on, axis_on )
    plt.show(block=False)

image_counter = 0

with tf.Session(graph=tf.Graph()) as sess:
    tf.set_random_seed(35)
    global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)

    model_mean = tf.constant(model_mean_vector, dtype=np.float32)
    model_cov = tf.constant(model_cov_matrix, dtype=np.float32)
    model_cov_chol = tf.linalg.cholesky(model_cov)

    theta = tf.compat.v1.Variable(name='theta', initial_value=init_var_vector, trainable=True)

    def loss_fn(): # maximize log pdf of theta under model
        loss_part = tf.linalg.cholesky_solve(model_cov_chol, tf.expand_dims(theta-model_mean, -1)) # find model_cov^-1*(theta-model_mean)
        return tf.linalg.matvec(loss_part, (theta-model_mean), transpose_a=True) # find (theta-model_mean)^T*true_cov^-1*(theta-model_mean), minimizing which maximizes the log pdf of theta under the gaussian.

    if optimization_mode == 'sgd':
        curr_learning_rate = tf.constant(0.1, dtype=np.float32)
        optimizer_kernel = tf.train.AdamOptimizer(learning_rate=curr_learning_rate, beta1=0.99, beta2=0.999, epsilon=1e-08)
        optimizer = optimizer_kernel.minimize(loss_fn, var_list=[theta,])
    elif optimization_mode == 'sgld':
        curr_learning_rate = tf.compat.v1.train.polynomial_decay(learning_rate=0.5, global_step=global_step, decay_steps=training_steps, end_learning_rate=1e-5, power=1.)
        optimizer_kernel = tfp.optimizer.StochasticGradientLangevinDynamics(learning_rate=curr_learning_rate, preconditioner_decay_rate=0.7)
        optimizer_kernel.iterations = global_step
        optimizer = optimizer_kernel.minimize(loss_fn, var_list=[theta,])

    samples = np.zeros([training_steps, theta.get_shape()[0].value])
    lrs = np.zeros([training_steps, 1])
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(training_steps):
        _, theta_np, lr_np = sess.run([optimizer, theta, curr_learning_rate])
        samples[step, :] = theta_np
        lrs[step] = lr_np

        if step % vis_n_steps == 1: 
            print('\n\n')
            print('step: '+str(step)+'/'+str(training_steps))
            print('\n')

            valid_samples = samples[:step, :]
            valid_lrs = lrs[:step, :]
            lr_valid_samples_mean = np.sum(valid_lrs*valid_samples, axis=0)/np.sum(valid_lrs)
            lr_valid_samples_std = np.sqrt(np.sum(valid_lrs*(valid_samples-lr_valid_samples_mean[np.newaxis,:])**2, axis=0)/np.sum(valid_lrs))
            print('samples mean', np.mean(valid_samples, 0))
            print('samples std', np.std(valid_samples, 0))
            print('lr-weighted samples mean', lr_valid_samples_mean)
            print('lr-weighted samples std', lr_valid_samples_std)
            print('\n')

            valid_samples_skip = valid_samples[::200, :]
            valid_lrs_skip = valid_lrs[::200, :]
            lr_valid_samples_skip_mean = np.sum(valid_lrs_skip*valid_samples_skip, axis=0)/np.sum(valid_lrs_skip)
            lr_valid_samples_skip_std = np.sqrt(np.sum(valid_lrs_skip*(valid_samples_skip-lr_valid_samples_skip_mean[np.newaxis,:])**2, axis=0)/np.sum(valid_lrs_skip))
            print('samples skip (200) mean', np.mean(valid_samples_skip, 0))
            print('samples skip (200) std', np.std(valid_samples_skip, 0))
            print('lr-weighted samples skip (200) mean', lr_valid_samples_skip_mean)
            print('lr-weighted samples skip (200) std', lr_valid_samples_skip_std)
            print('\n')

            valid_samples_interval = valid_samples[max(0, step-vis_n_steps):step, :]
            valid_lr_interval = valid_lrs[max(0, step-vis_n_steps):step, :]
            lr_valid_samples_interval_mean = np.sum(valid_lr_interval*valid_samples_interval, axis=0)/np.sum(valid_lr_interval)
            lr_valid_samples_interval_std = np.sqrt(np.sum(valid_lr_interval*(valid_samples_interval-lr_valid_samples_interval_mean[np.newaxis,:])**2, axis=0)/np.sum(valid_lr_interval))
            print('Interval steps: ', max(0, step-vis_n_steps), step)
            print('samples interval mean', np.mean(valid_samples_interval, 0))
            print('samples interval std', np.std(valid_samples_interval, 0))
            print('lr-weighted samples interval mean', lr_valid_samples_interval_mean)
            print('lr-weighted samples interval std', lr_valid_samples_interval_std)
            print('\n')

            valid_samples_interval_skip = valid_samples_interval[::200, :]
            valid_lr_interval_skip = valid_lr_interval[::200, :]
            lr_valid_samples_interval_skip_mean = np.sum(valid_lr_interval_skip*valid_samples_interval_skip, axis=0)/np.sum(valid_lr_interval_skip)
            lr_valid_samples_interval_skip_std = np.sqrt(np.sum(valid_lr_interval_skip*(valid_samples_interval_skip-lr_valid_samples_interval_skip_mean[np.newaxis,:])**2, axis=0)/np.sum(valid_lr_interval_skip))
            print('Interval steps: ', max(0, step-vis_n_steps), step)
            print('samples interval skip (200) mean', np.mean(valid_samples_interval_skip, 0))
            print('samples interval skip (200) std', np.std(valid_samples_interval_skip, 0))
            print('lr-weighted samples interval skip (200) mean', lr_valid_samples_interval_skip_mean)
            print('lr-weighted samples interval skip (200) std', lr_valid_samples_interval_skip_std)
            print('\n')

            if dim == 2 and step > 1: 
                plt.cla()
                ax.contour(full_x0_range, full_x1_range, full_grid_pdf, levels=np.arange(np.min(full_grid_pdf), np.max(full_grid_pdf), (np.max(full_grid_pdf)-(np.min(full_grid_pdf)))/10))
                ax.axis([range_1_min, range_1_max, range_1_min, range_1_max])
                set_axis_prop(ax, grid_on, ticks_on, axis_on )

                # plt.scatter(valid_samples_interval[:,0], valid_samples_interval[:,1], color='red')
                for i in range(valid_samples_interval.shape[0]):
                    if i > 0 and i < 50:
                        delta = valid_samples_interval[i, :]-valid_samples_interval[i-1, :]
                        ax.scatter(valid_samples_interval[i-1, 0], valid_samples_interval[i-1, 1], color='r', alpha=1, edgecolors='k', linewidth=1)
                        quiv = ax.quiver(valid_samples_interval[i-1, 0], valid_samples_interval[i-1, 1], delta[0], delta[1], width=0.002, angles='xy', scale_units='xy', scale=1, color='Teal')

                plt.draw()
                plt.pause(0.2)
                plt.savefig('/Users/mevlana.gemici/sgld_vis/start_contour_'+ str(image_counter)+'.png', bbox_inches='tight', format='png', dpi=my_dpi, transparent=False)
                image_counter += 1

    if dim == 2: 
        plt.cla()
        ax.contour(full_x0_range, full_x1_range, full_grid_pdf, levels=np.arange(np.min(full_grid_pdf), np.max(full_grid_pdf), (np.max(full_grid_pdf)-(np.min(full_grid_pdf)))/10))
        ax.axis([range_1_min, range_1_max, range_1_min, range_1_max])
        set_axis_prop(ax, grid_on, ticks_on, axis_on )

        # plt.scatter(samples[:,0], samples[:,1], color='red')
        for i in range(samples.shape[0]):
            if i > 0:
                delta = samples[i, :]-samples[i-1, :]
                ax.scatter(samples[i-1, 0], samples[i-1, 1], color='r', alpha=1, edgecolors='k', linewidth=1)
                quiv = ax.quiver(samples[i-1, 0], samples[i-1, 1], delta[0], delta[1], width=0.002, angles='xy', scale_units='xy', scale=1, color='Teal')

        plt.scatter(init_var_vector[0], init_var_vector[1], color='blue')
        plt.draw()
        plt.pause(0.1)
        plt.savefig('/Users/mevlana.gemici/sgld_vis/start_contour_'+ str(image_counter)+'.png', bbox_inches='tight', format='png', dpi=my_dpi, transparent=False)
        image_counter += 1

plt.pause(2)
plt.close('all')




