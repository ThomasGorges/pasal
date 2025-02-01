import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import json

matplotlib.rc('font', size=13)

model_paths = [
    '../output/study_results/11082023_fold_999_999/718_a1.40/',
    '../output/study_results/11082023_fold_999_999/718_b0.70/',
    '../output/study_results/11082023_fold_999_999/718/',
    '../output/study_results/11082023_fold_999_999/718_a1.60/',
    '../output/study_results/11082023_fold_999_999/718_b0.90/',
    '../output/study_results/11082023_fold_999_999/718_a0.00_b0.00/',
]

def head_to_name(head):
    head_name = ''

    if head == 'dream_mean':
        head_name = 'DREAM Olfaction Prediction Challenge (mean)'
    elif head == 'dream_std':
        head_name = 'DREAM Olfaction Prediction Challenge (std)'
    elif head == 'arctander':
        head_name = 'Arctander'
    elif head == 'ifra_2019':
        head_name = 'IFRA'
    elif head == 'leffingwell':
        head_name = 'Leffingwell'
    elif head == 'sigma_2014':
        head_name = 'Sigma'
    
    return head_name

fig, ax = plt.subplots(2, 3)
fig.subplots_adjust(hspace=0)
fig.set_figwidth(10)
fig.set_figheight(8)

for model_path in model_paths:
    config_path = model_path + 'config.json'

    alpha = 0
    beta = 0
    with open(config_path) as f:
        config = json.load(f)
        alpha = config['Alpha']
        beta = config['Beta']

    with open(model_path + 'accuracy_history.json') as f:
        accuracy_history = json.load(f)

    with open(model_path + 'loss_balancing_history.json') as f:
        loss_balancing_history = json.load(f)

    lambda_dream_mean = loss_balancing_history['lambda_history']['dream_mean']
    lambda_dream_std = loss_balancing_history['lambda_history']['dream_std']
    val_z_scores = accuracy_history['val']['DREAM']

    color = None
    fmt = '-'
    
    alpha_var = False
    beta_var = False
    is_const = False
    is_default = False

    if alpha == 0.0 and beta == 0.0:
        label = r'const. $\lambda$'
        color = 'black'
        is_const = True
        fmt = '--'
    
    if beta == 0.8 and not alpha == 1.5:
        alpha_var = True
        label = r'$\alpha=$' + f'{alpha:.2f}'

    if alpha == 1.5 and not beta == 0.8:
        beta_var = True
        label = r'$\beta=$' + f'{beta:.2f}'
    
    if alpha == 1.5 and beta == 0.8:
        is_default = True
        alpha_var = True
        beta_var = True

    if is_default:  
        label = r'$\alpha=$' + f'{alpha:.2f}'

    if alpha_var or is_const:
        ax[0][0].plot(np.arange(len(val_z_scores)), val_z_scores, fmt, label='_' + label, color=color)
        ax[0][0].xaxis.set_major_locator(MultipleLocator(10))

        ax[0][1].plot(np.arange(len(lambda_dream_mean)), lambda_dream_mean, fmt, label=label, color=color)
        ax[0][1].xaxis.set_major_locator(MultipleLocator(10))
        
        ax[0][2].plot(np.arange(len(lambda_dream_std)), lambda_dream_std, fmt, label='_' + label, color=color)
        ax[0][2].xaxis.set_major_locator(MultipleLocator(10))

    if is_default:  
        label = r'$\beta=$' + f'{beta:.2f}'

    if beta_var or is_const:
        ax[1][0].plot(np.arange(len(val_z_scores)), val_z_scores, fmt, label='_' + label, color=color)
        ax[1][0].xaxis.set_major_locator(MultipleLocator(10))

        ax[1][1].plot(np.arange(len(lambda_dream_mean)), lambda_dream_mean, fmt, label=label, color=color)
        ax[1][1].xaxis.set_major_locator(MultipleLocator(10))
        
        ax[1][2].plot(np.arange(len(lambda_dream_std)), lambda_dream_std, fmt, label='_' + label, color=color)
        ax[1][2].xaxis.set_major_locator(MultipleLocator(10))

    for x in range(3):
        for y in range(2):
            ax[y][x].set_xlim((0, len(val_z_scores) - 1))

            if x == 0:
                ax[y][x].set_ylim((0.0, 10.0))


ax[0][0].set(xlabel='Epoch', ylabel=r'Z-Score $\uparrow$')
ax[0][1].set(xlabel='Epoch', ylabel=r'$\lambda$')
ax[0][2].set(xlabel='Epoch', ylabel=r'$\lambda$')

ax[1][0].set(xlabel='Epoch', ylabel=r'Z-Score $\uparrow$')
ax[1][1].set(xlabel='Epoch', ylabel=r'$\lambda$')
ax[1][2].set(xlabel='Epoch', ylabel=r'$\lambda$')

ax[0][0].set_title('Performance')
ax[1][0].set_title('Performance')
ax[0][1].set_title('\n\n\n\n' + 'Mean prediction')
ax[1][1].set_title('\n\n\n\n' + 'Mean prediction')
ax[0][2].set_title('Uncertainty prediction')
ax[1][2].set_title('Uncertainty prediction')
plt.tight_layout()
ax[0][1].legend(loc='upper center', ncols=4, fancybox=False, shadow=True, bbox_to_anchor=(0.5, 1.6), title=r'$\beta=0.8$ with $\alpha$ variations')
ax[1][1].legend(loc='upper center', ncols=4, fancybox=False, shadow=True, bbox_to_anchor=(0.5, 1.6), title=r'$\alpha=1.5$ with $\beta$ variations')
plt.savefig('../output/plots/fig9.pdf')
