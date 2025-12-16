import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgb
import numpy as np
from scipy.stats import multivariate_normal

def plot_gmm_clusters(samples, means, covariances, weights, trajectories=False, point_size=1, alpha=0.1, low_traj=False, plot_ellipses=False):

    n_components = len(weights)
    n_samples = samples.shape[0]
    responsibilities = np.zeros((n_samples, n_components))
    
    for k in range(n_components):
        if weights[k] >= 0.01: 
            rv = multivariate_normal(mean=means[k], cov=covariances[k], allow_singular=True)
            responsibilities[:, k] = weights[k] * rv.pdf(samples)
        else:
            responsibilities[:, k] = -1
            
    labels = np.argmax(responsibilities, axis=1)
    
    if n_components <= 7:
        colors = [
            "#EE6677",  
            "#228833",  
            "#4477AA",  
            "#CCBB44",  
            "#66CCEE",  
            "#AA3377",  
            "#BBBBBB"   
        ]
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] * (n_components // 10 + 1)
        colors = colors[:n_components]
    
    plt.figure(figsize=(8, 6), dpi=100)
    
    if trajectories is not False:
        for k in range(n_components):
            if low_traj or weights[k] >= 0.01:
                skip = max(1, trajectories.shape[2] // 500) 
                traj = trajectories[k, :, ::skip]
                if weights[k] >= 0.01:
                    plt.plot(traj[0], traj[1], '-', color=colors[k], lw=3, 
                            zorder=3, alpha=0.9, label=f'Trajectory {k}')
                else:
                    plt.plot(traj[0], traj[1], '--', color=colors[k], lw=1, 
                            zorder=3, alpha=0.9, label=f'Trajectory {k}')
                plt.scatter(traj[0,0], traj[1,0], marker='s', color='white', 
                          s=100, zorder=4, edgecolors=colors[k], linewidth=1)
                plt.scatter(traj[0,0], traj[1,0], marker='s', color=colors[k], 
                          s=60, zorder=5)
                
    for k in range(n_components):
        if weights[k] >= 0.01: 
            cluster_points = samples[labels == k]
            n_cluster_points = len(cluster_points)
            print(n_cluster_points)
            
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[k], s=point_size, alpha=alpha, 
                       rasterized=True, zorder=1) 
    
    if low_traj:
        mean_sizes = 200
        plt.scatter(means[:, 0], means[:, 1], c='white', marker='x', 
                   s=mean_sizes + 50, linewidths=4, zorder=7)
        plt.scatter(means[:, 0], means[:, 1], c='black', marker='x', 
                   s=mean_sizes, linewidths=3, zorder=8)
    else:
        significant_indices = weights >= 0.01
        if np.any(significant_indices):
            significant_means = means[significant_indices]
            significant_weights = weights[significant_indices]
            mean_sizes = 100 + 800 * significant_weights
            plt.scatter(significant_means[:, 0], significant_means[:, 1], c='white', marker='x', 
                       s=mean_sizes + 50, linewidths=4, zorder=7)
            plt.scatter(significant_means[:, 0], significant_means[:, 1], c='black', marker='x', 
                       s=mean_sizes, linewidths=3, zorder=8)
    
    if plot_ellipses:
        for k in range(n_components):
            if low_traj or weights[k] >= 0.01:
                cov = covariances[k]
                mean = means[k]
                
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = eigvals.argsort()[::-1]
                eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                
                theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
                width, height = 2 * np.sqrt(5.991 * eigvals) 
                
                from matplotlib.patches import Ellipse
                ell = Ellipse(xy=mean, width=width, height=height, angle=theta, 
                            edgecolor=colors[k], facecolor='none', lw=2, 
                            alpha=0.7, zorder=2)
                plt.gca().add_patch(ell)
    
    from matplotlib.lines import Line2D
    legend_elements = []
    for k in range(n_components):
        if trajectories is not False and (low_traj or weights[k] >= 0.01):
            if weights[k] >= 0.01:  
                legend_elements.append(
                    Line2D([0], [0], color=colors[k], lw=3, 
                          label=f'Traj. {k} (w={weights[k]:.3f})'))
            else:        
                legend_elements.append(
                    Line2D([0], [0], color=colors[k], lw=1, ls='--',
                          label=f'Traj. {k} (w={weights[k]:.3f})'))
        elif trajectories is False and weights[k] >= 0.01:
            legend_elements.append(
                Line2D([0], [0], color=colors[k], lw=3, 
                      label=f'Comp. {k} (w={weights[k]:.3f})'))

    legend_elements.append(
        Line2D([0], [0], marker='x', color='black', lw=0, 
               markersize=10, markeredgewidth=3, label='Final Means'))
    
    if trajectories is not False:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='gray', lw=0, 
                   markersize=8, markerfacecolor='white', 
                   markeredgecolor='gray', markeredgewidth=2, label='Starting Means'))
    
    legend = plt.legend(handles=legend_elements, loc='upper left', 
                       fontsize=8, framealpha=0.9)
    legend.set_zorder(100) 
    
    plt.title(f'GERVE Clusters ({n_components} components)', 
              fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    
    lim = 3
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    
    plt.tight_layout()
    plt.show()
