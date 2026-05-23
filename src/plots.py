import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def plot_res(var_list, fit_times,query_times, motif, N=None, D=None, K=None):
    """Plot for visualizing build time and query time with var_list varying"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'brute': 'blue', 'kd_tree': 'green', 'ball_tree': 'red'}


    # fit time
    for algo, color in colors.items():
        ax1.plot(var_list, fit_times[algo], label=algo, color=color, marker='o')
    ax1.set_title(f'Fit Time vs {motif}')
    ax1.set_xlabel(motif)
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()
    ax1.grid(True)


    # query time
    for algo, color in colors.items():
        ax2.plot(var_list, query_times[algo], label=algo, color=color, marker='o')

    ax2.set_title(f'Query Time vs {motif}')
    ax2.set_xlabel(motif)
    ax2.set_ylabel('Time (seconds)')
    ax2.legend()
    ax2.grid(True)
    #ax2.set_xscale('log')

    plt.suptitle(f'Comparing algos with differing {motif}: N={N}, D={D}, K={K}')
    plt.tight_layout()
    plt.savefig(f'results/knn_performance_{motif}.png', dpi=150)
    plt.show()

def plot_time_per_point(var_list, time_per_point, motif, N=None, D=None, K=None):
    colors = {'brute': 'blue', 'kd_tree': 'green', 'ball_tree': 'red'}
    fig, ax = plt.subplots()
    for algo, color in colors.items():
        ax.plot(var_list, time_per_point[algo], label=algo, color=color)
    ax.set_title('Query Time per Test Point')
    ax.set_xlabel(motif)
    ax.set_ylabel('Time per point (seconds)')
    ax.legend()
    ax.grid(True)

    plt.suptitle(f'Comparing algos with differing {motif}: N={N}, D={D}, K={K}')
    plt.tight_layout()
    plt.savefig(f'results/knn_per_point_{motif}.png', dpi=150)
    plt.show()


def _3D_plot(var_list1, var_list2, motif,query_times,xlabel='X', ylabel='Y', title='3D KNN Performance' ,has_BT=True):
    colors = {'kd_tree': 'green', 'brute': 'blue'}
    if has_BT:
        colors = {'brute': 'blue', 'kd_tree': 'green', 'ball_tree': 'red'}

    # create meshgrid for 3D surface
    X, Y = np.meshgrid(var_list1, var_list2)

    #fig = plt.figure(figsize=(10, 8)) #matplotlib
    fig = go.Figure() #plotly
    #ax = plt.axes(projection='3d')
    for algo, color in colors.items():
        Z = np.array(query_times[algo]).reshape(len(var_list2), len(var_list1))
        '''#matplotlib
        ax.plot_surface(X, Y, Z, alpha=0.3, color=color)
        ax.scatter(X, Y, Z, c=color, s=50, marker='o', zorder=5)
        # add wireframe on top for clarity
        ax.plot_wireframe(X, Y, Z,color=color,linewidth=0.8,alpha=0.8)'''
        #--plotly--
        # surface
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.4,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=algo
        ))

        # scatter points
        fig.add_trace(go.Scatter3d(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            mode='markers',
            marker=dict(size=4, color=color),
            name=algo
        ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title='Query Time (seconds)'
        ),
        width=900,
        height=700
    )
    os.makedirs('results', exist_ok=True)
    fig.write_html(f'results/3d_{xlabel}_{ylabel}_{motif}.html')  # saves as interactive HTML
    fig.show()

    '''ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
    ax.set_zlabel('Query Time (seconds)', fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=12)

    legend_elements = [plt.Line2D([0], [0], color=c, label=a, linewidth=2)
                       for a, c in colors.items()]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'results/3d_{xlabel}_{ylabel}.png', dpi=150)
    plt.show()'''

def plot_heatmaps(var_list1, var_list2, query_times, xlabel='X', ylabel='Y', title="Heatmap", value_format='.4f', colorbar_label='Value', has_brute=False):
    algos = ['kd_tree', 'ball_tree']
    if has_brute:
        algos = ['kd_tree', 'ball_tree', 'brute']
    nb_plots = len(algos)
    fig, axes = plt.subplots(1, nb_plots, figsize=(18, 5))


    for algo, axe in zip(algos, axes):
        Z = np.array(query_times[algo]).reshape(len(var_list2), len(var_list1))

        im = axe.imshow(Z, aspect='auto', cmap='viridis', origin='lower')

        for i in range(len(var_list2)):
            for j in range(len(var_list1)):
                axe.text(j, i, f'{Z[i,j]:{value_format}}', ha='center', va='center', fontsize=8, color='white')


        axe.set_xticks(range(len(var_list1)))
        axe.set_xticklabels(var_list1)
        axe.set_yticks(range(len(var_list2)))
        axe.set_yticklabels(var_list2)
        axe.set_xlabel(xlabel)
        axe.set_ylabel(ylabel)
        axe.set_title(f'{algo} - {title}')
        plt.colorbar(im, ax=axe, label=colorbar_label)

    plt.suptitle(f'{title}: {xlabel} vs {ylabel}')
    plt.tight_layout()
    plt.savefig(f'results/heatmap_{title}_{xlabel}_{ylabel}.png', dpi=150)
    plt.show()