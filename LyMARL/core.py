import os 
import sys

import numpy as np
import matplotlib.pyplot as plt


from typing import List, Tuple
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
path = os.path.join(cur_dir, 'UA')
# sys.path.append('/Users/sungweon-hong/Projects/rlkit/test_env/UA')
# sys.path.append('/Users/sungweon-hong/Projects/rlkit/test_env/UA')
sys.path.append(path)


def generate_triangle_coverage(area_size: float=100, 
                               coverage_radius: float=35, 
                               spacing: float=1.2) -> list: 
    """ 
    Generate a triangle layout for small cell coverage visualization. 
    
    Parameters: 
    - area_size (float): Size of the area (default=100) 
    - coverage_radius (float): Coverage radius for small cells (default=35) 
    - spacing (float): Spacing between small cells (default=1.2)

    Returns: 
    - list: List of positions for small cells in the triangle layout 
    """
    spacing = 1.2 * coverage_radius  # Spacing between base stations    
    center_x, center_y = area_size / 2, area_size / 2
    # Calculate positions of small cells in a triangle layout
    small_bs_positions = [
        (center_x - spacing / 2, center_y - spacing / (2 * np.sqrt(3))),  # Bottom left
        (center_x + spacing / 2, center_y - spacing / (2 * np.sqrt(3))), # Bottom right
        (center_x, center_y + spacing / np.sqrt(3)),  # Top
    ]
    return small_bs_positions


def plot_associations(sbs_list, ue_list, all_bs, 
                      associations, sumrate): 
    fig, ax = plt.subplots(figsize=(6, 6)) 
    sumrate_gbps = sumrate / 10e9 
    for bs in sbs_list:  
        ax.scatter(bs.position[0], bs.position[1], label='SBS' if bs.bs_id==1 else None, s=100, marker='^', facecolors='none', edgecolors='black')
        ax.text(bs.position[0]+5, bs.position[1], f'SBS{bs.bs_id}', fontsize=12, color='black')
        circle = plt.Circle(bs.position, bs.coverage_radius, color='blue', fill=False, linestyle='--', alpha=0.4) 
        ax.add_patch(circle)

    for ue in ue_list: 

        ax.scatter(ue.position[0], ue.position[1], label='UE' if ue.ue_id==1 else None, s=50, marker='o', color='green')
        ax.text(ue.position[0]+5, ue.position[1], f'UE{ue.ue_id}', fontsize=12, color='red')
        
        associated_bs_id = associations.get(ue.ue_id) 
        if associated_bs_id is not None: 
            bs = next(bs for bs in all_bs if bs.bs_id == associated_bs_id)
            ax.plot([ue.position[0], bs.position[0]], [ue.position[1], bs.position[1]], 
                    color='black', linestyle='--', alpha=0.5)  # Adjust linewidth based on SNR
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f"UE and SBS associations ({sumrate_gbps:.2f}Gbps)") 
    ax.legend() 
    ax.grid(True); ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    # Save the plot
    plt.savefig('associations.png', dpi=300)
    plt.show()