import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualize_room_plan(a, b, c, collision_map):
    index_to_name = {}
    for idx, furniture_name in enumerate(a.keys()):
        if "lamp" in furniture_name:
            continue
        index_to_name[idx + 1] = furniture_name

    # Display collision_map with bounding boxes and labels using Matplotlib
    plt.imshow(np.transpose(collision_map), cmap='gray')

    # Draw bounding boxes and labels
    for idx, (furniture_name, position) in enumerate(a.items()):
        if "lamp" in furniture_name:
            continue
        dimensions = b[furniture_name]
        width, height = dimensions
        rect = patches.Rectangle((position[0], position[1]), width, height,
                                linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(position[0] + width / 2, position[1] + height / 2,
                f"{idx + 1}", color='r',
                fontsize=8, ha='center', va='center')

        orientation = c[furniture_name]
        if orientation == 'E':
            plt.annotate("", xy=(position[0] + width / 2 + 5, position[1] + height / 4),
                        xytext=(position[0] + width / 2, position[1] + height / 4),
                        arrowprops=dict(arrowstyle="->", color='g', linewidth=2))
        elif orientation == 'W':
            plt.annotate("", xy=(position[0] + width / 2 - 5, position[1] + height / 4),
                        xytext=(position[0] + width / 2, position[1] + height / 4),
                        arrowprops=dict(arrowstyle="->", color='g', linewidth=2))
        elif orientation == 'N':
            plt.annotate("", xy=(position[0] + width / 4, position[1] + height / 2 - 5),
                        xytext=(position[0] + width / 4, position[1] + height / 2),
                        arrowprops=dict(arrowstyle="->", color='g', linewidth=2))
        elif orientation == 'S':
            plt.annotate("", xy=(position[0] + width / 4, position[1] + height / 2 + 5),
                        xytext=(position[0] + width / 4, position[1] + height / 2),
                        arrowprops=dict(arrowstyle="->", color='g', linewidth=2))

    # Create legend mapping index to furniture name
    legend_labels = [f"{idx}: {furniture_name}" for idx, furniture_name in index_to_name.items()]
    plt.legend(legend_labels, loc='best', fontsize=8)
    plt.show()