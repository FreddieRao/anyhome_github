import numpy as np
import cv2
from skimage.measure import label
import torch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from termcolor import colored

import warnings
warnings.filterwarnings("ignore")


def one_hot_embedding(labels, num_classes=19):
    y = torch.eye(num_classes)
    return y[labels]

def check_post_processing(nds, eds, masks, door_list, img_size=256):
    """Check whether the output masks are valid based on nodes and edges."""
    def check_is_interconnected(house_layout, val1, val2):
        # Check whether the two rooms are interconnected
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Get the positions of val1 and val2
        positions_val1 = set(zip(*np.where(house_layout == val1)))
        positions_val2 = set(zip(*np.where(house_layout == val2)))

        for x, y in positions_val1:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) in positions_val2:
                    return True
        return False

    # Check whether the output masks are valid based on nodes and edges
    real_nodes = np.where(nds.detach().cpu()==1)[-1]
    if len(real_nodes) != masks.shape[0]:  # If there are extra nodes or less nodes
        print(colored("Floorplan Invalid: Node Number Inequivalent. Retrying...", "grey"))
        return False, None, None, None

    result_masks = []
    for i, m in enumerate(masks):
        m[m > 0] = 255
        m[m < 0] = 0
        m_lg = cv2.resize(m, (img_size, img_size), interpolation = cv2.INTER_AREA)
        # Check whether the mask is null
        if np.all(m_lg == 0):
            print(colored(f"Floorplan Invalid: Zero Mask Error for Node {i}. Retrying...", "grey"))
            return False, None, None, None
        # Check whehter the mask contains one or more separate regions
        labeled_img = label(m_lg)
        num_regions = np.max(labeled_img)
        if num_regions > 1:
            return False, None, None, None
        result_masks.append(m_lg)

    # Parse the final masks into a single integer map
    result_map = np.full([img_size, img_size], -1)
    for i in range(img_size):
        for j in range(img_size):
            for ind, m in enumerate(result_masks):
                if m[i, j] == 255:
                    result_map[i, j] = ind

    # Modify eds to include only the connected pairs that include a door
    eds = eds.tolist()
    eds = [pair for pair in eds if (pair[1] == 1) and (real_nodes[pair[0]] == 14
        or real_nodes[pair[2]] == 14 or real_nodes[pair[0]] == 16
        or real_nodes[pair[2]] == 16)]
    
    # Check if the doors and the rooms are interconnected as stated
    for pair in eds:
        if not check_is_interconnected(result_map, pair[0], pair[2]):
            print(colored("Floorplan Invalid: Disconnected Error. Retrying...", "grey"))
            return False, None, None, None

    result_map_no_doors = np.full([img_size, img_size], -1)
    for i in range(img_size):
        for j in range(img_size):
            for ind, m in enumerate(result_masks[:len(result_masks)-len(door_list)]):
                if m[i, j] == 255:
                    result_map_no_doors[i, j] = ind

    return True, result_map, result_map_no_doors, result_masks

def generate_border_map_no_doors(house_map_no_doors, result_masks, door_values, nds, eds, img_size=256, value=-2):
    """Replace the border pixels of rooms in the given house map with a specific value."""
    # Get unique room values
    house_map = house_map_no_doors.copy()
    rooms = np.unique(house_map)

    # Store the indices to change
    indices_to_change = []

    # Loop through each room
    for room in rooms:
        # Find indices of current room
        indices = np.where(house_map == room)
        # Check adjacent cells
        for r, c in zip(indices[0], indices[1]):
            if r > 0 and house_map[r-1, c] != room:
                indices_to_change.append((r, c))
            if r < house_map.shape[0]-1 and house_map[r+1, c] != room:
                indices_to_change.append((r, c))
            if c > 0 and house_map[r, c-1] != room:
                indices_to_change.append((r, c))
            if c < house_map.shape[1]-1 and house_map[r, c+1] != room:
                indices_to_change.append((r, c))

    # Set the border cells and outside to the given value
    for r, c in indices_to_change:
        house_map[r, c] = value

    house_map[house_map == -1] = -10
    real_nodes = np.where(nds.detach().cpu()==1)[-1]
    eds = eds.tolist()
    # modify eds to include only the connected pairs that include a door
    eds = [pair for pair in eds if (pair[1] == 1) and (real_nodes[pair[0]] == 14
        or real_nodes[pair[2]] == 14 or real_nodes[pair[0]] == 16
        or real_nodes[pair[2]] == 16)]
    eds.append([-10, 1, door_values[-1]])  # add the connection between the front door and the outside
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    front_door_coors = []

    # Handling the doors
    for i in range(1, img_size-1):
        for j in range(1, img_size-1):
            for door in door_values:
                if house_map[i, j] == value and result_masks[door][i, j] == 255:
                    if door == door_values[-1]:
                        house_map[i, j] = door
                        front_door_coors.append([i, j])
                        break
                    for (dx, dy) in directions:
                        neighbor = house_map[i+dx, j+dy]
                        if neighbor == -2 or neighbor == -1 or neighbor == door:
                            continue
                        if not ([neighbor, 1, door] in eds):
                            break
                    else:
                        house_map[i, j] = -1
                    break

    # Clear the single_unit walls
    four_directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    indices_to_change = []
    for i in range(2, img_size-2):
        for j in range(2, img_size-2):
            if house_map[i, j] == -1:
                for (dx, dy) in four_directions:
                    if house_map[i+dx, j+dy] == -1 and house_map[i+dx+dx, j+dy+dy] != -1:
                        break
                else:
                    indices_to_change.append((i, j))

    for (x, y) in indices_to_change:
        house_map[x, y] = -2

    for i in range(2, img_size-2):
        for j in range(2, img_size-2):
            if house_map[i, j] == -1:
                for (dx, dy) in four_directions:
                    if house_map[i+dx, j+dy] == -1:
                        break
                else:
                    house_map[i, j] = -2

    # Modify the front door
    four_directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    indices_to_change = []
    for i in range(2, img_size-2):
        for j in range(2, img_size-2):
            if house_map[i, j] == door_values[-1]:
                for (dx, dy) in four_directions:
                    if house_map[i+dx, j+dy] == door_values[-1] and house_map[i+dx+dx, j+dy+dy] != door_values[-1]:
                        break
                else:
                    indices_to_change.append((i, j))

    for (x, y) in indices_to_change:
        house_map[x, y] = -2

    for i in range(2, img_size-2):
        for j in range(2, img_size-2):
            if house_map[i, j] == door_values[-1]:
                for (dx, dy) in four_directions:
                    if house_map[i+dx, j+dy] == door_values[-1]:
                        break
                else:
                    house_map[i, j] = -2

    start_coors = tuple(front_door_coors[len(front_door_coors)//2])

    return house_map, start_coors

def decompose_into_rectangles(binary_img, min_pixels):
    """Decompose an irregular room shape into regular rectangles and return their bounding boxes and centers."""
    # Calculate bounding box of the whole room
    rows, cols = np.where(binary_img == 1)
    x1, y1, x2, y2 = min(rows), min(cols), max(rows), max(cols)
    # Check if bounding box covers at least 90% of the room's area
    area_bbox = (x2 - x1 + 1) * (y2 - y1 + 1)
    area_room = np.sum(binary_img)
    if area_room >= 0.80 * area_bbox:
        x_center = x1 + (x2 - x1) // 2
        y_center = y1 + (y2 - y1) // 2
        return [(x1, y1, x2, y2)], [(x_center, y_center)]

    # Store rectangles and centers from vertical and horizontal searches
    rectangles_vertical = []
    rectangles_horizontal = []
    centers_vertical = []
    centers_horizontal = []

    # Perform vertical-first search
    img_copy = binary_img.copy()
    while np.sum(img_copy) > 0:
        # Get indices of all remaining pixels
        rows, cols = np.where(img_copy == 1)
        # Start from top left pixel
        x_start, y_start = rows[0], cols[0]
        # Initialize width and height of rectangle
        width, height = 1, 1
        # Expand height first
        while x_start + height < img_copy.shape[0] and np.all(img_copy[x_start:x_start + height + 1, y_start]):
            height += 1
        # Then expand width
        while y_start + width < img_copy.shape[1] and np.all(img_copy[x_start:x_start + height, y_start:y_start + width + 1]):
            width += 1

        # Save the rectangle coordinates
        rectangles_vertical.append((x_start, y_start, x_start + height - 1, y_start + width - 1))
        # Calculate and save the center coordinates
        x_center = x_start + (height - 1) // 2
        y_center = y_start + (width - 1) // 2
        centers_vertical.append((x_center, y_center))
        # Remove the pixels of the found rectangle from the binary image
        img_copy[x_start:x_start + height, y_start:y_start + width] = 0

    # Perform horizontal-first search
    img_copy = binary_img.copy()
    while np.sum(img_copy) > 0:
        # Get indices of all remaining pixels
        rows, cols = np.where(img_copy == 1)
        # Start from top left pixel
        x_start, y_start = rows[0], cols[0]
        # Initialize width and height of rectangle
        width, height = 1, 1
        # Expand width first
        while y_start + width < img_copy.shape[1] and np.all(img_copy[x_start, y_start:y_start + width + 1]):
            width += 1
        # Then expand height
        while x_start + height < img_copy.shape[0] and np.all(img_copy[x_start:x_start + height + 1, y_start:y_start + width]):
            height += 1
        # Only save the rectangle if it is larger than the specified number of pixels

        # Save the rectangle coordinates
        rectangles_horizontal.append((x_start, y_start, x_start + height - 1, y_start + width - 1))
        # Calculate and save the center coordinates
        x_center = x_start + (height - 1) // 2
        y_center = y_start + (width - 1) // 2
        centers_horizontal.append((x_center, y_center))
        # Remove the pixels of the found rectangle from the binary image
        img_copy[x_start:x_start + height, y_start:y_start + width] = 0

    if len(rectangles_vertical) > 1:
        # Only save the rectangle if it is larger than the specified number of pixels
        valid_rectangles_vertical = [rect for rect in rectangles_vertical
                                    if (rect[2] - rect[0] + 1) >= min_pixels and (rect[3] - rect[1] + 1) >= min_pixels]
        valid_centers_vertical = [center for rect, center in zip(rectangles_vertical, centers_vertical)
                                if (rect[2] - rect[0] + 1) >= min_pixels and (rect[3] - rect[1] + 1) >= min_pixels]

    if len(rectangles_horizontal) > 1:
        # Do the same for horizontal rectangles
        valid_rectangles_horizontal = [rect for rect in rectangles_horizontal
                                    if (rect[2] - rect[0] + 1) >= min_pixels and (rect[3] - rect[1] + 1) >= min_pixels]
        valid_centers_horizontal = [center for rect, center in zip(rectangles_horizontal, centers_horizontal)
                                    if (rect[2] - rect[0] + 1) >= min_pixels and (rect[3] - rect[1] + 1) >= min_pixels]


    # Calculate total areas of rectangles from both searches
    total_area_vertical = sum([(x2 - x1 + 1) * (y2 - y1 + 1) for x1, y1, x2, y2 in valid_rectangles_vertical])
    total_area_horizontal = sum([(x2 - x1 + 1) * (y2 - y1 + 1) for x1, y1, x2, y2 in valid_rectangles_horizontal])

    # Return rectangles and centers from the search that produced the larger total area
    if total_area_vertical > total_area_horizontal:
        return valid_rectangles_vertical, valid_centers_vertical
    else:
        return valid_rectangles_horizontal, valid_centers_horizontal

def get_room_boundaries(map_array, front_door_index, start_coors, min_pixels=12):
    """Get the bounding boxes and centers of each room in the given map array."""
    # Get unique room numbers
    room_numbers = np.unique(map_array)
    room_numbers = room_numbers[room_numbers != -1]  # Exclude -1 (door)
    room_numbers = room_numbers[room_numbers != -2]  # Exclude -2 (wall)
    room_numbers = room_numbers[room_numbers != -10]  # Exclude -10 (outside)
    room_numbers = room_numbers[room_numbers != front_door_index]
    # print(room_numbers)

    boxes = {}
    centers = {}

    # Add the front door indices
    centers[front_door_index] = [start_coors]

    for room in room_numbers:
        # print("room", room)
        # Create binary image for each room
        binary_img = np.where(map_array == room, 1, 0).astype(np.int32)

        # Decompose the room into rectangles
        rectangles, center = decompose_into_rectangles(binary_img, min_pixels)
        # print(rectangles, center)
        boxes[room] = rectangles
        centers[room] = center

    return boxes, centers

def visualize_map_with_centers(map_array, boxes, centers):
    # Define the custom colors
    custom_colors = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), *plt.cm.tab20b(np.linspace(0, 1, 19))]
    # Create the custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', custom_colors)

    fig, ax = plt.subplots()
    ax.imshow(map_array, cmap=custom_cmap)

    for room, room_boxes in boxes.items():
        for box in room_boxes:
            y1, x1, y2, x2 = box
            rect = plt.Rectangle((x1-0.5, y1-0.5), x2 - x1 + 1, y2 - y1 + 1, fill=False, edgecolor='white', linewidth=1)
            ax.add_patch(rect)

    # Plot the points from coor_list
    for room in centers.values():
        for (y, x) in room:
            ax.scatter(x, y, color='red')  # color and size of points can be customized

    plt.show()

def find_segments(grid):
    """Find the segments of the grid."""
    segments = []

    # Find horizontal segments
    for i in range(grid.shape[0]):
        start_col = None
        for j in range(grid.shape[1]):
            if grid[i, j] == -2 and start_col is None:
                start_col = j
            elif grid[i, j] != -2 and start_col is not None:
                if j - start_col > 2:
                    segments.append(((i, start_col), (i, j-1)))
                start_col = None

    # Find vertical segments
    for j in range(grid.shape[1]):
        start_row = None
        for i in range(grid.shape[0]):
            if grid[i, j] == -2 and start_row is None:
                start_row = i
            elif grid[i, j] != -2 and start_row is not None:
                if i - start_row > 2:
                    segments.append(((start_row, j), (i-1, j)))
                start_row = None

    return segments


def generate_floor_mesh(grid, vertices, faces, vertex_count):
    """Generate the floor mesh from the grid."""
    # Convert the grid into binary: 1 for spaces without -10 and 0 otherwise
    binary_grid = np.where(grid != -10, 1, 0).astype('uint8')

    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_grid)

    for i in range(1, num_labels):
        # Identify rows and columns with the current label
        rows, cols = np.where(labels == i)
        # Create the bounding box from these rows and cols
        min_i, max_i = np.min(rows), np.max(rows)
        min_j, max_j = np.min(cols), np.max(cols)
        # Append the corners of the bounding box to the vertices list.
        vertices.extend([[min_i, min_j, 0],
                         [max_i, min_j, 0],
                         [max_i, max_j, 0],
                         [min_i, max_j, 0]])
        # Append the faces
        faces.extend([[vertex_count, vertex_count+1, vertex_count+2],
                      [vertex_count, vertex_count+2, vertex_count+3]])

        vertex_count += 4

    return vertices, faces, vertex_count


def write_to_obj(segments, grid, base_mesh_dir="output/base_mesh.obj"):
    """Write the base structure to an obj file."""
    wall_thickness = 0.05
    wall_height = 25
    vertices = []
    faces = []

    vertex_count = 1

    for segment in segments:
        (x1, y1), (x2, y2) = segment
        if x1 == x2:  # Vertical
            vertices.extend([
                [x1-wall_thickness, y1, 0],
                [x1+wall_thickness, y1, 0],
                [x1+wall_thickness, y2, 0],
                [x1-wall_thickness, y2, 0],
                [x1-wall_thickness, y1, wall_height],
                [x1+wall_thickness, y1, wall_height],
                [x1+wall_thickness, y2, wall_height],
                [x1-wall_thickness, y2, wall_height]
            ])
            faces.extend([
                [vertex_count, vertex_count + 1, vertex_count + 2],
                [vertex_count, vertex_count + 2, vertex_count + 3],
                [vertex_count + 4, vertex_count + 5, vertex_count + 6],
                [vertex_count + 4, vertex_count + 6, vertex_count + 7],
                [vertex_count, vertex_count + 1, vertex_count + 5],
                [vertex_count, vertex_count + 5, vertex_count + 4],
                [vertex_count + 1, vertex_count + 2, vertex_count + 6],
                [vertex_count + 1, vertex_count + 6, vertex_count + 5],
                [vertex_count + 2, vertex_count + 3, vertex_count + 7],
                [vertex_count + 2, vertex_count + 7, vertex_count + 6],
                [vertex_count + 3, vertex_count, vertex_count + 4],
                [vertex_count + 3, vertex_count + 4, vertex_count + 7]
            ])
        else:  # Horizontal
            vertices.extend([
                [x1, y1-wall_thickness, 0],
                [x2, y1-wall_thickness, 0],
                [x2, y1+wall_thickness, 0],
                [x1, y1+wall_thickness, 0],
                [x1, y1-wall_thickness, wall_height],
                [x2, y1-wall_thickness, wall_height],
                [x2, y1+wall_thickness, wall_height],
                [x1, y1+wall_thickness, wall_height]
            ])
            faces.extend([
                [vertex_count, vertex_count + 1, vertex_count + 2],
                [vertex_count, vertex_count + 2, vertex_count + 3],
                [vertex_count + 4, vertex_count + 5, vertex_count + 6],
                [vertex_count + 4, vertex_count + 6, vertex_count + 7],
                [vertex_count, vertex_count + 1, vertex_count + 5],
                [vertex_count, vertex_count + 5, vertex_count + 4],
                [vertex_count + 1, vertex_count + 2, vertex_count + 6],
                [vertex_count + 1, vertex_count + 6, vertex_count + 5],
                [vertex_count + 2, vertex_count + 3, vertex_count + 7],
                [vertex_count + 2, vertex_count + 7, vertex_count + 6],
                [vertex_count + 3, vertex_count, vertex_count + 4],
                [vertex_count + 3, vertex_count + 4, vertex_count + 7]
            ])

        vertex_count += 8

    vertices, faces, _ = generate_floor_mesh(grid, vertices, faces, vertex_count)

    with open(base_mesh_dir, "w") as file:
        for v in vertices:
            file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for f in faces:
            if len(f) == 4:
                file.write("f {} {} {} {}\n".format(f[0], f[1], f[2], f[3]))
            else:
                file.write("f {} {} {}\n".format(f[0], f[1], f[2]))

    return vertices, faces