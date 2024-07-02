import numpy as np
import random


# All the placement rule code uses NumPy's indexing system
def is_valid_position(x, y, w, h, collision_map, buffer=3):
    """Check if the furniture can be placed at the given position"""
    if x is None or y is None:
        return False

    if not (0 < x < x + w < collision_map.shape[0] and 0 < y < y + h < collision_map.shape[1]):  # index out of range
        return False

    # Avoid placing furniture sticked to the doors
    door_buffer = 6
    if -1 in collision_map[x - door_buffer:x + w + door_buffer, y - door_buffer:y + h + door_buffer]:
        return False
    
    furniture_buffer = min(buffer, 3)  # buffer between the furnitures; min is used to avoid discarding furnitures that need to stick together
    # Avoid placing furniture sticked to the other furnitures
    if 255 in collision_map[x - furniture_buffer:x + w + furniture_buffer, y - furniture_buffer:y + h + furniture_buffer]:
        return False

    return np.all(collision_map[x:x + w, y:y + h] == 1)  # if within the room's bounding box

def place_corner(bbox, centers, size, collision_map):
    """Place the furniture at a corner"""
    corners = []
    for i, (x1, y1, x2, y2) in enumerate(bbox):
        corners.extend([(x1, y2, 0, i), (x1, y1, 1, i), (x2, y1, 2, i), (x2, y2, 3, i)])  # (x, y, orientation, index)

    deleted_corners = []

    for i, corner1 in enumerate(corners[:-1]):
        for j, corner2 in enumerate(corners[i + 1:]):
            if corner1[3] == corner2[3]:  # if they are from the same bbx
                continue
            if abs(corner1[0] - corner2[0]) < 3 and abs(corner1[1] - corner2[1]) < 3:
                deleted_corners.extend([corner1, corner2])

    corners = [corner for corner in corners if corner not in deleted_corners]

    for corner in corners:
        if corner[2] == 0:  # bottom-left corner
            x, y, ang = corner[0], corner[1] - size[1], "N"  # facing top
            if is_valid_position(x, y, *size, collision_map):
                return x, y, ang
            x, y, ang = corner[0], corner[1] - size[0], "E"  # facing right, size[0] because it's rotated
            if is_valid_position(x, y, *size[::-1], collision_map):
                return x, y, ang
        elif corner[2] == 1:  # top-left corner
            x, y, ang = corner[0], corner[1], "S"  # facing bottom
            if is_valid_position(x, y, *size, collision_map):
                return x, y, ang
            x, y, ang = corner[0], corner[1], "E"  # facing right
            if is_valid_position(x, y, *size[::-1], collision_map):
                return x, y, ang
        elif corner[2] == 0:  # toneight corner
            x, y, ang = corner[0] - size[0], corner[1], "S"  # facing bottom
            if is_valid_position(x, y, *size, collision_map):
                return x, y, ang
            x, y, ang = corner[0] - size[1], corner[1], "W"  # facing left
            if is_valid_position(x, y, *size[::-1], collision_map):
                return x, y, ang
        else:  # bottom-right corner
            x, y, ang = corner[0] - size[0], corner[1] - size[1], "N"  # facing top
            if is_valid_position(x, y, *size, collision_map):
                return x, y, ang
            x, y, ang = corner[0] - size[1], corner[1] - size[0], "W"  # facing left
            if is_valid_position(x, y, *size[::-1], collision_map):
                return x, y, ang

    return None, None, None


def place_wall(bbox, centers, size, collision_map):
    """Place the furniture with its back against the wall"""
    def is_adjacent_wall(x, y):
        return collision_map[x, y + 1] == 0 or collision_map[x + 1, y] == 0 or \
            collision_map[x, y - 1] == 0 or collision_map[x - 1, y] == 0

    for (x1, y1, x2, y2) in bbox:
        buffer_x, buffer_y = (x2 - x1) // 8, (y2 - y1) // 8

        # search the left edge
        for y in range(y1 + buffer_y, y2 - size[0] - buffer_y):
            if not is_adjacent_wall(x1, y):
                break
            if is_valid_position(x1, y, *size[::-1], collision_map):  # the furniture is rotated
                return x1, y, "E"
        # search the right edge
        for y in range(y1 + buffer_y, y2 - size[0] - buffer_y):
            if not is_adjacent_wall(x2, y):
                break
            if is_valid_position(x2 - size[1], y, *size[::-1], collision_map):  # the furniture is rotated
                return x2 - size[1], y, "W"
        # search the top edge
        for x in range(x1 + buffer_x, x2 - size[0] - buffer_x):
            if not is_adjacent_wall(x, y1):
                break
            if is_valid_position(x, y1, *size, collision_map):  # the furniture is rotated
                return x, y1, "S"
        # search the bottom edge
        for x in range(x1 + buffer_x, x2 - size[0] - buffer_x):
            if not is_adjacent_wall(x, y2):
                break
            if is_valid_position(x, y2 - size[1], *size, collision_map):  # the furniture is rotated
                return x, y2 - size[1], "N"

    return None, None, None

def place_center(bbox, centers, size, collision_map):
    """Place the furniture at the center of a rectangle in the room, since a irregular room will be split into multiple rectangles."""
    for i, center in enumerate(centers):
        x1, y1, x2, y2 = bbox[i]
        if x2 - x1 >= y2 - y1:  # if length >= height, place the furniture parallel to the length
            if is_valid_position(center[0] - size[0] // 2, center[1] - size[1] // 2, *size, collision_map):  # do not rotate
                return center[0] - size[0] // 2, center[1] - size[1] // 2, random.choice(["N", "S"])
            if is_valid_position(center[0] - size[1] // 2, center[1] - size[0] // 2, *size[::-1], collision_map):
                return center[0] - size[1] // 2, center[1] - size[0] // 2, random.choice(["E", "W"])
        else:
            if is_valid_position(center[0] - size[1] // 2, center[1] - size[0] // 2, *size[::-1], collision_map):
                return center[0] - size[1] // 2, center[1] - size[0] // 2, random.choice(["E", "W"])
            if is_valid_position(center[0] - size[0] // 2, center[1] - size[1] // 2, *size, collision_map):  # do not rotate
                return center[0] - size[0] // 2, center[1] - size[1] // 2, random.choice(["N", "S"])

    return None, None, None

# All the rules below requires an anchor furniture to be placed first, and the paramters include the position, size, and angle of the anchor furniture
def place_beside(size, bed_pos, bed_siz, bed_ang, buffer, collision_map): 
    """Place the furniture beside an anchor furniture"""
    if bed_ang == "S":  # the bed is facing the bottom
        if is_valid_position(bed_pos[0] - size[0] - buffer, bed_pos[1], *size, collision_map, buffer):
            return bed_pos[0] - size[0] - buffer, bed_pos[1], "S"
        if is_valid_position(bed_pos[0] + bed_siz[0] + 1 + buffer, bed_pos[1], *size, collision_map, buffer):
            return bed_pos[0] + bed_siz[0] + 1 + buffer, bed_pos[1], "S"
    elif bed_ang == "N":  # the bed is facing the top
        if is_valid_position(bed_pos[0] - size[0] - buffer, bed_pos[1] + bed_siz[1] - size[1], *size, collision_map, buffer):
            return bed_pos[0] - size[0] - buffer, bed_pos[1] + bed_siz[1] - size[1], "N"
        if is_valid_position(bed_pos[0] + bed_siz[0] + 1 + buffer, bed_pos[1] + bed_siz[1] - size[1], *size, collision_map, buffer):
            return bed_pos[0] + bed_siz[0] + 1 + buffer, bed_pos[1] + bed_siz[1] - size[1], "N"
    elif bed_ang == "E":  # the bed is facing the right
        if is_valid_position(bed_pos[0], bed_pos[1] - size[0] - buffer, *size[::-1], collision_map, buffer):
            return bed_pos[0], bed_pos[1] - size[0] - buffer, "E"
        if is_valid_position(bed_pos[0], bed_pos[1] + bed_siz[1] + 1 + buffer, *size[::-1], collision_map, buffer):
            return bed_pos[0], bed_pos[1] + bed_siz[1] + 1 + buffer, "E"
    else:  # the bed is facing the left
        if is_valid_position(bed_pos[0] + bed_siz[0] - size[1], bed_pos[1] - size[0] - buffer, *size[::-1], collision_map, buffer):
            return bed_pos[0] + bed_siz[0] - size[1], bed_pos[1] - size[0] - buffer, "W"
        if is_valid_position(bed_pos[0] + bed_siz[0] - size[1], bed_pos[1] + bed_siz[1] + 1 + buffer, *size[::-1], collision_map, buffer):
            return bed_pos[0] + bed_siz[0] - size[1], bed_pos[1] + bed_siz[1] + 1 + buffer, "W"

    return None, None, None

def place_around(size, bed_pos, bed_siz, bed_ang, buffer, collision_map):
    """Place the furniture around an anchor furniture"""
    nonzero_indices = np.argwhere(collision_map == 1)
    min_row, min_col = np.min(nonzero_indices, axis=0)
    max_row, max_col = np.max(nonzero_indices, axis=0)
    width = max_col - min_col
    height = max_row - min_row

    if width > height:
        # The left edge
        for y in range(bed_pos[1] + bed_siz[1] // 4, bed_pos[1] + bed_siz[1] - bed_siz[1] // 4):
            if is_valid_position(bed_pos[0] - size[1], y, *size[::-1], collision_map, buffer):  # it is rotated
                return bed_pos[0] - size[1] - 1 - buffer, y, "E"

        # The right edge
        for y in range(bed_pos[1] + bed_siz[1] // 4, bed_pos[1] + bed_siz[1] - bed_siz[1] // 4):
            if is_valid_position(bed_pos[0] + bed_siz[0] + 1, y, *size[::-1], collision_map, buffer):  # it is rotated
                return bed_pos[0] + bed_siz[0] + 1 + buffer, y, "W"

        # The top edge
        for x in range(bed_pos[0] + bed_pos[0] // 4, bed_pos[0] + bed_siz[0] - bed_siz[0] // 4):
            if is_valid_position(x, bed_pos[1] - size[1], *size, collision_map, buffer):
                return x, bed_pos[1] - size[1] - 1 - buffer, "S"

        # The bottom edge
        for x in range(bed_pos[0] + bed_pos[0] // 4, bed_pos[0] + bed_siz[0] - bed_siz[0] // 4):
            if is_valid_position(x, bed_pos[1] + bed_siz[1] + 1, *size, collision_map, buffer):
                return x, bed_pos[1] + bed_siz[1] + 1 + buffer, "N"
    else:
        # The top edge
        for x in range(bed_pos[0] + bed_pos[0] // 4, bed_pos[0] + bed_siz[0] - bed_siz[0] // 4):
            if is_valid_position(x, bed_pos[1] - size[1] - buffer, *size, collision_map, buffer):
                return x, bed_pos[1] - size[1] - 1 - buffer, "S"

        # The bottom edge
        for x in range(bed_pos[0] + bed_pos[0] // 4, bed_pos[0] + bed_siz[0] - bed_siz[0] // 4):
            if is_valid_position(x, bed_pos[1] + bed_siz[1] + 1 + buffer, *size, collision_map, buffer):
                return x, bed_pos[1] + bed_siz[1] + 1 + buffer, "N"

        # The left edge
        for y in range(bed_pos[1] + bed_siz[1] // 4, bed_pos[1] + bed_siz[1] - bed_siz[1] // 4):
            if is_valid_position(bed_pos[0] - size[1], y, *size[::-1], collision_map, buffer):  # it is rotated
                return bed_pos[0] - size[1] - 1 - buffer, y, "E"

        # The right edge
        for y in range(bed_pos[1] + bed_siz[1] // 4, bed_pos[1] + bed_siz[1] - bed_siz[1] // 4):
            if is_valid_position(bed_pos[0] + bed_siz[0] + 1, y, *size[::-1], collision_map, buffer):  # it is rotated
                return bed_pos[0] + bed_siz[0] + 1 + buffer, y, "W"

    return None, None, None

def place_front(size, bed_pos, bed_siz, bed_ang, buffer, collision_map):
    """Place the furniture in front of an anchor furniture, with it facing the anchor furniture"""
    if bed_ang == "N":
        # The top edge
        for x in range(bed_pos[0] + bed_pos[0] // 10, bed_pos[0] + bed_siz[0] - bed_siz[0] // 10):
            if is_valid_position(x, bed_pos[1] - size[1] - buffer, *size, collision_map, buffer):
                return x, bed_pos[1] - size[1] - buffer, "S"
    elif bed_ang == "S":
        # The bottom edge
        for x in range(bed_pos[0], bed_pos[0] + bed_siz[0]):
            if is_valid_position(x, bed_pos[1] + bed_siz[1] + 1 + buffer, *size, collision_map, buffer):
                return x, bed_pos[1] + bed_siz[1] + 1 + buffer, "N"
    elif bed_ang == "E":
        # The right edge
        for y in range(bed_pos[1] + bed_siz[1] // 10, bed_pos[1] + bed_siz[1] - bed_siz[1] // 10):
            if is_valid_position(bed_pos[0] + bed_siz[0] + 1 + buffer, y, *size[::-1], collision_map, buffer):  # it is rotated
                return bed_pos[0] + bed_siz[0] + 1 + buffer, y, "W"
    else:
        # The left edge
        for y in range(bed_pos[1] + bed_siz[1] // 10, bed_pos[1] + bed_siz[1] - bed_siz[1] // 10):
            if is_valid_position(bed_pos[0] - size[1] - buffer, y, *size[::-1], collision_map, buffer):  # it is rotated
                return bed_pos[0] - size[1] - buffer, y, "E"

    return None, None, None

def place_next_wall(bbox, centers, size, collision_map):
    """Place the furniture next to a wall in the room, with its side against the wall"""
    def is_adjacent_wall(x, y):
        return collision_map[x, y + 1] == 0 or collision_map[x + 1, y] == 0 or \
            collision_map[x, y - 1] == 0 or collision_map[x - 1, y] == 0

    for (x1, y1, x2, y2) in bbox:
        buffer_x, buffer_y = (x2 - x1) // 9, (y2 - y1) // 9
        # search the left edge
        for y in range(y1 + buffer_y, y2 - size[1] - buffer_y):
            if not is_adjacent_wall(x1, y):
                break
            if is_valid_position(x1, y, *size, collision_map):  # the furniture is rotated
                return x1 + 5, y, "S"  # add a buffer to avoid the wall
        # search the right edge
        for y in range(y1 + buffer_y, y2 - size[1] - buffer_y):
            if not is_adjacent_wall(x2, y):
                break
            if is_valid_position(x2 - size[0], y, *size, collision_map):  # the furniture is rotated
                return x2 - size[0] - 5, y, "S"
        # search the top edge
        for x in range(x1 + buffer_x, x2 - size[1] - buffer_x):
            if not is_adjacent_wall(x, y1):
                break
            if is_valid_position(x, y1, *size[::-1], collision_map):  # the furniture is rotated
                return x, y1 + 5, "E"
        # search the bottom edge
        for x in range(x1 + buffer_x, x2 - size[1] - buffer_x):
            if not is_adjacent_wall(x, y2):
                break
            if is_valid_position(x, y2 - size[0], *size[::-1], collision_map):  # the furniture is rotated
                return x, y2 - size[0] - 5, "E"
            
    return None, None, None

def place_next(size, bed_pos, bed_siz, bed_ang, buffer, collision_map):
    """Place the anchor furniture next to another anchor furniture in the room"""
    if bed_ang == "S":  # the bed is facing the bottom
        if is_valid_position(bed_pos[0] - size[0] - buffer, bed_pos[1], *size, collision_map, buffer):
            return bed_pos[0] - size[0] - buffer, bed_pos[1], "S"
        if is_valid_position(bed_pos[0] + bed_siz[0] + 1 + buffer, bed_pos[1], *size, collision_map, buffer):
            return bed_pos[0] + bed_siz[0] + 1 + buffer, bed_pos[1], "S"
    elif bed_ang == "N":  # the bed is facing the top
        if is_valid_position(bed_pos[0] - size[0] - buffer, bed_pos[1] + bed_siz[1] - size[1], *size, collision_map, buffer):
            return bed_pos[0] - size[0] - buffer, bed_pos[1] + bed_siz[1] - size[1], "N"
        if is_valid_position(bed_pos[0] + bed_siz[0] + 1 + buffer, bed_pos[1] + bed_siz[1] - size[1], *size, collision_map, buffer):
            return bed_pos[0] + bed_siz[0] + 1 + buffer, bed_pos[1] + bed_siz[1] - size[1], "N"
    elif bed_ang == "E":  # the bed is facing the right
        if is_valid_position(bed_pos[0], bed_pos[1] - size[0] - buffer, *size[::-1], collision_map, buffer):
            return bed_pos[0], bed_pos[1] - size[0] - buffer, "E"
        if is_valid_position(bed_pos[0], bed_pos[1] + bed_siz[1] + 1 + buffer, *size[::-1], collision_map, buffer):
            return bed_pos[0], bed_pos[1] + bed_siz[1] + 1 + buffer, "E"
    else:  # the bed is facing the left
        if is_valid_position(bed_pos[0] + bed_siz[0] - size[1], bed_pos[1] - size[0] - buffer, *size[::-1], collision_map, buffer):
            return bed_pos[0] + bed_siz[0] - size[1], bed_pos[1] - size[0] - buffer, "W"
        if is_valid_position(bed_pos[0] + bed_siz[0] - size[1], bed_pos[1] + bed_siz[1] + 1 + buffer, *size[::-1], collision_map, buffer):
            return bed_pos[0] + bed_siz[0] - size[1], bed_pos[1] + bed_siz[1] + 1 + buffer, "W"

    return None, None, None

def place_ceiling(centers, size, height):  # for lamps
    """Place the furniture on the ceiling at a center position"""
    return centers[0][0] - size[0] // 2, centers[0][1] - size[1] // 2, height
