import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import openai
import json

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from floorplan.houseganpp.models import Generator
from floorplan.houseganpp.utils import _init_input, draw_masks, draw_graph
import floorplan.utils as utils


class FloorplanGenerator:
    def __init__(self, description, output_dir="floorplan/output", houseganpp_weight="floorplan/houseganpp/checkpoints/pretrained.pth"):
        self.description = description
        self.output_dir = output_dir
        self.houseganpp_weight = houseganpp_weight

    def generate_house_mesh(self):
        # Generate graph from descriptoin
        nds, eds, room_name_dict, room_list = self.generate_bubble_diagram(self.description)

        # Control whether the generated map is valid
        status = False
        while not status:
            # Generate masks for house layout
            masks = self.generate_layout_masks(nds, eds)
            # Check whether the resulted floor plan is valid and generate a single integer map for the house layout
            img_size = 256
            status, result_map, result_map_no_doors, result_masks = utils.check_post_processing(nds, eds, masks, list(range(len(room_name_dict), len(room_list))), img_size)

        # Change the color map into a border map
        border_map_no_doors, start_coors = utils.generate_border_map_no_doors(result_map_no_doors, result_masks, list(range(len(room_name_dict), len(room_list))), nds, eds)
        # Define the custom colors
        custom_colors = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), *plt.cm.tab20b(np.linspace(0, 1, 19))]
        # Create the custom colormap
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', custom_colors)
        plt.imshow(border_map_no_doors, cmap=custom_cmap)
        # Decompose each room into regular rectangles and obtain the centers
        boxes, centers = utils.get_room_boundaries(border_map_no_doors, len(room_list)-1, start_coors)
        utils.visualize_map_with_centers(border_map_no_doors, boxes, centers)
        # Generate house mesh
        segments = utils.find_segments(border_map_no_doors)
        house_v, house_f = utils.write_to_obj(segments, border_map_no_doors)

        return house_v, house_f
    
    @staticmethod
    def _infer(graph, model, prev_state=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # configure input to the network
        z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state)
        # run inference model
        with torch.no_grad():
            masks = model(z.to(device), given_masks_in.to(device), given_nds.to(device), given_eds.to(device))
            masks = masks.detach().cpu().numpy()
        return masks

    def generate_bubble_diagram(self, description):
        # Generate graph from descriptoin using GPT-4
        context_msg = '''1. You will be provided a text description of a house delimited by "" which includes the house's
        style (e.g., gothic) and type (e.g., a castle); there might be additional information provided. 2. You need to
        generate four things: 1) the complete list of rooms in the house, 2) the modified list of rooms in the house,
        3) the connection of each room, and 4) what rooms the front doors are at. 3. For 1) the complete list of rooms in
        the house, you need to generate a list of rooms one by one with an index at the end, and repeat for those rooms
        that occur more than once but with incrementing indices. For example, if there are two bedrooms and one dining
        room, the generated response should be "[bedroom1, bedroom2, dining room1]". The list of rooms should be diverse,
        with different room types and room functionalities. Examples of rooms generated at this step include "library",
        "wine_celler" and "entrance". 4. For 2) the modified list of rooms in the house, you need to transform each room
        generated in step 1) based on their style and functionality so that it is from the list: "kitchen", “storage”,
        "bathroom", "study_room", "balcony", "living_room", “bedroom”, "entrance", "dining_room" and "unknown". For
        example, "dungeon" has a similar property to "storage" so you just transform "dungeon1" to "storage1",
        if there are no existing storage rooms. Else, you need to transform "dungeon1" to "storage2". If the room is so
        different from any existing room in the accepted list, just transform the room into "unknown". You should return
        the answer for 2) in the form of a list as well. Make sure that each room in the modified room list corresponds
        to each room in the complete room list that has the same index. 5. For 3) the connection of each room in its original name,
        please generate, for each room, the rooms that are connected to the room in the form of tuples. For example,
        if dining hall1 is connected to bedroom1 and bedroom2, but bedroom1 and bedroom2 are not interconnected,
        the generated response should be "[[dining_room1, bedroom1], [dining_room1, bedroom2]]". 6. For 4) what rooms the
        front doors are, please generate the rooms that contain a front door, the door acting as the only entrance to the
        house. For example, if the front door is located in bedroom1, please generate "[bedroom1]". 7. Return your answer
        in the form of a JSON file, with field names "complete_room_list", "modified_room_list", "connection",
        and "front_door" corresponding to each of the questions 1) to 4) above. You only need to return this JSON file, and no other things needed.'''

        client = openai.OpenAI()
        raw_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": context_msg},
                {"role": "user", "content": self.description}
            ],
            temperature=0,
            max_tokens=2048
        )

        response_str = raw_response.choices[0].message.content
        raw_response = response_str.replace("\n", "").replace(" ", "")
        response = json.loads(raw_response)
        print("House Bubble Diagram:\n", response)

        complete_room_list = response["complete_room_list"]
        room_list = response["modified_room_list"]
        room_name_dict = dict(zip(complete_room_list, room_list))
        real_nodes_len = len(room_list)

        connection = response["connection"]
        original_connection = connection.copy()
        # change to the original room names to the HouseGAN++ room names
        connection = [[room_name_dict[pair[0]], room_name_dict[pair[1]]] for pair in connection]
        raw_connection = connection.copy()
        front_door = response["front_door"]
        front_door = [room_name_dict[door] for door in front_door]

        # add interior doors nodes to the room_list
        interior_door_list = ["interior_door" + str(i) for i in range(1, len(connection) + 1)]
        room_list.extend(interior_door_list)
        front_door_list = ["front_door" + str(i) for i in range(1, len(front_door) + 1)]
        room_list.extend(front_door_list)
        connection = [(pair[0], "interior_door" + str(i), pair[1]) for i, pair in enumerate(connection, 1)]

        # the list of room types and their corresponding indices
        type_list = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6,
                    "dining_room": 7, "study_room": 8, "storage": 10, "front_door": 15, "unknown": 16, "interior_door": 17}

        # turn the room types into indices
        room_list_indices = [type_list[room[:-1]] if room[:-1] in type_list else type_list["unknown"] for room in room_list]
        # add the index to each room for better formatting the edges of the bubble graph
        room_dict = {room: i for i, room in enumerate(room_list)}
        connection = [(room_dict[pair[0]], room_dict[pair[1]], room_dict[pair[2]]) for pair in connection]

        # generate the edges as triples
        triples = []
        for i in range(len(room_list)):
            for j in range(len(room_list)):
                if j > i:
                    triples.append((i, -1, j))

        for i in range(real_nodes_len):
            for j in range(real_nodes_len):
                if j > i:
                    is_adjacent = any([True for e_map in connection if (i in e_map) and (j in e_map)])
                    # print(i, j, is_adjacent)
                    if is_adjacent:
                        edge = [pair[1] for pair in connection if (pair[0] == i and pair[2] == j) or
                                (pair[0] == j and pair[2] == i)]
                        triples.remove((i, -1, edge[0]))
                        triples.remove((j, -1, edge[0]))
                        triples.remove((i, -1, j))
                        triples.append((i, 1, edge[0]))
                        triples.append((j, 1, edge[0]))
                        triples.append((i, 1, j))

        # add the front door edges
        for i, room in enumerate(front_door, 1):
            triples.remove((room_dict[room], -1, room_dict["front_door" + str(i)]))
            triples.append((room_dict[room], 1, room_dict["front_door" + str(i)]))

        organized_triples = sorted(triples, key=lambda x: (x[0], x[2]))

        graph_nodes = utils.one_hot_embedding(room_list_indices)[:, 1:]
        graph_nodes = torch.FloatTensor(graph_nodes)
        graph_edges = torch.LongTensor(organized_triples)

        return graph_nodes, graph_edges, room_name_dict, room_list

    def generate_layout_masks(self, nds, eds):
        # Create output dir
        os.makedirs(self.output_dir, exist_ok=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize generator and discriminator
        model = Generator()
        model.load_state_dict(torch.load(self.houseganpp_weight, map_location=torch.device(device)), strict=True)
        model = model.eval()

        # Initialize variables
        if torch.cuda.is_available():
            model.cuda()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor # Optimizers

        # Draw real graph
        real_nodes = np.where(nds.detach().cpu()==1)[-1]
        graph = [nds, eds]
        true_graph_obj, graph_im = draw_graph([real_nodes, eds.detach().cpu().numpy()])
        graph_im.save('./{}/input_graph.png'.format(self.output_dir)) # save graph

        # Add room types incrementally
        _types = sorted(list(set(real_nodes)))
        selected_types = [_types[:k+1] for k in range(10)]
        os.makedirs('./{}/'.format(self.output_dir), exist_ok=True)
        _round = 0

        # Initialize layout
        state = {'masks': None, 'fixed_nodes': []}
        masks = self._infer(graph, model, state)
        im0 = draw_masks(masks.copy(), real_nodes)
        im0 = torch.tensor(np.array(im0).transpose((2, 0, 1)))/255.0
        save_image(im0, './{}/init_fp.png'.format(self.output_dir), nrow=1, normalize=False) # visualize init image

        # Generate per room type
        for _iter, _types in enumerate(selected_types):
            _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types]) \
                if len(_types) > 0 else np.array([])
            state = {'masks': masks, 'fixed_nodes': _fixed_nds}
            masks = self._infer(graph, model, state)

        # Save final floorplans
        imk = draw_masks(masks.copy(), real_nodes)
        imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0
        save_image(imk, './{}/final_fp.png'.format(self.output_dir), nrow=1, normalize=False)

        return masks
