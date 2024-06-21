import numpy as np
import openai
import json
from termcolor import colored

class LayoutGenerator:
    def __init__(self, description, house_v, house_f, border_map_no_doors, room_name_dict, boxes, centers):
        self.description = description
        self.house_v = house_v
        self.house_f = house_f
        self.border_map_no_doors = border_map_no_doors
        self.room_name_dict = room_name_dict
        self.boxes = boxes
        self.centers = centers
    
    def generate_room_objects(self, edit=False):
        room_graph_list = {}  # Store the room graph for each room
        poss, sizs, angs = [], [], []  # Store the positions, sizes, and angles of the objects
        for i, room in enumerate(list(self.room_name_dict.keys())):
            print(colored(f"Generating furniture for {room}...", "grey"))
            # Obtain the room area
            box = self.boxes[i]  # Obtain the bounding boxes for this room
            areas = [(x2-x1) * (y2-y1) for (x1, y1, x2, y2) in box]
            total_area = sum(areas)
            # Generate the furniture diagram
            room_graph = self.generate_furniture_diagram(total_area, room)
            while edit:  # Allow multiple-round language-guided editing
                edit_description = input(colored("Enter the description of the changes you want to make to the layout graph (Enter q to stop editing): ", "green"))
                if edit_description == "q":
                    break
                room_graph = self.generate_furniture_diagram(total_area, room, is_edit=True, edit_description=edit_description, edit_graph=room_graph)
    
    def generate_furniture_diagram(self, room_area, room_type, is_edit=False, edit_description=None, edit_graph=None):
        # Generate a graph from description using GPT-4
        context_msg = """
        Task: You are an awesome 3D Scene Designer. Design a 3D indoor scene for a {} located within a {}. Ensure that the design fits within an area of {} square meters. Provide the details as a scene graph, structured in a JSON format.

        Requirements:
        1. Furniture List:
            - Enumerate the furniture in the {}. If an item appears multiple times, append an index. Remember that the furniture should be from a {}. Generate as many furniture as you can, but maintain the style of the room. For example, you shouldn't just generate one set of table and chair for an office room.
            - Use only from this predefined list: children_cabinet, nightstand, bookcase, wardrobe, coffee_table, corner_table, side_cabinet, wine_cabinet, tv_stand, drawer_chest, shelf, round_end_table, double_bed, queen_bed, king_bed, bunk_bed, bed_frame, single_bed, kids_bed, dining_chair, lounge_chair, office_chair, dressing_chair, classic_chinese_chair, barstool, dressing_table, dining_table, desk, three_seat_sofa, armchair, loveseat_sofa, l_shaped_sofa, lazy_sofa, chaise_longue_sofa, stool, kitchen_cabinet, toilet, bathtub, sink.
            - Example: If there are two lazy sofas and an armchair in the room, list them as: “[lazy_sofa1, lazy_sofa2, armchair1]”.
        2. Furniture Description:
            - For each furniture piece, provide a description of its aesthetic shape and structure considering the house and room's styles.
            - Example: "A three-seated sofa with two armrests at the sides."
            - Return format: {{<furniture_name>: <description>}}
        3. Furniture Size:
            - For each single furniture piece (even if the furniture is repeated), provide its dimensions (length, width, height) in pixels, keeping in mind the {} square meters room area.
            - Return format: {{<furniture_name>: [length(meters), width(meters), height(meters)]}}
        4. Furniture Groups & Placement Rules:
            - Separate the furniture into groups that will exist together in the scene, such as a bed and two nightstands. For those furniture that is unrelated with each other, put them in different groups. Put the important furniture groups at first when returning your answer. For example, put the funiture group containing the kitchen cabinet at first in a kitchen, and that containing the sofa, tv stand and coffee table at first in a living room.
            - For each group, you should decide the anchor furniture. For example, the anchor for a bed and two nightstands is the bed.
            - Then, you should decide on the placement rule for each furniture item.
                - For anchor furnitures, you can use (1) "place_center" which places the anchor furniture at the center of available spaces in the room, (2) "place_wall" which places the anchor with its back on the wall, (3) "place_corner" which places the anchor at the corner, (4) "place_next_wall" which places the anchor next to a wall, and (5) “place_next” which places the anchor furniture beside the anchor other group (which is useful for generating scenes with furniture arranged in rows and columns).
                - For other furnitures in the group, you can use (1) "place_front(x)" which places the furniture in front of the anchor with a buffer distance of x meters, like a TV stand before a coffee table, (2) "place_beside(x)" which places the furniture beside the anchor with a buffer distance of x meters, like a nightstand beside a bed, and (3) "place_around(x)" which places the furniture around the anchor with a buffer distance of x meters, like placing four chairs around a dining table.
            - Return format: [[[<anchor_furniture_name>, <anchor_placement_rule>], [<furniture_name>, <placement_rule>], ...], ...]

        Output: Provide the information in a valid JSON structure with no spaces. I'll give you 100 bucks if you help me design a perfect scene and return it in the right format:
        {{
            "furniture_list": [...],
            "furniture_descriptions": {{...}},
            "furniture_sizes": {{...}},
            "furniture_groups_and_placement_rules": [...]
        }}
        """
        context_msg = context_msg.format(room_type, self.description, room_area, room_type, room_type, room_area)
        
        edit_context_msg = """
        Task: You are an awesome 3D Scene Designer serving a customer. You are given a 3D indoor scene graph for a {} located within a {}. The design fits within an area of {} square meters. The scene graph is generated with the four requirements below:

        Requirements:
        1. Furniture List:
            - Enumerate the furniture in the {}. If an item appears multiple times, append an index. Remember that the furniture should be from a {}. Generate as many furniture as you can, but maintain the style of the room. For example, you shouldn't just generate one set of table and chair for an office room.
            - Use only from this predefined list: children_cabinet, nightstand, bookcase, wardrobe, coffee_table, corner_table, side_cabinet, wine_cabinet, tv_stand, drawer_chest, shelf, round_end_table, double_bed, queen_bed, king_bed, bunk_bed, bed_frame, single_bed, kids_bed, dining_chair, lounge_chair, office_chair, dressing_chair, classic_chinese_chair, barstool, dressing_table, dining_table, desk, three_seat_sofa, armchair, loveseat_sofa, l_shaped_sofa, lazy_sofa, chaise_longue_sofa, stool, kitchen_cabinet, toilet, bathtub, sink.
            - Example: If there are two lazy sofas and an armchair in the room, list them as: “[lazy_sofa1, lazy_sofa2, armchair1]”.
        2. Furniture Description:
            - For each furniture piece, provide a description of its aesthetic shape and structure considering the house and room's styles.
            - Example: "A three-seated sofa with two armrests at the sides."
            - Return format: {{<furniture_name>: <description>}}
        3. Furniture Size:
            - For each single furniture piece (even if the furniture is repeated), provide its dimensions (length, width, height) in pixels, keeping in mind the {} square meters room area.
            - Return format: {{<furniture_name>: [length(meters), width(meters), height(meters)]}}
        4. Furniture Groups & Placement Rules:
            - Separate the furniture into groups that will exist together in the scene, such as a bed and two nightstands. For those furniture that is unrelated with each other, put them in different groups. Put the important furniture groups at first when returning your answer. For example, put the funiture group containing the kitchen cabinet at first in a kitchen, and that containing the sofa, tv stand and coffee table at first in a living room.
            - For each group, you should decide the anchor furniture. For example, the anchor for a bed and two nightstands is the bed.
            - Then, you should decide on the placement rule for each furniture item.
                - For anchor furnitures, you can use (1) "place_center" which places the anchor furniture at the center of available spaces in the room, (2) "place_wall" which places the anchor with its back on the wall, (3) "place_corner" which places the anchor at the corner, (4) "place_next_wall" which places the anchor next to a wall, and (5) “place_next” which places the anchor furniture beside the anchor other group (which is useful for generating scenes with furniture arranged in rows and columns).
                - For other furnitures in the group, you can use (1) "place_front(x)" which places the furniture in front of the anchor with a buffer distance of x meters, like a TV stand before a coffee table, (2) "place_beside(x)" which places the furniture beside the anchor with a buffer distance of x meters, like a nightstand beside a bed, and (3) "place_around(x)" which places the furniture around the anchor with a buffer distance of x meters, like placing four chairs around a dining table.
            - Return format: [[[<anchor_furniture_name>, <anchor_placement_rule>], [<furniture_name>, <placement_rule>], ...], ...]

        Generated Data of the Scene Graph of the {}:
        {}

        Now, the customer wants to edit the scene graph. They have provided the following description of the changes they want to make:
        {}

        Task: Could you please update the floorplan based on the customer's request and provide the updated data strictly following the same requirements above? If the customer's request collides with the requirement above, followed the requirements above. For example, if the customer wants to use "place_around(x)" as the placement rule for an anchor furniture, you should not follow it and generate the placement rule based on the requirements above.

        Output: Provide the information in a valid JSON structure with no spaces. I'll give you 100 bucks if you help me design a perfect scene and return it in the right format:
        {{
            "furniture_list": [...],
            "furniture_descriptions": {{...}},
            "furniture_sizes": {{...}},
            "furniture_groups_and_placement_rules": [...]
        }}
        """
        edit_context_msg = edit_context_msg.format(room_type, self.description, room_area, room_type, room_type, room_area, room_type, edit_graph, edit_description)

        client = openai.OpenAI()
        raw_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": context_msg if not is_edit else edit_context_msg},
            ],
            temperature=0.7,
            top_p=0.7,
            max_tokens=4096
        )

        response_str = raw_response.choices[0].message.content
        raw_response = response_str.replace("\n", "").strip()
        response = json.loads(raw_response)

        print(colored(f"{room_type} Furniture Graph", "green"))
        print('\n'.join([f'{colored(k, "blue")}: {v}' for k, v in response.items()]))

        return response