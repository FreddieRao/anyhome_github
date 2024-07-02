from floorplan.floorplan_generator import FloorplanGenerator
from layout.layout_generator import LayoutGenerator
import credentials

# Enter your prompt here
prompt = "A 1B1B haunted house."

# Create a floorplan generator, the floor plan mesh is stored at ./output, and the fp visualizations are at ./floorplan/output
floorplanGenerator = FloorplanGenerator(prompt)
house_v, house_f, border_map_no_doors, room_name_dict, boxes, centers = floorplanGenerator.generate_house_mesh(edit=True)  # Set edit to True to allow multiple-round language-guided editing

# Create a room layout generato
layoutGenerator = LayoutGenerator(prompt, house_v, house_f, border_map_no_doors, room_name_dict, boxes, centers)
layoutGenerator.generate_room_objects(edit=True)  # Set edit to True to allow multiple-round language-guided editing