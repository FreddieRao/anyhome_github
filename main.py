from floorplan.floorplan_generator import FloorplanGenerator
from layout.layout_generator import LayoutGenerator
from opt import get_default_parser
import credentials

def main(args):
    # Enter your prompt here
    prompt = "A 1B1B haunted house."

    # Create a floorplan generator, the floor plan mesh is stored at ./output, and the fp visualizations are at ./floorplan/output
    floorplanGenerator = FloorplanGenerator(args, prompt)
    house_v, house_f, border_map_no_doors, room_name_dict, boxes, centers = floorplanGenerator.generate_house_mesh(edit=True)  # Set edit to True to allow multiple-round language-guided editing

    # Create a room layout generator
    layoutGenerator = LayoutGenerator(args, prompt, house_v, house_f, border_map_no_doors, room_name_dict, boxes, centers)
    layoutGenerator.generate_room_objects(edit=True)  # Set edit to True to allow multiple-round language-guided editing

if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    main(args)