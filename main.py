from floorplan.floorplanGenerator import FloorplanGenerator
import credentials

# Enter your prompt here
prompt = "A large house with only a long hallway connected to three bedrooms, no other rooms."

# Create a floorplan generator, the floor plan mesh is stored at ./output, and the fp visualizations are at ./floorplan/output
floorplanGenerator = FloorplanGenerator(prompt)
house_v, house_f = floorplanGenerator.generate_house_mesh(edit=True)  # Set edit to True to allow multiple-round language-guided editing