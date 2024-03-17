from floorplan.floorplanGenerator import FloorplanGenerator
import credentials

# Enter your prompt here
prompt = "a 1B1B house with a garage and a kitchen."

# Create a floorplan generator, the floor plan mesh is stored at ./output, and the fp visualizations are at ./floorplan/output
floorplanGenerator = FloorplanGenerator(prompt)
house_v, house_f = floorplanGenerator.generate_house_mesh()