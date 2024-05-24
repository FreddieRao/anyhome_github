# AnyHome Github
This is the official Github repo for [*AnyHome: Open-Vocabulary Generation of Structured and Textured 3D Homes*](https://arxiv.org/abs/2312.06644).

We are planning to gradually release the code for each module. Please stay tuned!

## Instructions
To run the codebase, you first need to clone the repository:
```bash
git clone https://github.com/FreddieRao/anyhome_github.git
```

Then, install the dependencies at `requirements.txt`:
```bash
pip install -r requirements.txt
```

Also, you should enter you OPENAI API key in the `credentials.py` file:
```python
os.environ["OPENAI_API_KEY"] = "YOU API KEY"
```
Last, you could simply run the code by calling:
```bash
python main.py
```
You can change the prompt for generation in the `main.py` file:
```python
# Enter your prompt here
prompt = "a 1B1B house with a garage and a kitchen."
```

## Modules
- [x] House Floorplan Generation (Prompt, Generator, Mesh Creation)
- [ ] Room Layout Generation (Prompt, Placement Algorithms, Object Retrieval)
- [ ] Trajectory Generation
- [ ] SDS Optimization
- [ ] Texture Generation
- [ ] Editability