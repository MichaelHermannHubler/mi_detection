from SpecialistNeuralNetwork import FinetunedCNN
from ds_PTBXL import PTBXLDataset

from torchview import draw_graph
from torchviz import make_dot
import graphviz

ds = PTBXLDataset()

x = ds[0][0].view(1, 12, 5000)

model = FinetunedCNN()
y = model(x)

dot_graph = make_dot(y['NORM'], params=dict(model.named_parameters())).render("cnn_torchviz", format="png")

# model_graph = draw_graph(FinetunedCNN(), input_size=(12, 5000), expand_nested=True)
# model_graph.visual_graph