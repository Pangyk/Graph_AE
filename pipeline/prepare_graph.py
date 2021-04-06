from utils.CustomDataSet import SelectGraph
import os

SelectGraph.data_name = "Shana"
data_set = SelectGraph(os.path.join('data', SelectGraph.data_name))
