from utils.CustomDataSet import SelectGraph
import os


SelectGraph.data_name = "Shana"
SelectGraph.thresh = 100
SelectGraph.direction = 0
data_set = SelectGraph(os.path.join('data', SelectGraph.data_name))
