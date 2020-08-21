import yaml
from types import SimpleNamespace 

class Read_Yaml():
  def __init__(self, filename):
    with open(filename, 'r') as file:
      self.params = SimpleNamespace(**yaml.load(file, Loader = yaml.FullLoader))
