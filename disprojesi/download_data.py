from roboflow import Roboflow

rf = Roboflow(api_key="8NSTWSAx6tJvS8zFRwac")
project = rf.workspace("meyve-eyp6i").project("wisdom-teeth-nbnzt-mnmd0")
version = project.version(2)
dataset = version.download("yolov8")