from roboflow import Roboflow

rf = Roboflow(api_key="rf_bbBGUkdzGDNhO5HwcD3UibV08iA3")
project = rf.workspace("meyve-eyp6i").project("wisdom-teeth-nbnzt-mnmd0")
version = project.version(2)

dataset = version.download("yolov8")

