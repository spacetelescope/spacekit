from spacekit.builder.architect import Builder
import os


# build from spacekit package trained_networks
builder_wc = Builder(blueprint="wall_reg")
builder_wc.load_saved_model(arch="calmodels")

# build from local filepath
model_path = "data/2022-02-14-1644848448/models"

wc_model_path = os.path.join(model_path, "wall_reg")
builder_wc = Builder(blueprint="wall_reg", model_path=wc_model_path)
builder_wc.load_saved_model()

inputs = []

# prep inputs

# do inference

# write to ddb
