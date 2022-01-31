from pytest import mark
from spacekit.generator.draw import DrawMosaics


# @mark.paramatrize(["input_path","fname","pattern","crpt"], [
#     ("reg","reg.csv","",0),
#     ("reg","reg.csv","*",0),
#     #("synth","synth.csv","",1),
#     #("synth","synth.csv","*",1)
# ])
# def test_autoload_datasets(input_path, fname, pattern, crpt):
#     draw = DrawMosaics(input_path, output_path="tmp/img", fname=fname, pattern=pattern, crpt=crpt)
#     assert len(draw.datasets) > 0

# @mark.paramatrize(["input_path","fname","pattern","crpt"], [
#     ("reg","reg.csv","",0),
#     ("reg","reg.csv","*",0),
#     #("synth","synth.csv","",1),
#     #("synth","synth.csv","*",1)
# ])
# def test_image_generator(input_path, fname, pattern, crpt):
#     draw = DrawMosaics(input_path, output_path="tmp/img", fname=fname, pattern=pattern, crpt=crpt)
#     draw.generate_total_images()
#     assert True


# @mark.paramatrize(["input_path","fname","crpt"], [
#     ("reg","reg.csv",0),
#     #("synth","synth.csv",1),
# ])
# def test_load_from_file(input_path, fname, crpt):
#     draw = DrawMosaics(input_path, output_path="tmp/img", fname=fname, pattern="", crpt=crpt)
#     draw.load_from_file()
#     assert len(draw.datasets) > 0

# @mark.paramatrize(["input_path","pattern","crpt"], [
#     ("reg","*",0),
#     #("synth","*",1),
# ])
# def test_local_search(input_path, pattern, crpt):
#     draw = DrawMosaics(input_path, output_path="tmp/img", fname=None, pattern=pattern, crpt=crpt)
#     draw.local_search()
#     assert len(draw.datasets) > 0

# @mark.paramatrize(["input_path","visit","crpt"], [
#     ("reg","ibl738",0),
#     #("synth","ibl738",1),
# ])
# def test_single_visit(input_path, visit, crpt):
#     draw = DrawMosaics(input_path, output_path="tmp/img", fname=None, pattern="", visit=visit, crpt=crpt)
#     assert len(draw.datasets) > 0
