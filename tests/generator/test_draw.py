from pytest import mark
from spacekit.generator.draw import DrawMosaics
import os


@mark.generator
@mark.draw
def test_draw_from_pattern(single_visit_path, img_outpath, draw_mosaic_pattern):
    mos = DrawMosaics(
        single_visit_path,
        output_path=img_outpath,
        pattern=draw_mosaic_pattern,
    )
    assert mos.datasets[0] == "ibl738"
    mos.local_search()
    assert len(mos.datasets) > 0


@mark.generator
@mark.draw
def test_draw_from_fname(single_visit_path, img_outpath, draw_mosaic_fname):
    mos = DrawMosaics(
        single_visit_path,
        output_path=img_outpath,
        fname=draw_mosaic_fname,
    )
    assert mos.datasets[0] == "ibl738"
    mos.load_from_file()
    assert len(mos.datasets) > 0


@mark.generator
@mark.draw
def test_draw_from_visit(single_visit_path, img_outpath):
    mos = DrawMosaics(
        single_visit_path,
        output_path=img_outpath,
        visit="ibl738",
    )
    assert mos.datasets[0] == "ibl738"


@mark.generator
@mark.draw
def test_draw_from_priority(single_visit_path, img_outpath, draw_mosaic_fname):
    mos = DrawMosaics(
        single_visit_path,
        output_path=img_outpath,
        fname=None,
        pattern="",
        visit="ibl738",
    )
    # defaults to visit
    assert mos.datasets == os.listdir(single_visit_path) == [mos.visit]
    # defaults to pattern
    mos.visit, mos.pattern = None, "ibl*"
    data_from_pattern = mos.local_search()
    assert len(data_from_pattern) > 0
    # defaults to file
    mos.pattern, mos.fname = "", draw_mosaic_fname
    data_from_file = mos.load_from_file()
    assert len(data_from_file) > 0


@mark.generator
@mark.draw
@mark.parametrize("P, S, G", [(0, 0, 0), (1, 1, 0), (0, 0, 1)])
def test_draw_total_images(single_visit_path, img_outpath, P, S, G):
    mos = DrawMosaics(single_visit_path, output_path=img_outpath)
    mos.draw_total_images(mos.datasets[0], P=P, S=S, G=G)
    pfx = "hst_12286_38_wfc3_ir_total_ibl738"
    img_dir = os.path.join(img_outpath, pfx)
    assert len(os.listdir(img_dir)) == 1
    if G == 1:
        sfx = "_gaia.png"
    elif S == 1 and P == 1:
        sfx = "_source.png"
    else:
        sfx = ".png"
    name = f"{pfx}{sfx}"
    img_path = os.path.join(img_dir, name)
    assert os.path.exists(img_path)


@mark.generator
@mark.draw
def test_draw_generator(single_visit_path, img_outpath):
    mos = DrawMosaics(single_visit_path, output_path=img_outpath)
    mos.generate_total_images()
    pfx = "hst_12286_38_wfc3_ir_total_ibl738"
    img_dir = os.path.join(img_outpath, pfx)
    assert len(os.listdir(img_dir)) == 3
    image_files = []
    fmt = "png"
    for i in ["hst_12286_38_wfc3_ir_total_ibl738"]:
        img_frames = (
            f"{img_outpath}/{i}/{i}.{fmt}",
            f"{img_outpath}/{i}/{i}_source.{fmt}",
            f"{img_outpath}/{i}/{i}_gaia.{fmt}",
        )
        if os.path.exists(img_frames[0]):
            image_files.append(img_frames)
    assert len(image_files) == 1
    assert len(image_files[0]) == 3
