import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits, ascii
from astropy.visualization import ImageNormalize, ZScaleInterval
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from spacekit.analyzer.track import stopwatch


def point_flag_color(x):
    if x <= 1:
        return "red", "Flag <= 1"
    elif x <= 5:
        return "green", "2 <= Flag <= 5"
    else:
        return None, None  # 'yellow', 'Flag > 5'


def segment_flag_color(x):
    if x <= 1:
        return "blue", "Flag <= 1"
    elif x <= 5:
        return "green", "2 <= Flag <= 5"
    else:
        return None, None  # 'yellow', 'Flag > 5'


def draw_catalogs(cfile, catalog):
    cat, fcolor_, fcolor = None, None, None
    if os.path.exists(cfile):
        # cat = Table.read(catfile, format='ascii.ecsv')
        cat = ascii.read(cfile).to_pandas()
    else:
        cat = ""
    if len(cat) > 0:
        if "Flags" in cat.columns:
            flagcols = cat["Flags"]
        else:
            flagcols = [c for c in cat.columns if "Flags" in c]
        if len(flagcols) > 0:
            flags = (
                cat.loc[:, flagcols]
                .fillna(100, axis=0, inplace=False)
                .apply(min, axis=1)
            )
            if catalog == "point":
                fcolor_ = flags.apply(point_flag_color)
            elif catalog == "segment":
                fcolor_ = flags.apply(segment_flag_color)
            fcolor = fcolor_.apply(lambda x: x[0]).values
    return cat, fcolor_, fcolor


def create_image_name(name, dataset, P, S, G, crpt, outpath):
    if P == 1 and S == 1:
        catstr = "_source"
    elif P == 1 and S == 0:
        catstr = "_point"
    elif P == 0 and S == 1:
        catstr = "_segment"
    elif G == 1:
        catstr = "_gaia"
    else:
        catstr = ""
    if crpt:
        sfx = "_".join(dataset.split("_")[1:])
        name = f"{name}_{sfx}"
    outpath = f"{outpath}/{name}"
    os.makedirs(outpath, exist_ok=True)
    imgpath = os.path.join(outpath, f"{name}{catstr}")
    return imgpath


def draw_total_images(
    input_path, outpath, dataset, P=0, S=0, G=0, figsize=(24, 24), crpt=0
):
    """
    Opens fits files from local directory path to generate total detection drizzled images
    aligned to WCS with point/segment/gaia catalog overlay options. Saves figure as png.

    **args**
    input_path: path to dataset subdirectories containing total or filter fits files
    dataset: name of subdirectory containing .fits and .ecsv files

    **kwargs**
    output_img: where to save the pngs (path) default='./img'
    P: draw point catalog references (0=off, 1=on) default is 0
    S: draw segment catalog references (0=off, 1=on) default is 0
    G: draw GAIA catalog references (0=off, 1=on) default is 0
    figsize: size to make the figures (default=24 sets figsize=(24,24))
    corrs: determines png file naming convention

    PNG naming convention is based on fits file unless corrs=1:
    ./input_path/dataset/filename.fits >> ./img_path/dataset/filename.png

    catalog overlay pngs have an additional suffix:
    P=1: _point.png
    S=1: _segment.png
    P=1, S=1: _source.png
    G=1: _gaia.png


    Normal SVM data (dataset=ib1f0a):
    ./{input_path}/ib1f0a/hst_11570_0a_wfc3_uvis_total_ib1f0a_drc.fits
    saves as >> ./{imgdir}/hst_11570_0a_wfc3_uvis_total_ib1f0a/hst_11570_0a_wfc3_uvis_total_ib1f0a.png

    Corruption SVM data (dataset=ia0m04_f110w_all_stat):
    ./{input_path}/ia0m04_f110w_all_stoc/hst_11099_04_wfc3_ir_total_ia0m04_drz.fits
    saves as >> ./{imgdir}/hst_f110w_all_stoc_uvis_total_ib1f0a/hst_f110w_all_stoc_uvis_total_ib1f0a.png
    """
    # allows for corruption subdir names e.g. ia0m04_f110w_all_stat and ia0m04
    subdir, dname = f"{input_path}/{dataset}", dataset.split("_")[0]
    hfiles = glob.glob(f"{subdir}/*total_{dname}_dr?.fits")
    if len(hfiles) > 0:
        for hfile in hfiles:
            name = os.path.basename(hfile).split(".")[0][:-4]
            detector = name.split("_")[4]
            ras, decs = np.ndarray((0,)), np.ndarray((0,))
            with fits.open(hfile) as ff:
                hdu = ff[1]
                wcs = WCS(hdu.header)
                footprint = wcs.calc_footprint(hdu.header)
                ras = np.append(ras, footprint[:, 0])
                decs = np.append(decs, footprint[:, 1])
                ralim = [np.max(ras), np.min(ras)]
                declim = [np.max(decs), np.min(decs)]
                radeclim = np.stack([ralim, declim], axis=1)
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection=wcs, frameon=False)
                plt.axis(False)
                interval = ZScaleInterval()
                _, vmax = interval.get_limits(hdu.data)
                norm = ImageNormalize(hdu.data, vmin=0, vmax=vmax * 2, clip=True)
                ax.imshow(hdu.data, origin="lower", norm=norm, cmap="gray")

            if P:
                p_cat = glob.glob(f"{subdir}/{name}_point-cat.ecsv")
                if len(p_cat) > 0:
                    point, pfcolor_, pfcolor = draw_catalogs(p_cat[0], "point")
                    if pfcolor_ is not None:
                        for fcol in pfcolor_.unique():
                            if fcol is not None:
                                q = pfcolor == fcol[0]
                                ax.scatter(
                                    point[q]["RA"],
                                    point[q]["DEC"],
                                    edgecolor=fcol[0],
                                    facecolor="none",
                                    transform=ax.get_transform("fk5"),
                                    marker="o",
                                    s=15,
                                    alpha=0.5,
                                )
                # else:
                #     print("Point cat not found: ", dataset)

            if S:
                s_cat = glob.glob(f"{subdir}/{name}_segment-cat.ecsv")
                if len(s_cat) > 0:
                    seg, sfcolor_, sfcolor = draw_catalogs(s_cat[0], "segment")
                    if sfcolor_ is not None:
                        for fcol in sfcolor_.unique():
                            if fcol is not None:
                                q = sfcolor == fcol[0]
                                ax.scatter(
                                    seg[q]["RA"],
                                    seg[q]["DEC"],
                                    edgecolor=fcol[0],
                                    facecolor="none",
                                    transform=ax.get_transform("fk5"),
                                    marker="o",
                                    s=15,
                                    alpha=0.5,
                                )
                # else:
                #     print("Segment cat not found: ", dataset)

            if G:
                g_cat = glob.glob(f"{subdir}/*_{detector}_*GAIAeDR3_ref_cat.ecsv")
                if len(g_cat) > 0:
                    if os.path.exists(g_cat[0]):
                        gaia = ascii.read(g_cat[0]).to_pandas()
                        ax.scatter(
                            gaia["RA"],
                            gaia["DEC"],
                            edgecolor="cyan",
                            facecolor="none",
                            transform=ax.get_transform("fk5"),
                            marker="o",
                            s=15,
                        )
                # else:
                #     print("GAIA cat not found: ", dataset)

            xlim, ylim = wcs.wcs_world2pix(radeclim, 1).T
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            imgpath = create_image_name(name, dataset, P, S, G, crpt, outpath)
            plt.savefig(imgpath, bbox_inches="tight")
            plt.close(fig)
            # print(f"\t{imgpath}.png")
    else:
        print(f"{dataset} fits file could not be found")
        return


def list_visits(dataset, outpath):
    df = pd.read_csv(dataset, index_col="index")
    idx = list(df.index)
    datasets = []
    skip = 0
    for i in idx:
        impath = os.path.join(outpath, i)
        visit = i.split("_")[6]
        if os.path.exists(impath):
            num = len(glob.glob(f"{impath}/*"))
            if num < 3:
                datasets.append(visit)
            else:
                skip += 1
        else:
            datasets.append(visit)
    if skip > 0:
        print("Skipping pre-existing images: ", skip)
    return list(set(datasets))


def generate_total_images(
    input_path, outpath, dataset=None, figsize=(24, 24), crpt=0, gen=3
):
    if dataset is not None:
        if dataset.endswith(".csv"):
            datasets = list_visits(dataset, outpath)
        else:
            datasets = [dataset]
    else:
        if crpt == 0:
            inputs = glob.glob(f"{input_path}/??????")
        else:
            inputs = glob.glob(f"{input_path}/??????_*_???_st??")
        datasets = [i.split("/")[-1] for i in inputs]
    print(f"\nFound {len(datasets)} datasets.")
    if len(datasets) == 0:
        print("Exiting.")
        sys.exit(1)
    start = time.time()
    stopwatch("DRAWING IMAGES", t0=start)
    print(f"Generating images for {len(datasets)} datasets.")
    for dataset in tqdm(datasets):
        # print(dataset)
        if gen == 3:  # original, point-segment, and GAIA
            draw_total_images(input_path, outpath, dataset, figsize=figsize, crpt=crpt)
            draw_total_images(
                input_path, outpath, dataset, P=1, S=1, figsize=figsize, crpt=crpt
            )
            draw_total_images(
                input_path, outpath, dataset, G=1, figsize=figsize, crpt=crpt
            )
        elif gen == 2:  # GAIA
            draw_total_images(
                input_path, outpath, dataset, G=1, figsize=figsize, crpt=crpt
            )
        elif gen == 1:  # point-segment
            draw_total_images(
                input_path, outpath, dataset, P=1, S=1, figsize=figsize, crpt=crpt
            )
        else:  # original (0)
            draw_total_images(input_path, outpath, dataset, figsize=figsize, crpt=crpt)
    end = time.time()
    stopwatch("IMAGE GENERATION", t0=start, t1=end)


def draw_filter_images(input_path, outpath, dataset, figsize=(24, 24), crpt=0):
    subdir, dname = f"{input_path}/{dataset}", dataset.split("_")[0]
    filter_files = glob.glob(f"{subdir}/*[!total]_{dname}_dr?.fits")
    if len(filter_files) > 0:
        outpath = os.path.join(outpath, dname)
        os.makedirs(outpath, exist_ok=True)
    else:
        print("Filter images missing: ", dataset)
        return
    for hfile in filter_files:
        ras, decs = np.ndarray((0,)), np.ndarray((0,))
        with fits.open(hfile) as ff:
            hdu = ff[1]
            wcs = WCS(hdu.header)
            footprint = wcs.calc_footprint(hdu.header)
            ras = np.append(ras, footprint[:, 0])
            decs = np.append(decs, footprint[:, 1])
            ralim = [np.max(ras), np.min(ras)]
            declim = [np.max(decs), np.min(decs)]
            radeclim = np.stack([ralim, declim], axis=1)
            fig = plt.figure(figsize=figsize, edgecolor="k", frameon=False)
            ax = fig.add_subplot(111, projection=wcs, frameon=False)
            plt.axis(False)
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(hdu.data)
            norm = ImageNormalize(hdu.data, vmin=vmin, vmax=vmax * 2, clip=True)
            xlim, ylim = wcs.wcs_world2pix(radeclim, 1)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.imshow(hdu.data, origin="lower", norm=norm, cmap="gray")

        name = os.path.basename(hfile).split(".")[0][:-4]
        if crpt:
            pfx, sfx = "_".join(dataset.split("_")[1:]), "_".join(name.split("_")[4:])
            name = f"hst_{pfx}_{sfx}"
        imgpath = os.path.join(outpath, name)
        plt.savefig(imgpath, bbox_inches="tight")
        plt.close(fig)
        print(f"\t{imgpath}.png")


def generate_filter_images(input_path, outpath, dataset=None, figsize=(24, 24), crpt=0):
    if dataset is not None:
        datasets = [dataset]
    else:
        if os.path.exists(f"{input_path}/.DS_Store"):  # only happens on Mac
            os.remove(f"{input_path}/.DS_Store")
        datasets = os.listdir(input_path)
    for dataset in datasets:
        print(dataset)
        draw_filter_images(input_path, outpath, dataset, figsize=figsize, crpt=crpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="path to datasets directory")
    parser.add_argument("outpath", type=str, help="directory path to save png images")
    parser.add_argument(
        "-t",
        "--imgtype",
        type=str,
        choices=["total", "filter"],
        default="total",
        help="draw total detection or filter level images",
    )
    parser.add_argument(
        "-g",
        "--generator",
        type=int,
        choices=[0, 1, 2, 3],
        default=3,
        help="0: generate original only; 1: point-segment, 2: gaia; 3: original, point-segment, and gaia (3 separate images)",
    )
    parser.add_argument("-s", "--size", type=int, default=24, help="figsize")
    parser.add_argument(
        "-c",
        "--corruptions",
        type=int,
        default=0,
        help="corrupted datasets (used to format png names)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="specify single dataset (default is None to generate images for all dataset subdirectories in input_path",
    )
    args = parser.parse_args()
    input_path = args.input_path
    outpath = args.outpath
    img_type = args.imgtype
    gen = args.generator
    size = (args.size, args.size)
    crpt = args.corruptions
    dataset = args.dataset
    if img_type == "total":
        generate_total_images(
            input_path, outpath, dataset=dataset, figsize=size, crpt=crpt, gen=gen
        )
    else:
        generate_filter_images(
            input_path, outpath, dataset=dataset, figsize=size, crpt=crpt
        )
