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

class DrawMosaics:
    def __init__(self, input_path, output_path=None, filename=None, visit=None, pattern="", gen=3, size=(24,24), crpt=0):
        self.input_path = input_path
        self.output_path = self.checkout_output(output_path)
        self.filename = filename
        self.visit = visit
        self.pattern = pattern
        self.gen = gen
        self.size = size
        self.crpt = crpt
        self.datasets = self.get_datasets()

    def check_output(self, output_path):
        if output_path is None:
            self.output_path = os.path.join(os.getcwd(), "img")
            # TODO if permission error, write to /tmp 
        os.makedirs(self.output_path)
        return self.output_path

    def get_datasets(self):
        if self.filename:
            return self.load_from_file()
        elif self.visit:
            return [self.visit]
        else:
            return self.local_search()

    def load_from_file(self):
        df = pd.read_csv(self.filename, index_col="index")
        idx = list(df.index)
        self.datasets = []
        skip = 0
        for i in idx:
            impath = os.path.join(self.output_path, i)
            visit = i.split("_")[6]
            if os.path.exists(impath):
                num = len(glob.glob(f"{impath}/*"))
                if num < 3:
                    self.datasets.append(visit)
                else:
                    skip += 1
            else:
                self.datasets.append(visit)
        if skip > 0:
            print("Skipping pre-existing images: ", skip)
        return list(set(self.datasets))

    def local_search(self):
        if self.crpt == 0:
            search_sfx = "/??????"
        else:
            search_sfx = "/??????_*_???_st??"
        search_pattern = self.pattern + search_sfx
        inputs = glob.glob(f"{self.input_path}/{search_pattern}")
        if len(inputs) == 0:
            inputs = glob.glob(f"{self.input_path}{search_sfx}")
        try:
            self.datasets = [i.split("/")[-1] for i in inputs]
            print(f"\nFound {len(self.datasets)} datasets.")
            return self.datasets
        except Exception as e:
            print(e)
            print("No datasets found. Exiting.")
            sys.exit(1)

    def point_flag_color(self, x):
        if x <= 1:
            return "red", "Flag <= 1"
        elif x <= 5:
            return "green", "2 <= Flag <= 5"
        else:
            return None, None  # 'yellow', 'Flag > 5'

    def segment_flag_color(self, x):
        if x <= 1:
            return "blue", "Flag <= 1"
        elif x <= 5:
            return "green", "2 <= Flag <= 5"
        else:
            return None, None  # 'yellow', 'Flag > 5'

    def draw_catalogs(self, cfile, catalog):
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
                    fcolor_ = flags.apply(self.point_flag_color)
                elif catalog == "segment":
                    fcolor_ = flags.apply(self.segment_flag_color)
                fcolor = fcolor_.apply(lambda x: x[0]).values
        return cat, fcolor_, fcolor

    def create_image_name(self, name, dataset, P=0, S=0, G=0):
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
        if self.crpt:
            sfx = "_".join(dataset.split("_")[1:])
            name = f"{name}_{sfx}"
        img_out = f"{self.output_path}/{name}"
        os.makedirs(img_out, exist_ok=True)
        imgpath = os.path.join(self.output_path, f"{name}{catstr}")
        return imgpath

    def generate_total_images(self):
        start = time.time()
        stopwatch("DRAWING IMAGES", t0=start)
        if self.datasets is None:
            print("No datasets available. Exiting")
            sys.exit(1)

        print(f"Generating images for {len(self.datasets)} datasets.")
        
        for dataset in tqdm(self.datasets):
            if self.gen == 3:  # original, point-segment, and GAIA
                self.draw_total_images(dataset)
                self.draw_total_images(dataset, P=1, S=1)
                self.draw_total_images(dataset, G=1)
            elif self.gen == 2:  # GAIA
                self.draw_total_images(dataset, G=1)
            elif self.gen == 1:  # point-segment
                self.draw_total_images(dataset, P=1, S=1)
            else:  # original (0)
                self.draw_total_images(dataset)
        end = time.time()
        stopwatch("IMAGE GENERATION", t0=start, t1=end)

    def draw_total_images(self, dataset, P=0, S=0, G=0):
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
        subdir, dname = f"{self.input_path}/{dataset}", dataset.split("_")[0]
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
                    fig = plt.figure(figsize=self.size)
                    ax = fig.add_subplot(111, projection=wcs, frameon=False)
                    plt.axis(False)
                    interval = ZScaleInterval()
                    _, vmax = interval.get_limits(hdu.data)
                    norm = ImageNormalize(hdu.data, vmin=0, vmax=vmax * 2, clip=True)
                    ax.imshow(hdu.data, origin="lower", norm=norm, cmap="gray")

                if P:
                    p_cat = glob.glob(f"{subdir}/{name}_point-cat.ecsv")
                    if len(p_cat) > 0:
                        point, pfcolor_, pfcolor = self.draw_catalogs(p_cat[0], "point")
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
                if S:
                    s_cat = glob.glob(f"{subdir}/{name}_segment-cat.ecsv")
                    if len(s_cat) > 0:
                        seg, sfcolor_, sfcolor = self.draw_catalogs(s_cat[0], "segment")
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
                xlim, ylim = wcs.wcs_world2pix(radeclim, 1).T
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                imgpath = self.create_image_name(name, dataset, P=P, S=S, G=G)
                plt.savefig(imgpath, bbox_inches="tight")
                plt.close(fig)
        else:
            return

    def generate_filter_images(self):
        start = time.time()
        stopwatch("DRAWING IMAGES", t0=start)
        if self.datasets is None:
            print("No datasets available. Exiting")
            sys.exit(1)
        print(f"Generating images for {len(self.datasets)} datasets.")
        for dataset in tqdm(self.datasets):
            self.draw_filter_images(dataset)

    def draw_filter_images(self, dataset):
        subdir, dname = f"{self.input_path}/{dataset}", dataset.split("_")[0]
        filter_files = glob.glob(f"{subdir}/*[!total]_{dname}_dr?.fits")
        if len(filter_files) > 0:
            outpath = os.path.join(self.output_path, dname)
            os.makedirs(outpath, exist_ok=True)
        else:
            print("Filter images missing: ", dataset)
            return
        for hfile in filter_files:
            name = os.path.basename(hfile).split(".")[0][:-4]
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
                fig = plt.figure(figsize=self.size, edgecolor="k", frameon=False)
                ax = fig.add_subplot(111, projection=wcs, frameon=False)
                plt.axis(False)
                interval = ZScaleInterval()
                vmin, vmax = interval.get_limits(hdu.data)
                norm = ImageNormalize(hdu.data, vmin=vmin, vmax=vmax * 2, clip=True)
                xlim, ylim = wcs.wcs_world2pix(radeclim, 1)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.imshow(hdu.data, origin="lower", norm=norm, cmap="gray")

            imgpath = self.create_image_name(name, dataset)
            plt.savefig(imgpath, bbox_inches="tight")
            plt.close(fig)
            print(f"\t{imgpath}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="path to datasets directory")
    parser.add_argument("-o", "--output_path", type=str, help="directory path to save png images")
    parser.add_argument(
        "-f",
        "--fname",
        type=str,
        default=None,
        help="csv (dataframe) filename for generating specific list of datasets",
    )
    parser.add_argument(
        "-v",
        "--visit",
        type=str,
        default=None,
        help="specify one visit (default is None to generate images for all visit subdirectories in input_path",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="",
        help="glob search pattern (to restrict image generator to look for only certain visit names, e.g. 'ia*'",
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
        help="set equal to 1 to format png names for artificially misaligned images; default=0 (off)",
    )
    parser.add_argument(
        "-t",
        "--imgtype",
        type=str,
        choices=["total", "filter"],
        default="total",
        help="draw total detection (default) or filter level images",
    )
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    fname = args.fname
    visit = args.visit
    pattern = args.pattern
    gen = args.generator
    size = (args.size, args.size)
    crpt = args.corruptions
    draw = DrawMosaics(input_path, output_path=output_path, fname=fname, visit=visit, pattern=pattern, gen=gen, size=size, crpt=crpt)
    if args.img_type == "total":
        draw.generate_total_images()
    else:
        draw.generate_filter_images()
