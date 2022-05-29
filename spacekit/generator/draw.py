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
    """Class for generating machine-learning image inputs from drizzled total detection fits files and their associated catalog files. Primarily used for creating multiple images at once (batch) with capability for single images/single dataset also available."""

    def __init__(
        self,
        input_path,
        output_path=None,
        fname=None,
        visit=None,
        pattern="",
        gen=3,
        size=(24, 24),
        crpt=0,
    ):
        """Initializes a DrawMosaics class object.

        Parameters
        ----------
        input_path : str (path)
            path to dataset subdirectories containing total or filter fits files
        output_path : [type], optional
            where to save the pngs (path), by default None (will create and save in relative dir path named "img" )
        fname : str (path), optional
            csv (dataframe) fname for generating specific list of datasets, by default None
        visit : [type], optional
            name of specific subdirectory (typically ipppssoot or visit name) containing .fits and .ecsv files, by default None
        pattern : str, optional
            glob search pattern (to restrict image generator to look for only certain visit names, e.g. 'ia*', by default ''
        gen : int, optional
            generator method to use: 0=generate original only; 1=point-segment, 2=gaia; 3=original, point-segment, and gaia (3 separate images), by default 3
        size : tuple, optional
            size to make the figures i.e. figsize=(size,size), by default (24, 24)
        crpt : int, optional
            modifies the input search pattern as well as output png file naming convention (so that a non-corrupt visit of the same name is not overwritten), by default 0
        """
        self.input_path = input_path
        self.output_path = output_path
        self.check_output()
        self.fname = fname
        self.visit = visit
        self.pattern = pattern
        self.gen = gen
        self.size = size
        self.crpt = crpt
        self.rgx = self.check_format()
        self.status = {"new": [], "skip": [], "err": []}
        self.datasets = self.get_datasets()
        self.clip = True
        self.manual = None

    def check_output(self):
        """check if a custom output_path is set, otherwise create a subdirectory "img" in the current working directory and set as the output_path attribute.

        Returns
        -------
        str (path)
            path to subdirectory for saving png images.
        """
        if self.output_path is None:
            self.output_path = os.path.join(os.getcwd(), "img")
            # TODO if permission error, write to /tmp
        os.makedirs(self.output_path, exist_ok=True)
        return self.output_path

    def check_format(self, dname=None):
        if dname is None:
            dname = "??????"
        if self.crpt == 1:
            return f"{dname}_*_???_st??"
        else:
            return dname

    def get_hapfiles(self, dataset):
        if self.pattern:
            subdir = f"{self.input_path}/{self.pattern}/{dataset}"
        else:
            subdir = f"{self.input_path}/{dataset}"
        dname = dataset.split("_")[0]
        hfiles = glob.glob(f"{subdir}/*total_{dname}_dr?.fits")
        return subdir, hfiles

    def get_datasets(self):
        """Locate inputs (fits file directories) to use for drawing the images. Search method used is based on parameters set when the DrawMosaics class object is instantiated. If multiple parameters are passed in, the order of search priority is 1) `fname`: only look for visits found in the csv file/dataframe (uses `load_from_file` method); 2) `visit` only look for subdirectories matching this specific visit name; 3) `local_search`: glob-based search for any visit subdirectories matching the pattern set in `pattern` attribute. (if crpt)

        Returns
        -------
        list
            list of datasets/visits found according to search method used
        """
        if self.fname:
            return self.load_from_file()
        elif self.visit:
            return [self.visit]
        else:
            return self.local_search()

    def load_from_file(self):
        """only look for visits found in the csv file/dataframe.

        Returns
        -------
        list
            restricted list of inputs (visits) for which images will be drawn
        """
        if not self.fname.endswith("csv"):
            self.fname += ".csv"
        df = pd.read_csv(self.fname, index_col="index")
        idx = list(df.index)

        self.datasets = []
        skip = []
        for i in idx:
            impath = os.path.join(self.output_path, i)
            visit = i.split("_")[6] if not self.crpt else "_".join(i.split("_")[6:])

            if os.path.exists(impath):
                num = len(glob.glob(f"{impath}/*"))
                if num < 3:
                    self.datasets.append(visit)
                else:
                    skip.append(visit)
            else:
                self.datasets.append(visit)
        if len(skip) > 0:
            self.status["skip"] = list(set(skip))
            print("Skipping found images: ", len(self.status["skip"]))
        print(f"\nFound {len(self.datasets)} new datasets.")
        return list(set(self.datasets))

    def local_search(self):
        """only look for visit names matching a glob-based search pattern.

        Returns
        -------
        list
            list of inputs (visits) for which images will be drawn
        """
        search = self.pattern if self.pattern else self.rgx
        inputs = glob.glob(f"{self.input_path}/{search}")
        if len(inputs) == 0:  # try one more directory down
            print("None found - Checking subdirectories")
            inputs = glob.glob(f"{self.input_path}/*/{search}")
        if len(inputs) == 0:  # fall back to wildcard
            print("None found - using fallback (wildcard)")
            inputs = glob.glob(f"{self.input_path}/*{self.rgx}")
        try:
            self.datasets = [i.split("/")[-1] for i in inputs]
            print(f"\nFound {len(self.datasets)} datasets.")
            return self.datasets
        except Exception as e:
            print(e)
            print("No datasets found. Exiting.")
            sys.exit(1)

    def point_flag_color(self, x):
        """determines whether or not to draw a small red (or green) circle on top of the original image data depending on the value found in point source catalog. More info on the values associated with the "flag" color can be found in the Drizzlepac handbook at drizzlepac.readthedocs.io (Drizzlepac.catalog_generation api)

        Parameters
        ----------
        x : int
            value pulled from point catalog file

        Returns
        -------
        str
            "red", "green" or None depending on input value
        """
        if x <= 1:
            return "red", "Flag <= 1"
        elif x <= 5:
            return "green", "2 <= Flag <= 5"
        else:
            return None, None  # 'yellow', 'Flag > 5'

    def segment_flag_color(self, x):
        """draw a small blue circle on top of the original image data depending on the value found in segment source catalog.

        Parameters
        ----------
        x : int
            value pulled from segment catalog file

        Returns
        -------
        str
            "blue", "green" or None depending on input value
        """
        if x <= 1:
            return "blue", "Flag <= 1"
        elif x <= 5:
            return "green", "2 <= Flag <= 5"
        else:
            return None, None  # 'yellow', 'Flag > 5'

    def draw_catalogs(self, cfile, catalog):
        """Open and read .escv catalog file associated with the visit (if available) and map the appropriate values and coordinates to draw as an overlay on the original image. Credit: based in part on code by M. Burger

        Parameters
        ----------
        cfile : str (path)
            path to source catalog file
        catalog : str
            "point", "segment" or "gaia"

        Returns
        -------
        Pandas dataframe, lists of flag colors
            table of catalog values and associated flag colors.
        """
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

    def create_image_name(self, name, dataset, P=0, S=0, G=0, fgroup=None):
        """Determines which suffix to append to the output png file based on which catalog(s) are used (if any).

        Parameters
        ----------
        name : str
            visit name
        dataset : [type]
            visit name (used to adjust `name` if crpt=1)
        P : int, optional
            draw point catalog overlay (if available), by default 0
        S : int, optional
            draw segment catalog overlay (if available), by default 0
        G : int, optional
            draw Gaia catalog overlay (if eDR3 or GSC242 available), by default 0

        Returns
        -------
        str
            path to png output for this image.
        """
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
        if fgroup:  # rel filter images share same parent dir
            img_out = fgroup
        else:
            img_out = f"{self.output_path}/{name}"
            os.makedirs(img_out, exist_ok=True)
        imgpath = os.path.join(img_out, f"{name}{catstr}")
        return imgpath

    def generate_total_images(self):
        """Batch image generation method for multiple datasets (and multiple catalog types)"""
        base = os.path.dirname(os.path.abspath(self.output_path))
        start = time.time()
        stopwatch("DRAWING IMAGES", t0=start, out=base)
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
        stopwatch("IMAGE GENERATION", t0=start, t1=end, out=base)

    def draw_total_images(self, dataset, P=0, S=0, G=0):
        """Primary class method for plotting the data, drawing the catalogs (if any) and saving to local disk as png.

        Parameters
        ----------
        dataset : str
            visit name (single dataset, for batch jobs this comes from looping over the list stored in self.datasets)
        P : int, optional
            draw point catalog overlay (if available), by default 0
        S : int, optional
            draw segment catalog overlay (if available), by default 0
        G : int, optional
            draw Gaia catalog overlay (eDR3 or GSC242 if available), by default 0
        """
        subdir, hfiles = self.get_hapfiles(dataset)
        if len(hfiles) > 0:
            for hfile in hfiles:
                name = os.path.basename(hfile).split(".")[0][:-4]
                detector = name.split("_")[4]
                with fits.open(hfile) as ff:
                    hdu = ff[1]
                    wcs = WCS(hdu.header)
                    footprint = wcs.calc_footprint(hdu.header)
                    ralim = [np.max(footprint[:, 0]), np.min(footprint[:, 0])]
                    declim = [np.max(footprint[:, 1]), np.min(footprint[:, 1])]
                    radeclim = np.stack([ralim, declim], axis=1)
                    fig = plt.figure(figsize=self.size)
                    ax = fig.add_subplot(111, projection=wcs, frameon=False)
                    plt.axis(False)
                    interval = ZScaleInterval()
                    zmin, zmax = interval.get_limits(hdu.data)
                    if self.manual is None:
                        norm = ImageNormalize(
                            hdu.data, vmin=0, vmax=zmax * 2, clip=self.clip
                        )
                    elif self.manual == "zscale":
                        norm = ImageNormalize(
                            hdu.data, vmin=zmin, vmax=zmax, clip=self.clip
                        )
                    elif type(self.manual) == dict:
                        try:
                            norm = ImageNormalize(hdu.data, **self.manual)
                        except Exception as e:
                            print(e)
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
                    g_cat = glob.glob(f"{subdir}/*_{detector}_*G*_ref_cat.ecsv")
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
                self.status["new"].append(dataset)
        else:
            self.status["err"].append(dataset)

    def generate_filter_images(self):
        """Generate batch of relative filter drizzle file images."""
        start = time.time()
        stopwatch("DRAWING IMAGES", t0=start)
        if self.datasets is None:
            print("No datasets available. Exiting")
            sys.exit(1)
        print(f"Generating images for {len(self.datasets)} datasets.")
        for dataset in tqdm(self.datasets):
            self.draw_filter_images(dataset)

    def draw_filter_images(self, dataset):
        """Generate relative filter drizzle file images

        Parameters
        ----------
        dataset : str
            name of input visit dataset
        """
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
            imgpath = self.create_image_name(name, dataset, fgroup=outpath)
            plt.savefig(imgpath, bbox_inches="tight")
            plt.close(fig)
            print(f"\n\t{imgpath}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="path to datasets directory")
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="directory path to save png images",
    )
    parser.add_argument(
        "-f",
        "--fname",
        type=str,
        default=None,
        help="csv (dataframe) fname for generating specific list of datasets",
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
        help="restrict to directory paths matching a glob-based pattern, e.g. 'ia*' or 'ia*/*' for subdirectories",
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
        "--crpt",
        type=int,
        default=0,
        help="set equal to 1 to also search for synthetic dataset inputs (artificially misaligned images) as well as format png names; default=0 (off)",
    )
    parser.add_argument(
        "-t",
        "--img_type",
        type=str,
        choices=["total", "filter"],
        default="total",
        help="draw total detection (default) or filter level images",
    )
    args = parser.parse_args()
    draw = DrawMosaics(
        args.input_path,
        output_path=args.output_path,
        fname=args.fname,
        visit=args.visit,
        pattern=args.pattern,
        gen=args.generator,
        size=(args.size, args.size),
        crpt=args.crpt,
    )
    if args.img_type == "total":
        draw.generate_total_images()
    else:
        draw.generate_filter_images()
