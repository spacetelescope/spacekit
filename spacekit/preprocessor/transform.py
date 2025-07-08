import os
import json
import pandas as pd
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf
from astropy.coordinates import SkyCoord
from astropy import units as u

from spacekit.logger.log import Logger


class SkyTransformer:
    """Calculate sky separation / reference pixel offset statistics

    Parameters
    ----------
    mission : str
        Name of mission or observatory, e.g. "JWST", "HST"
    name : str, optional
        logging name, by default "SkyTransformer"
    """
    def __init__(self, mission, name="SkyTransformer", **log_kws):
        self.__name__ = name
        self.log = Logger(self.__name__).spacekit_logger(**log_kws)
        self.mission = mission
        self.pixel_scales = self.image_pixel_scales()
        self.instr = None
        self.detector = None
        self.channel = None
        self.count_exposures = True
        self.refpix = dict()
        self.set_keys()

    def set_keys(self, **kwargs):
        """
        Set keys used in exposure header dictionary to identify values
        (typically derived from fits file sciheaders). Possible keyword
        arguments include: instr,detector,channel,ra,dec where 'ra','dec'
        refer to the fiducial (center pixel coordinate in degrees).
        None values will use defaults (see below); unrecognized kwargs
        will be ignored.

        Defaults
        --------
        * instr="INSTRUME"
        * detector="DETECTOR"
        * channel="CHANNEL"
        * band="BAND"
        * exp_type="EXP_TYPE"
        * ra="CRVAL1" / could also use "RA_REF"
        * dec="CRVAL2" / could also use "DEC_REF"
        """
        self.instr_key = kwargs.get("instr", "INSTRUME")
        self.detector_key = kwargs.get("detector", "DETECTOR")
        self.channel_key = kwargs.get("channel", "CHANNEL")
        self.band_key = kwargs.get("band", "BAND")
        self.exp_key = kwargs.get("exp_type", "EXP_TYPE")
        self.ra_key = kwargs.get("ra", "CRVAL1")
        self.dec_key = kwargs.get("dec", "CRVAL2")
        self.ra_key2 = "RA_REF" if self.ra_key == "CRVAL1" else "CRVAL1"
        self.dec_key2 = "DEC_REF" if self.dec_key == "CRVAL2" else "CRVAL2"

    def calculate_offsets(self, product_exp_headers):
        """Given key-value pairs of header info from a set of input exposures,
        estimate the fiducial (center pixel coordinates) of the final image product
        and calculated pixel offset statistics between inputs and final output using
        detector-based footprints and sky separation angles.

        NOTE: the product keys and input exposure keys could be any strings and are used
        simply for organization. The fits-related key-value pairs nested within each input
        exposure dictionary must contain, at minimum, the instrument and fiducial
        ra/dec coordinates (e.g. "INSTRUME","CRVAL1","CRVAL1"). The keys themselves
        can be custom set using `self.set_keys(**kwargs)` but must match the contents
        of the nested dictionary passed into `product_exp_headers`. Typically these are
        derived directly from fits file sci headers of the input exposures.

        Some missions and instruments require additional information such as "CHANNEL"
        (JWST Nircam) or "DETECTOR" (HST) in order to identify the correct pixel scale
        and footprint size based on the detector and/or wavelength channel.

        Parameters
        ----------
        product_exp_headers : dict
            nested dictionary of (typically Level 3) product names (keys),
            their input exposures (values) and relevant fits header information
            per exposure (key-value pairs).
        
        Returns
        -------
        dict
            calculated pixel offset statistics for each L3 product's group of input exposures
        """
        product_refpix = dict()
        for product, exp_headers in product_exp_headers.items():
            product_refpix[product] = self.get_pixel_offsets(exp_headers)
        return product_refpix

    def validate_fiducial(self, fiducial, exp):
        """Checks fiducial to ensure value is valid. 

        Parameters
        ----------
        fiducial : tuple of floats
            ra and dec values
        exp : str
            fiducial type (TARG_RA/TARG_DEC, CRVAL1/CRVAL2, RA_REF/DEC_REF)

        Returns
        -------
        bool
            Valid (True) or invalid (False) 
        """
        (ra, dec) = fiducial
        if isinstance(ra, float) and isinstance(dec, float):
            return True
        else:
            warning_message = f"Invalid RA/DEC fiducial value ({ra}, {dec}) in {str(exp)}"
            if exp == "TARG_RA/TARG_DEC":
                self.log.debug(warning_message)
            else:
                self.log.warning(warning_message)
            return False

    def get_pixel_offsets(self, exp_data):
        """Calculates the relative pixel offset statistics for a group of L1 input exposures.

        Parameters
        ----------
        exp_data : dict
            key value pairs of input exposure names and their associated Fits header metadata.

        Returns
        -------
        dict
            pixel offset statistics for this group of input exposures
        """
        if self.count_exposures is True:
            refpix = dict(NEXPOSUR=len(list(exp_data.keys())))
        else:
            refpix = dict()
        offsets, targ_offsets, detectors, bands = [], [], [], []
        targ_radec = None
        bad_fiducials = {}
        for exp, data in exp_data.items():
            fiducial = (data.get(self.ra_key, self.ra_key2), data.get(self.dec_key, self.dec_key2))
            # only need to set once bc consisent across exposures
            if targ_radec is None:
                targ_radec = (data.get("TARG_RA", ''), data.get("TARG_DEC", ''))
            # validate fiducials
            if self.validate_fiducial(fiducial, exp) is False:
                bad_fiducials[exp] = str(exp)
                continue
            instr = data[self.instr_key]
            detector = data.get(self.detector_key, None)
            channel = data.get(self.channel_key, None)
            band = data.get(self.band_key, None)
            exp_type = data.get(self.exp_key, None)
            scale = self.get_scale(
                instr, channel=channel, detector=detector, exp_type=exp_type
            )
            shape = self.data_shapes(instr)
            # footprint from shape
            footprint = self.footprint_from_shape(fiducial, scale, shape)
            exp_data[exp].update(
                dict(
                    fiducial=fiducial,
                    footprint=footprint,
                    scale=scale,
                )
            )
            if detector is not None and detector.upper() not in detectors:
                detectors.append(detector.upper())
            # MIRI MRS: determine bands used: short, long, shortmedium, shortmediumlong
            if band is not None:
                bands.extend([b.upper() for b in band.split('-') if b.upper() not in bands])
        # Throw out any exposures with invalid data
        for k in bad_fiducials.keys():
            del exp_data[k]
            if 'NEXPOSUR' in refpix:
                refpix['NEXPOSUR'] -= 1
        # if all exposures were bad, return empty dict
        if len(exp_data) < 1:
            return {}
        # find fiducial (final product)
        footprints = [v["footprint"] for v in exp_data.values()]
        lon_fiducial, lat_fiducial = self.estimate_fiducial(footprints)
        refpix["fx_ra"], refpix["fy_dec"] = lon_fiducial, lat_fiducial
        # pixel sky sep offsets from estimated fiducial
        pcoord = SkyCoord(lon_fiducial, lat_fiducial, unit="deg")
        tcoord = None
        if self.validate_fiducial(targ_radec, 'TARG_RA/TARG_DEC') is True:
            tcoord = SkyCoord(targ_radec[0], targ_radec[1], unit="deg")
        for exp, data in exp_data.items():
            (ra, dec) = data["fiducial"]
            pixel = self.pixel_sky_separation(ra, dec, pcoord, data["scale"])
            exp_data[exp]["offset"] = pixel
            offsets.append(pixel)
            if tcoord:
                targ_pixel = self.pixel_sky_separation(ra, dec, tcoord, data["scale"])
                exp_data[exp]["targ_offset"] = targ_pixel
                targ_offsets.append(targ_pixel)
        # fill in metadata for product using reference exposure (usually vals are equal across inputs)
        ref_exp = [
            k for k, v in exp_data.items() if v["offset"] == np.min(np.asarray(offsets))
        ][0]
        keys = [
            k
            for k in list(exp_data[ref_exp].keys())
            if k not in ["DETECTOR", "BAND", "footprint", "fiducial"]
        ]
        for k in keys:
            refpix[k] = exp_data[ref_exp][k]
        if len(detectors) > 1:
            refpix["DETECTOR"] = "|".join(sorted([d for d in detectors]))
        else:
            refpix["DETECTOR"] = detectors[0]
        if len(bands) > 1:
            refpix["BAND"] = "|".join(sorted([b for b in bands], reverse=True))
        elif len(bands) == 1:
            refpix["BAND"] = bands[0]
        else:
            refpix["BAND"] = 'NONE'
        # offset statistics
        offset_stats = self.offset_statistics(offsets)
        refpix.update(offset_stats)
        if targ_offsets:
            targ_offset_stats = self.offset_statistics(targ_offsets, pfx="targ_")
            refpix.update(targ_offset_stats)
        # experimental
        try:
            # set default to 0.0 as fallback if calculation fails
            refpix["t_offset"] = 0.0
            refpix["gs_offset"] = 0.0
            refpix["gs_offset"] = self.pixel_sky_separation(
                refpix["GS_RA"], refpix["GS_DEC"], pcoord, refpix["scale"]
            )
            refpix["t_offset"] = self.pixel_sky_separation(
                refpix["TARG_RA"], refpix["TARG_DEC"], pcoord, refpix["scale"]
            )
        except (ValueError, TypeError):
            self.log.debug("TARG/GS RA DEC vals missing or NaN - setting to 0.0")
        return refpix

    def image_pixel_scales(self):
        return dict(
            HST=dict(ACS=dict(WFC=0.05), WFC3=dict(UVIS=0.04, IR=0.13)),
            JWST=dict(
                NIRCAM=dict(
                    SHORT=0.03,
                    LONG=0.06,
                ),
                MIRI=dict(
                    GEN=0.11,
                    MRS=0.196,
                ),
                NIRISS=0.06,
                NIRSPEC=0.12,
                FGS=0.069,
            ),
        )[self.mission]

    def data_shapes(self, instr):
        return dict(
            JWST=dict(
                NIRCAM=(2048, 2048),
                MIRI=(1032, 1024),
                NIRISS=(2048, 2048),
                NIRSPEC=(2048, 2048),
            ),
            HST=dict(
                ACS=(4096, 2048),  # ACS -> WFC,
                WFC3=(4096, 2051),  # WFC3 -> UVIS (IR=(1024,1024))
            ),
        )[self.mission][instr]

    def get_scale(self, instr, channel=None, detector=None, exp_type=None):
        if channel.upper() in ["SHORT", "LONG"]:
            return self.pixel_scales[instr][channel]
        elif instr.upper() == "MIRI":
            if exp_type in ["MIR_MRS"]:
                return self.pixel_scales[instr]["MRS"]
            else:
                return self.pixel_scales[instr]["GEN"]
        elif detector.upper() in ["WFC", "UVIS", "IR"]:
            return self.pixel_scales[instr][detector]
        else:
            return self.pixel_scales[instr]

    @staticmethod
    def footprint_from_shape(fiducial, scale, shape):
        sep_x = (shape[0] / 2 * scale * u.arcsec).to(u.deg).value
        sep_y = (shape[1] / 2 * scale * u.arcsec).to(u.deg).value

        ra_ref, dec_ref = fiducial

        footprint = np.array(
            [
                [ra_ref - sep_x, dec_ref - sep_y],
                [ra_ref + sep_x, dec_ref - sep_y],
                [ra_ref + sep_x, dec_ref + sep_y],
                [ra_ref - sep_x, dec_ref + sep_y],
            ]
        )
        return footprint

    @staticmethod
    def estimate_fiducial(footprints: list):
        footprints = np.vstack([foot for foot in footprints])

        lon, lat = footprints[:, 0], footprints[:, 1]
        lon, lat = np.deg2rad(lon), np.deg2rad(lat)
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

        x_mid = (np.max(x) + np.min(x)) / 2.0
        y_mid = (np.max(y) + np.min(y)) / 2.0
        z_mid = (np.max(z) + np.min(z)) / 2.0
        lon_fiducial = np.rad2deg(np.arctan2(y_mid, x_mid)) % 360.0
        lat_fiducial = np.rad2deg(np.arctan2(z_mid, np.sqrt(x_mid**2 + y_mid**2)))
        return lon_fiducial, lat_fiducial

    @staticmethod
    def pixel_sky_separation(ra, dec, p_coords, scale, unit="deg"):
        coords = SkyCoord(ra, dec, unit=unit)
        skysep_angle = p_coords.separation(coords)
        arcsec = skysep_angle.arcsecond
        pixel = arcsec / scale
        return pixel

    @staticmethod
    def offset_statistics(offsets, pfx=""):
        offsets = np.asarray(offsets)
        stats = dict()
        stats[f"{pfx}max_offset"] = np.max(offsets)
        stats[f"{pfx}mean_offset"] = np.mean(offsets)
        stats[f"{pfx}sigma_offset"] = np.std(offsets)
        stats[f"{pfx}err_offset"] = np.std(offsets) / np.sqrt(len(offsets))
        sigma1_idx = np.where(offsets > np.mean(offsets) + np.std(offsets))[0]
        if len(sigma1_idx) > 0:
            stats[f"{pfx}sigma1_mean"] = np.mean(offsets[sigma1_idx])
            stats[f"{pfx}frac"] = len(offsets[sigma1_idx]) / len(offsets)
        else:
            stats[f"{pfx}sigma1_mean"] = 0.0
            stats[f"{pfx}frac"] = 0.0
        return stats


class Transformer:
    """Transformer base class. Unless the `cols` attribute is empty, the Transformer object will automatically instantiate some
    of the other attributes needed to transform the data. Using the Transformer subclasses instead is recommended (this
    class is mainly used as an object with general methods to load or save the transform data as well as instantiate some of
    the initial attributes).

    Parameters
    ----------
    data : dataframe or numpy.ndarray
        input data containing continuous feature vectors to be transformed (may also contain vectors or columns of
        categorical and other datatypes as well).
    transformer : class, optional
        transform class to use (e.g. from scikit-learn), by default PowerTransformer(standardize=False)
    cols : list of str or int, optional
        column names (or index values if data is an np.array) of feature vectors to be transformed (i.e. continuous datatype features), by default []
    ncols : list of int, optional
        array index values of feature vectors to be transformed, by default None
    tx_data : dict, optional
        transform metadata calculated previously to be reused during this instantiation, by default None
    tx_file : string, optional
        path to saved transformer metadata, by default None
    save_tx : bool, optional
        save the transformer metadata as json file on local disk, by default True
    join_data : int, optional
        1: join normalized data with remaining columns of original; 2: join with complete original, all columns (requires
        renaming)
    rename : str or list
        if string, will be appended to normalized col names; if list, will rename normalized columns in this order
    output_path : string, optional
        where to save the transformer metadata, by default None (current working directory)
    """
    def __init__(
        self,
        data,
        cols=None,
        ncols=None,
        tx_data=None,
        tx_file=None,
        save_tx=True,
        join_data=1,
        rename="_scl",
        output_path=None,
        name="Transformer",
        **log_kws,
    ):
        self.__name__ = name
        self.log = Logger(self.__name__, **log_kws).spacekit_logger()
        self.data = self.check_shape(data)
        self.cols = cols
        self.ncols = self.check_columns(ncols=ncols)
        self.tx_file = tx_file
        self.save_tx = save_tx
        self.join_data = join_data
        self.rename = rename
        self.output_path = output_path
        self.tx_data = self.load_transformer_data(tx=tx_data)
        self.continuous = self.continuous_data()
        self.categorical = self.categorical_data()

    def check_shape(self, data):
        if len(data.shape) == 1:
            if isinstance(data, np.ndarray):
                data = data.reshape(1, -1)
            elif isinstance(data, pd.Series):
                name = data.name
                data = pd.DataFrame(
                    data.values.reshape(1, -1), columns=list(data.index)
                )
                data["index"] = name
                data.set_index("index", inplace=True)
        return data

    def check_columns(self, ncols=None):
        if ncols is not None and isinstance(self.data, np.ndarray):
            self.cols = ncols
        self.ncols = ncols

    def load_transformer_data(self, tx=None):
        """Loads saved transformer metadata from a dictionary or a json file on local disk.

        Returns
        -------
        dictionary
            transform metadata used for applying transformations on new data inputs
        """
        if tx:
            self.tx_data = tx
        elif self.tx_file is not None:
            with open(self.tx_file, "r") as j:
                self.tx_data = json.load(j)
            return self.tx_data
        else:
            return None

    def save_transformer_data(self, tx=None, fname="tx_data.json"):
        """Save the transform metadata to a json file on local disk. Typical use-case is when you need to transform new inputs
        prior to generating a prediction but don't have access to the original dataset used to train the model.

        Parameters
        ----------
        tx : dictionary
            statistical metadata calculated when applying a transform to the training dataset; for PowerTransform this consists
            of lambdas, means and standard deviations for each continuous feature vector of the dataset.

        Returns
        -------
        string
            path where json file is saved on disk
        """
        if self.output_path is None:
            self.output_path = os.getcwd()
        else:
            os.makedirs(self.output_path, exist_ok=True)
        self.tx_file = f"{self.output_path}/{fname}"
        with open(self.tx_file, "w") as j:
            if tx is None:
                json.dump(self.tx_data, j)
            else:
                json.dump(tx, j)
        self.log.info(f"TX data saved as json file: {self.tx_file}")
        return self.tx_file

    def continuous_data(self):
        """Store continuous feature vectors in a variable using the column names (or axis index if using numpy arrays) from
        `cols` attribute.

        Returns
        -------
        dataframe or ndarray
            continuous feature vectors (as determined by `cols` attribute)
        """
        if self.cols is None:
            self.log.debug("`cols` attribute not instantiated.")
            return None
        if isinstance(self.data, pd.DataFrame):
            return self.data[self.cols]
        elif isinstance(self.data, np.ndarray):
            return self.data[:, self.cols]

    def categorical_data(self):
        """Stores the other feature vectors in a separate variable (any leftover from `data` that are not in `cols`).

        Returns
        -------
        dataframe or ndarray
            "categorical" i.e. non-continuous feature vectors (as determined by `cols` attribute)
        """
        if self.cols is None:
            return None
        if isinstance(self.data, pd.DataFrame):
            return self.data.drop(self.cols, axis=1, inplace=False)
        elif isinstance(self.data, np.ndarray):
            allcols = list(range(self.data.shape[1]))
            cat_cols = [c for c in allcols if c not in self.cols]
            return self.data[:, cat_cols]

    def normalized_dataframe(self, normalized):
        """Creates a new dataframe with the normalized data. Optionally combines with non-continuous vectors (original data) and
        appends `_scl` to the original column names for the ones that have been transformed.

        Parameters
        ----------
        normalized : dataframe
            normalized feature vectors
        join_data : bool, optional
            merge back with the original non-continuous data, by default True
        rename : bool, optional
            append '_scl' to normalized column names, by default True

        Returns
        -------
        dataframe
            dataframe of same shape as input data with continuous features normalized
        """
        try:
            idx = self.data.index
        except AttributeError:
            self.log.error(
                "Non-dataframe type detected - Trying `normalized_matrix` instead."
            )
            return self.normalized_matrix(normalized)
        if self.rename is None:
            newcols = self.cols
        elif isinstance(self.rename, str):
            newcols = [c + self.rename for c in self.cols]
        elif isinstance(self.rename, list):
            newcols = self.rename
        try:
            data_norm = pd.DataFrame(normalized, index=idx, columns=newcols)
            if self.join_data == 1:
                data_norm = data_norm.join(self.categorical, how="left")
            elif self.join_data == 2:
                data_norm = data_norm.join(self.data, how="left")
            return data_norm
        except Exception as e:
            self.log.error(e)
            return None

    def normalized_matrix(self, normalized):
        """Concatenates arrays of normalized data with original non-continuous data along the y-axis (axis=1).

        Parameters
        ----------
        normalized : numpy.ndarray
            normalized data

        Returns
        -------
        numpy.ndarray
            array of same shape as input data, with continuous vectors normalized
        """
        if isinstance(self.categorical, pd.DataFrame):
            cat = self.categorical.values
        else:
            cat = self.categorical
        return np.concatenate((normalized, cat), axis=1)

    def normalizeX(self, normalized):
        """Combines original non-continuous features/vectors with the transformed/normalized data. Determines datatype (array or
        dataframe) and calls the appropriate method.

        Parameters
        ----------
        normalized : dataframe or ndarray
            normalized data
        join_data : bool, optional
            merge back with non-continuous data, by default True
        rename : bool, optional
            append '_scl' to normalized column names, by default True

        Returns
        -------
        ndarray or dataframe
            array or dataframe of same shape and datatype as inputs, with continuous vectors/features normalized
        """
        if isinstance(self.data, pd.DataFrame):
            return self.normalized_dataframe(normalized)
        elif isinstance(self.data, np.ndarray):
            return self.normalized_matrix(normalized)
        else:
            self.log.error(
                "Input data type not recognized - must be a dataframe or array"
            )
            return None


class PowerX(Transformer):
    """Applies Leo-Johnson PowerTransform (via scikit learn) normalization and scaling to continuous feature vectors of a
    dataframe or numpy array. The `tx_data` attribute can be instantiated from a json file, dictionary or the input data itself.
    The training and test sets should be normalized separately (i.e. distinct class objects) to prevent data leakage when
    training a machine learning model. Loading the transform metadata from a json file allows you to transform a new input array
    (e.g. for predictions) without needing to access the original dataframe.

    Parameters
    ----------
    data : dataframe or numpy.ndarray
        input data containing continuous feature vectors to be transformed (may also contain vectors or columns of
        categorical and other datatypes as well).
    cols : list of str or int
        column names or array index values of feature vectors to be transformed (i.e. continuous datatype features)
    ncols : list of int, optional
        array index values of feature vectors to be transformed, by default None
    tx_data : dict, optional
        transform metadata (lambdas, mus, and sigmas) calculated previously to be reused during this instantiation, by default None
    tx_file : string, optional
        path to saved transformer metadata, by default None
    save_tx : bool, optional
        save the transformer metadata as json file on local disk, by default True
    output_path : string, optional
        where to save the transformer metadata, by default None (current working directory)
    join_data : int, optional
        1: join normalized data with remaining columns of original; 2: join with complete original, all columns (requires
        renaming), by default 1
    rename : str or list
        if string, will be appended to normalized col names; if list, will rename normalized columns in this order, by default _scl
    """
    def __init__(
        self,
        data,
        cols,
        ncols=None,
        tx_data=None,
        tx_file=None,
        save_tx=False,
        save_as="tx_data.json",
        output_path=None,
        join_data=1,
        rename="_scl",
        **log_kws,
    ):
        super().__init__(
            data,
            cols=cols,
            ncols=ncols,
            tx_data=tx_data,
            tx_file=tx_file,
            save_tx=save_tx,
            join_data=join_data,
            rename=rename,
            output_path=output_path,
            name="PowerX",
            **log_kws,
        )
        self.fname = save_as
        self.calculate_power()
        self.normalized = self.apply_power_matrix()
        self.Xt = super().normalizeX(self.normalized)

    def fitX(self):
        """Instantiates a scikit-learn PowerTransformer object and fits to the input data. If `tx_data` was passed as a kwarg or
        loaded from `tx_file`, the lambdas attribute for the transformer object will be updated to use these instead of
        calculated at the transform step.

        Returns
        -------
        PowerTransformer object
            transformer fit to the data
        """
        self.transformer = PowerTransformer(standardize=False).fit(self.continuous)
        self.transformer.lambdas_ = self.get_lambdas()
        return self.transformer

    def get_lambdas(self):
        """Instantiates the lambdas from file or dictionary if passed as kwargs; otherwise it uses the lambdas calculated in the
        transformX method. If transformX has not been called yet, returns None.

        Returns
        -------
        ndarray or float
            transform of multiple feature vectors returns an array of lambda values; otherwise a single vector returns a single
            (float) value.
        """
        if self.tx_data is not None:
            return self.tx_data["lambdas"]
        return self.transformer.lambdas_

    def transformX(self):
        """Applies a scikit-learn PowerTransform on the input data.

        Returns
        -------
        ndarray
            continuous feature vectors transformed via scikit-learn PowerTransform
        """
        return self.transformer.transform(self.continuous)

    def calculate_power(self):
        """Fits and transforms the continuous feature vectors using scikit learn PowerTransform. Calculates zero mean and unit
        variance for each vector as a separate step and stores these along with the lambdas in a dictionary `tx_data` attribute.
        This is so that the same normalization can be applied later for prediction inputs without requiring the original training
        data - otherwise it would be the same as using PowerTransform(standardize=True). Optionally, the calculated transform
        data can be stored in a json file on local disk.

        Returns
        -------
        self
            spacekit.preprocessor.transform.PowerX object with transformation metadata calculated for the input data and stored
            as attributes.
        """
        self.transformer = self.fitX()
        self.input_matrix = self.transformX()
        if self.tx_data is None:
            mu, sig = [], []
            for i in range(len(self.cols)):
                # normalized[:, i] = (v - m) / s
                mu.append(np.mean(self.input_matrix[:, i]))
                sig.append(np.std(self.input_matrix[:, i]))
            self.tx_data = {
                "lambdas": self.get_lambdas(),
                "mu": np.asarray(mu),
                "sigma": np.asarray(sig),
            }
            if self.save_tx is True:
                tx2 = {}
                for k, v in self.tx_data.items():
                    tx2[k] = list(v)
                _ = super().save_transformer_data(tx=tx2, fname=self.fname)
                del tx2
        return self

    def apply_power_matrix(self):
        """Transforms the input data. This method assumes we already have `tx_data` and a fit-transformed input_matrix (array of
        continuous feature vectors), which normally is done automatically when the class object is instantiated and
        `calculate_power` is called.

        Returns
        -------
        ndarray
            power transformed continuous feature vectors
        """
        xrow = self.continuous.shape[0]
        xcol = self.continuous.shape[1]
        self.normalized = np.empty((xrow, xcol))
        for i in range(xcol):
            v = self.input_matrix[:, i]
            m = self.tx_data["mu"][i]
            s = self.tx_data["sigma"][i]
            self.normalized[:, i] = np.round((v - m) / s, 5)
        return self.normalized


def normalize_training_data(
    df, cols, X_train, X_test, X_val=None, rename=None, output_path=None
):
    """Apply Leo-Johnson PowerTransform (via scikit learn) normalization and scaling to the training data, saving the transform
    metadata to json file on local disk and transforming the train, test and val sets separately (to prevent data leakage).

    Parameters
    ----------
    df : pandas dataframe
        training dataset
    cols: list
        column names or array index values of feature vectors to be transformed (i.e. continuous datatype features)
    X_train : ndarray
        training set feature inputs array
    X_test : ndarray
        test set feature inputs array
    X_val : ndarray, optional
        validation set inputs array, by default None

    Returns
    -------
    ndarrays
        normalized and scaled training, test, and validation sets
    """
    print("Applying Normalization (Leo-Johnson PowerTransform)")
    ncols = [i for i, c in enumerate(df.columns) if c in cols]
    Px = PowerX(
        df, cols=cols, ncols=ncols, save_tx=True, rename=rename, output_path=output_path
    )
    X_train = PowerX(
        X_train, cols=cols, ncols=ncols, rename=rename, tx_data=Px.tx_data
    ).Xt
    X_test = PowerX(
        X_test, cols=cols, ncols=ncols, rename=rename, tx_data=Px.tx_data
    ).Xt
    if X_val is not None:
        X_val = PowerX(
            X_val, cols=cols, ncols=ncols, rename=rename, tx_data=Px.tx_data
        ).Xt
        return X_train, X_test, X_val
    else:
        return X_train, X_test


def normalize_training_images(X_tr, X_ts, X_vl=None):
    """Scale image inputs so that all pixel values are converted to a decimal between 0 and 1 (divide by 255).

    Parameters
    ----------
    X_tr : ndarray
        training set images
    test : ndarray
        test set images
    val : ndarray, optional
        validation set images, by default None

    Returns
    -------
    ndarrays
        image set arrays
    """
    X_tr /= 255.0
    X_ts /= 255.0
    if X_vl is not None:
        X_vl /= 255.0
        return X_tr, X_ts, X_vl
    else:
        return X_tr, X_ts


def array_to_tensor(arr, reshape=False, shape=(-1, 1)):
    if isinstance(arr, tf.Tensor):
        return arr
    if reshape is True:
        arr = arr.reshape(shape[0], shape[1])
    return tf.convert_to_tensor(arr, dtype=tf.float32)


def y_tensors(y_train, y_test, reshape=True):
    y_train = array_to_tensor(y_train, reshape=reshape)
    y_test = array_to_tensor(y_test, reshape=reshape)
    return y_train, y_test


def X_tensors(X_train, X_test):
    X_train = array_to_tensor(X_train)
    X_test = array_to_tensor(X_test)
    return X_train, X_test


def arrays_to_tensors(X_train, y_train, X_test, y_test, reshape_y=False):
    """Converts multiple numpy arrays into tensorflow tensor datatypes at once (for convenience).

    Parameters
    ----------
    X_train : ndarray
        input training features
    y_train : ndarray
        training target values
    X_test : ndarray
        input test features
    y_test : ndarray
        test target values

    Returns
    -------
    tensorflow.tensors
        X_train, y_train, X_test, y_test
    """
    X_train = array_to_tensor(X_train)
    y_train = array_to_tensor(y_train, reshape=reshape_y)
    X_test = array_to_tensor(X_test)
    y_test = array_to_tensor(y_test, reshape=reshape_y)
    return X_train, y_train, X_test, y_test


def tensor_to_array(tensor, reshape=False, shape=(-1, 1)):
    """Convert a tensor back into a numpy array. Optionally reshape the array (e.g. for target class data).

    Parameters
    ----------
    tensor : tensor
        tensorflow tensor object
    reshape : bool, optional
        reshapes the array (-1, 1) using numpy, by default False

    Returns
    -------
    ndarray
        array of same shape as input tensor, unless reshape=True
    """
    if reshape:
        return np.asarray(tensor).reshape(shape[0], shape[1])
    else:
        return np.asarray(tensor)


def tensors_to_arrays(X_train, y_train, X_test, y_test):
    """Converts tensors into arrays, which is necessary for certain regression analysis computations. The y_train and y_test args
    are reshaped using numpy.reshape(-1, 1).

    Parameters
    ----------
    X_train : tensor
        training feature inputs
    y_train : tensor
        training target outputs
    X_test : tensor
        test feature inputs
    y_test : tensor
        test target outputs

    Returns
    -------
    numpy.ndarrays
        X_train, y_train, X_test, y_test
    """
    X_train = tensor_to_array(X_train)
    y_train = tensor_to_array(y_train, reshape=True)
    X_test = tensor_to_array(X_test)
    y_test = tensor_to_array(y_test, reshape=True)
    return X_train, y_train, X_test, y_test


def hypersonic_pliers(
    path_to_train, path_to_test, y_col=[0], skip=1, dlm=",", encoding='bytes', subtract_y=0.0, reshape=False
):
    """Extracts data into 1-dimensional arrays, using separate target classes (y) for training and test data. Assumes y (target)
    is first column in dataframe. If the target (y) classes in the raw data are 0 and 2, but you'd like them to be binaries (0
    and 1), set subtract_y=1.0

    Parameters
    ----------
    path_to_train : string
        path to training data file (csv)
    path_to_test : string
        path to test data file (csv)
    y_col : list, optional
        axis index of target class, by default [0]
    skip : int, optional
        skiprows parameter for np.loadtxt, by default 1
    dlm : str, optional
        delimiter, by default ","
    encoding: str, optional
        explicitly passed encoding type to numpy.loadtxt, by default bytes
    subtract_y : float, optional
        subtract this value from all y-values, by default 1.0

    Returns
    -------
    np.ndarrays
        X_train, X_test, y_train, y_test
    """
    Train = np.loadtxt(path_to_train, skiprows=skip, delimiter=dlm, encoding=encoding)
    cols = list(range(Train.shape[1]))
    xcols = [c for c in cols if c not in y_col]
    X_train = Train[:, xcols]
    y_train = Train[:, y_col, np.newaxis] - subtract_y

    Test = np.loadtxt(path_to_test, skiprows=skip, delimiter=dlm, encoding=encoding)
    X_test = Test[:, xcols]
    y_test = Test[:, y_col, np.newaxis] - subtract_y
    if reshape is True:
        y_train = y_train.reshape(y_train.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)

    del Train, Test
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)

    return X_train, X_test, y_train, y_test


def thermo_fusion_chisel(matrix1, matrix2=None):
    """Scales each vector of a 2d array (``matrix``) to zero mean and unit variance. The second (optional) matrix is to perform
    the same scaling on a separate set of inputs, e.g. train and test data. Note - normalization should be done separately to
    prevent data leakage in model training, hence the matrix2 kwarg.

    Parameters
    ----------
    matrix1 : ndarray
        input feature vectors to be scaled
    matrix2 : ndarray, optional
        second input feature vectors to be scaled, by default None

    Returns
    -------
    ndarray(s)
        scaled array(s) of same shape as input
    """
    matrix1 = (matrix1 - np.mean(matrix1, axis=1).reshape(-1, 1)) / np.std(
        matrix1, axis=1
    ).reshape(-1, 1)

    print("Mean: ", matrix1[0].mean())
    print("Variance: ", matrix1[0].std())

    if matrix2 is not None:
        matrix2 = (matrix2 - np.mean(matrix2, axis=1).reshape(-1, 1)) / np.std(
            matrix2, axis=1
        ).reshape(-1, 1)

        print("Mean: ", matrix2[0].mean())
        print("Variance: ", matrix2[0].std())
        return matrix1, matrix2
    else:
        return matrix1


def babel_fish_dispenser(matrix1, matrix2=None, step_size=None, axis=2):
    """Adds an input corresponding to the running average over a set number of time steps. This helps the neural network to
    ignore high frequency noise by passing in a uniform 1-D filter and stacking the arrays.

    Parameters
    ----------
    matrix1 : numpy array
        e.g. X_train
    matrix2 : numpy array, optional
        e.g. X_test, by default None
    step_size : int, optional
        timesteps for 1D filter (e.g. 200), by default None
    axis : int, optional
        which axis to stack the arrays, by default 2

    Returns
    -------
    numpy array(s)
        2D array (original input array with a uniform 1d-filter as noise)
    """
    if step_size is None:
        step_size = 200

    # calc input for flux signal rolling avgs
    filter1 = uniform_filter1d(matrix1, axis=1, size=step_size)
    # store in array and stack on 2nd axis for each obs of X data
    matrix1 = np.stack([matrix1, filter1], axis=axis)

    if matrix2 is not None:
        filter2 = uniform_filter1d(matrix2, axis=1, size=step_size)
        matrix2 = np.stack([matrix2, filter2], axis=axis)
        print(matrix1.shape, matrix2.shape)
        return matrix1, matrix2
    else:
        print(matrix1.shape)
        return matrix1


def fast_fourier(matrix, bins):
    """Takes an array (e.g. signal input values) and rotates number of ``bins`` to the left as a fast Fourier transform. Returns
    vector of length equal to ``matrix`` input array.

    Parameters
    ----------
    matrix : ndarray
        input values to transform
    bins : int
        number of rotations

    Returns
    -------
    ndarray
        vector of length equal to ``matrix`` input array
    """
    shape = matrix.shape
    fourier_matrix = np.zeros(shape, dtype=float)

    for row in matrix:
        signal = np.asarray(row)
        frequency = np.arange(signal.size / 2 + 1, dtype=np.float)
        phase = np.exp(
            complex(0.0, (2.0 * np.pi)) * frequency * bins / float(signal.size)
        )
        ft = np.fft.irfft(phase * np.fft.rfft(signal))
        fourier_matrix += ft
    return fourier_matrix
