.. _transform:

*******************************
spacekit.preprocessor.transform
*******************************

.. currentmodule:: spacekit.preprocessor.transform

.. toctree::
   :maxdepth: 1

.. inheritance-diagram:: spacekit.preprocessor.transform
   :parts: 3

.. autoclass:: SkyTransformer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Transformer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PowerX
   :members:
   :undoc-members:
   :show-inheritance:

PowerX Examples
^^^^^^^^^^^^^^^

*Calculate the normalization parameters of a dataframe ("training set") using the Leo-Johnson PowerTransform and save the params to json file on local disk. Use this metadata (``PowerTransform.lambdas_``, mean, and standard deviation for each continuous feature vector) to transform new inputs ("test set") in A) the same session or B) a separate session.*

**Example 1A: Normalize a Dataframe, Apply to Another Dataframe Separately**

.. code-block:: python

   Px = PowerX(df, cols=["numexp", "rms_ra", "rms_dec", "nmatches", "point", "segment", "gaia"], save_tx=True)
   dfX = PowerX(df2, cols=Px.cols, tx_data=PX.tx_data).Xt

**Example 1B: Load saved transform data from json file, apply to new data (separate session)**

.. code-block:: python

   tx_file = "data/tx_data"
   Px = PowerX(df2, cols=["numexp", "rms_ra", "rms_dec"], tx_file=tx_file)
   dfX = Px.Xt

**Example 2: Normalize 2D numpy array (exclude specific axes)**

.. code-block:: python

   # the last 3 columns are encoded categorical features so we exclude these columns
   X = np.asarray([[143.,235.,10.4, 79., 0, 1, 0],[109.,262.,15.9, 63., 1, 0, 1]])
   Px = PowerX(X, cols=[0,1,2,3])
   Xt = Px.Xt


.. autofunction:: normalize_training_data

.. autofunction:: normalize_training_images

.. autofunction:: arrays_to_tensors

.. autofunction:: tensor_to_array

.. autofunction:: tensors_to_arrays

.. autofunction:: hypersonic_pliers

.. autofunction:: babel_fish_dispenser

.. autofunction:: fast_fourier
