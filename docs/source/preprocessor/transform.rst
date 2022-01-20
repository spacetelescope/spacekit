.. _transform:

*******************************
spacekit.preprocessor.transform
*******************************

.. currentmodule:: spacekit.preprocessor.transform

.. inheritance-diagram:: spacekit.preprocessor.transform
   :parts: 2

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
.. code-block:: python
   
   Px_from_df = PowerX(df, cols=["numexp", "rms_ra", "rms_dec", "nmatches", "point", "segment", "gaia"], save_tx=True)
   Px_from_data = PowerX(xtrain, cols=PX.cols, tx_data=PX.tx_data)
   Px_from_file = PowerX(xtrain, cols=PX.cols, tx_file=PX.tx_file)
   Px_from_array = PowerX(np.array([X]), cols=[0,1,2,3,4,5,6])