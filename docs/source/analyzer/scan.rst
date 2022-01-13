**********************
spacekit.analyzer.scan
**********************

.. currentmodule:: spacekit.analyzer.scan

.. autofunction:: decode_categorical

decoder_key examples:

.. code-block:: python
    
    instrument_key = {"instr": {0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}}
    detector_key = {"det": {0: "hrc", 1: "ir", 2: "sbc", 3: "uvis", 4: "wfc"}}

.. autofunction:: import_dataset

.. autoclass:: MegaScanner
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: CalScanner
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: SvmScanner
    :members:
    :undoc-members:
    :show-inheritance: