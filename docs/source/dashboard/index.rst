.. _dashboard:

******************
spacekit.dashboard
******************

.. currentmodule:: spacekit.dashboard

.. toctree::
   :maxdepth: 1

   cal <cal.rst>
   svm <svm.rst>


Pull Existing Docker Image
--------------------------

Pull the most recent training data and model results from docker. For example, to get the latest HST cal (calcloud) data, the command is:

.. code:: bash

   docker pull alphasentaurii/spacekit:dash-cal-latest


Custom Configurations
---------------------

*Configuring Custom Datasets via Environment file*

The variables below are used by `spacekit.datasets.beam`` to find specific datasets. Using the defaults will pull in the 3 most recent dataset iterations and model training results. To configure the dashboard to use other datasets, you'll need to set some configuration options. Copy variables into the `.env` file located in `docker/images/dashboard_image` - feel free to use one of the templates (`spacekit/docker/images/dashboard_image/templates`) then customize further as desired.

.. code:: bash

   # pkg, s3, web, file
   SRC="web"
   # collection, bucketname, repo url, or local path
   COLLECTION="calcloud" # e.g. "svm", "calcloud", "myS3bucket"
   # used by spacekit.datasets as dictionary keys
   DATASETS="2022-02-14,2021-11-04,2021-10-28"
   # for s3 use the names of the .zip files 
   DATASETS="2022-02-14-1644848448,2021-11-04-1636048291,2021-10-28-1635457222"
   # for s3 this is the folder prefix
   PFX="archive"


Importing data from S3 (aws)
----------------------------

.. code:: bash
   
   SRC=s3
   COLLECTION=mybucket
   PFX=somefolder


Mounting local data
-------------------

You can also have your data in a local directory, and just bind mount the folder when you go to launch the container, or set container mode to "-it" and use spacekit.datasets to get the data before launching the dashboard. 

.. code:: bash

   CONTAINER_MODE="-d" # -d for detached, -it for interactive
   MOUNTS=1 # >0 will bind mount the below source and dest paths
   SOURCEDATA="/path/to/datasets"
   DESTDATA="/home/developer/data"


Build the image
---------------

Once you have variables set in the .env file, build the image:

.. code:: bash

   $ cd spacekit
   $ sh scripts/build.sh


Run the container
-----------------

Launch a container with your brand new image then fire it up in a browser: `http://0.0.0.0:8050/`

.. code:: bash

   $ sh scripts/launch.sh
   # you should see a SHA like: "6cb2bee87fbef53f44686"

