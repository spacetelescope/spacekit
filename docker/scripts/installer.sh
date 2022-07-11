envfile=${1:-"/home/developer/dashboard.env"}
source envfile

DEVHOME=/home/developer

if [ "${USEPYPI}" != "1" ]; then
    git clone https://github.com/${SPACEKIT_REPO}.git --single-branch --branch $SPACEKIT_BRANCH
    cd $DEVHOME/spacekit
    pip install -e .
    cd $DEVHOME
else
    pip install spacekit
fi