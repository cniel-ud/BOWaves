#!/bin/bash

# Script for downloading different datasets
# Use this before activating the virtual / conda environment

# Get the script's directory
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the root directory based on the script's directory
root_dir="$script_dir"

# Define the data directory
data_dir="$root_dir/data"

# Check if directories exist
[[ -e "$root_dir" ]] || ( echo "root dir not found: '$root_dir'"  ; exit 1 )
[[ -e "$data_dir" ]] || ( echo "data dir not found: '$data_dir'"  ; exit 1 )


function ds003004_download() {
    if [[ ! -e $data_dir/ds003004 ]] ; then
        echo Downloading ds003004 ...

        # Doesn't deepcopy. Only provides symlinks to the EEGlab files.
        # Feel free to test the script with the below on your PC though, otherwise it'll download ~36 GB of data
        #pipenv run datalad install -s https://github.com/OpenNeuroDatasets/ds003004.git $data_dir/ds003004

        # Deepcopies. Run on Caviness - nevermind. Datalad is hard to use on caviness,
        # requires creating own VALET package. Just use the curl script
        pipenv run datalad get --copy https://github.com/OpenNeuroDatasets/ds003004.git $data_dir/ds003004
    fi
}

ds003004_download

echo "Done"