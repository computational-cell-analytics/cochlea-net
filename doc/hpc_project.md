## Getting started

Connect to the cluster using an SSH connection. Follow the instructions in the [GWDG HPC docs](https://docs.hpc.gwdg.de/start_here/connecting/index.html).

## Install micromamba
Micromamba can be used to manage different environments for executing scripts.
It is a good alternative to the `conda` environment. Follow the instructions in the [docs](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
This command should be sufficient to install the latest version.
```bash
curl -Ls https://micro.mamba.pm/api/micromamba/$(uname)-$(uname -m)/latest | tar -xvj bin/micromamba
```

## Create micromamba environment for computing

The environment we want to use should support a GPU. It is easier to install it in an interactive environment with an access to GPU resources to install the correct packages which support the GPU infrastructure.
1) Start an interactive environment:
```bash
# an interactive node (preferred)
srun -p grete:interactive -G 1g.10gb -c 4 --constraint=inet -t 01:00:00 --pty bash
# a shared GPU
srun -p grete:shared -G A100:1 --constraint=inet --pty -n 1 -c 4 -t 01:00:00 bash
```
2) Clone the [µSAM](https://github.com/computational-cell-analytics/micro-sam) package, which installs most dependencies of `cochlea-net`:
```bash
# clone cochlea-net repository
git clone git@github.com:computational-cell-analytics/micro-sam.git
# install new micromamba environment
micromamba create -f environment.yaml -y
# activate the environment
micromamba activate sam
```

## Install `cochlea-net` and `mobie` functionality

Install the functionality for [CochleaNet](https://github.com/computational-cell-analytics/cochlea-net) and [MoBIE utils](https://github.com/mobie/mobie-utils-python)

```bash
# clone cochlea-net repository
git clone git@github.com:computational-cell-analytics/cochlea-net.git
# move to the cloned directory
cd cochlea-net
# update
micromamba env update -n sam --file environment.yaml -y
# install CLI functions, e.g. flamingo_tools.label_components
pip install -e .

# install MoBIE functionality
pip install mobie_utils
```
Alternative installation from [MoBIE-Python-utils repository](https://github.com/mobie/mobie-utils-python)
```bash
# do the same for MoBIE
git clone git@github.com:mobie/mobie-utils-python.git
cd mobie-utils-python
pip install -e .
```

## Use screens

Screens are a way to run processes without requiring an active connection to the cluster.

Create and manage screens using `tmux`, e.g.
```bash
# create a screen
tmux new -s 0
# resume a screen
tmux a -t 0
```
Detach from a screen: `Ctrl+b-->d`

## Transfer data from UKON100

It is helpful to look around the UKON100 and transfer single files. You can connect to UKON using:
```bash
# connect to UKON
smbclient //wfs-medizin.top.gwdg.de/ukon-all$/ukon100 -U GWDG/<gwdg_username>
# connect to UKON_spezial
smbclient //wfs-medizin-spezial.top.gwdg.de/ukon-all$ -U GWDG/<gwdg_username>
```
Once there, you can use
```bash
# toggle recursive copy of files
recurse
# toggle prompt for file transfer
prompt
# copy file to local directory
mget <UKON_file/directory>
# upload file to UKON100
mpu <local_file_name>
```
Because the connection is unstable, you can transfer the files with this script:
Example:
```bash
python ~/flamingo-tools/scripts/data_transfer/smb_transfer_resilient.py --username <gwdg_username> --remote_parent_dir "UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2026\Aleyna\M_AMD_00C202_L\3_fused" --remote_data MAMD_C202_PV_CR_Lypd1_fused.n5 -o .
```
You will be prompted to enter your GWDG password. Afterwards, the transfer process should start.

## Add a dataset to the S3 bucket
MoBIE manages the project with a `project.json` file. However, the local version only has the datasets, which were added to this MoBIE instance while the version of the S3 buckets is a collection of all datasets, which may come from different local MoBIE projects. Therefore, it is essential to not overwrite the external S3 bucket `project.json` but instead add the datasets to it and update it.
```bash
# copy the external project.json
rclone copyto cochlea-lightsheet:cochlea-lightsheet/project.json project_remote.json
# add the dataset to project_remote.json, e.g.
vim project_remote.json
# upload the edited file
rclone copyto project_remote.json cochlea-lightsheet:cochlea-lightsheet/project.json
```