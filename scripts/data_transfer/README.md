# Data Transfer Moser

## Transfer via smbclient

Current approach to the data transfer:
- Log in to SCC login node:
  $ ssh -i ~/.ssh/id_rsa_scc pape41@transfer-mdc.hpc.gwdg.de
- Go to "/scratch1/projects/cca/data/moser"
- Create subfolder <NAME> for cochlea to be copied
- Log in via $ smbclient \\\\wfs-medizin.top.gwdg.de\\ukon-all\$\\ukon100 -U GWDG\\pape41"
- Go to the folder with the cochlea to copy (cd works)
- Copy the folder via:
    - recurse ON
    - prompt OFF
    - mget *
- Copy this to HLRN by logging into it and running
  $ rsync  pape41:/scratch1/projects/cca/data/moser/<NAME>
  $ rsync -e "ssh -i ~/.ssh/id_rsa_hlrn" -avz pape41@login-mdc.hpc.gwdg.de:/scratch1/projects/cca/data/moser/<NAME> /mnt/lustre-grete/usr/u12086/moser/lightsheet/<NAME>
- Remove on SCC

## Transfer without manual navigation on UKON

The automatic transfer from files to the HPC without the need to navigate within UKON is possible with `smb_transfer.sh`.

**Example**:
```bash
FILE="keppeler-et-al-2021-PNAS.pdf"
UKON_FOLDER="\UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\MartinS\forMartin"
bash /path/to/cochlea-net/scripts/data_transfer/smb_transfer.sh <GWDG-username> "$UKON_FOLDER" "$FILE"
```
You are then prompted to enter your password.
Enter your password and press Enter.
The file transfer should start automatically.

## Remote paths

- Put a remote path with backslashes in quotation marks. Without them, the shell removes every
  backslash: `UKON100\archiv\imaging` arrives as `UKON100archivimaging`. The Python scripts warn
  about a path without a separator before they ask for the password.
- Forward slashes need no quotation marks: `-p UKON100/archiv/imaging/...` works too.
- A leading backslash is optional. `\UKON100\archiv` and `UKON100\archiv` name the same directory.
- `smb_transfer_resilient.py`, `smb_transfer_converted_stitching.py` and `smb_list_raw.py` check
  the login and the remote directory before any transfer, and stop with an error if either fails.
  `smb_transfer_files.py` checks the login only.
- The transfer scripts exit with code 1 when a transfer unit still failed after all retries, or
  when expected data was not found. The log file then lists these units.

## Converting raw data over an unstable connection

You do not need to transfer the raw data first in order to convert it. `flamingo_tools.convert_data`
reads the tiles one z-slab at a time, retries every read, and records its progress, so a lost
connection costs one slab instead of the whole run.

**From a mounted share** (works on Windows, where `smbclient` is not available):
```bash
flamingo_tools.convert_data -i G301L/ --out_path G_LR_000301_L.n5 --file_ext .raw
```
Run the same command again after any interruption; it continues where it stopped. Check the input
first with `--dry_run`, which reports the file pairing and the tile shapes in seconds. Use
`--slab_memory` (GB, default 1.0) to bound the memory per read, `--stage_dir` to copy each file to
local storage before converting it, and `--restart` to discard an existing output.

**Directly from the share via `smbclient`** (Linux or the cluster; needs `--stage_dir`, and enough
free disk for one raw file):
```bash
flamingo_tools.convert_data -u <GWDG-username> \
    -i "\UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\<...>\G301L" \
    --out_path G_LR_000301_L.n5 --file_ext .raw --stage_dir ./stage
```

Failed reads and transfers are appended to `<out_path>.convert_log.txt`.

The shared retry code lives in `flamingo_tools/data_transfer_utils.py`; `smb_transfer_resilient.py`
imports its SMB primitives from there.

## Improvements

Try to automate via https://github.com/jborean93/smbprotocol see `sync_smb.py` for ChatGPT's inital version.

## Transfer Back (ingest)

Upload a locally created N5 dataset (for example the result of a raw-data conversion) back to the
SMB share with the same disconnect resilience as the download. Use `--ingest` on the resilient
transfer script.

**Example**:
```bash
UKON_FOLDER="\UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\MartinS\forMartin"
python /path/to/cochlea-net/scripts/data_transfer/smb_transfer_resilient.py \
    --ingest -u <GWDG-username> -p "$UKON_FOLDER" -d my_cochlea.n5 -o /local/parent/dir
```
You are then prompted for your password.

**Behaviour and options**:
- `-o/--output-dir` is the local **parent** directory that contains the dataset to upload
  (`<output_dir>/<remote_data>` must exist).
- The script uploads a Phase 1 bulk `mput`. On a disconnect it falls back to a per-piece iterative
  upload (per chunk directory for scales `s0`–`s3`, whole scale for the rest), retrying each piece.
- Afterwards it lists the remote tree, compares every file size against the local source, and
  re-uploads any missing or size-mismatched file.
- `--remote_parent_dir` must already exist on the share; the script aborts otherwise so a typo
  cannot create stray folders on the shared server. Pass `--create-parents` to create the parent
  chain automatically.
- `--setup 0 1` restricts the upload to specific setup(s).
- Re-runs are idempotent (`mput`/`put` overwrite, `mkdir` on an existing directory is harmless), so
  an interrupted ingest can simply be re-run.

## Copying arbitrary (non-N5) directories

The script also copies directory trees that are not in N5 format (for example a folder with a few
very large files). Such a directory is detected automatically — it has no top-level `setupN`
directories — and is transferred **one file at a time**, so each file is an independent retry unit.
This works in both directions and needs no extra flag.

**Example** (download a directory tree from the share):
```bash
UKON_FOLDER="\UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\MartinS"
python /path/to/cochlea-net/scripts/data_transfer/smb_transfer_resilient.py \
    -u <GWDG-username> -p "$UKON_FOLDER" -d my_folder -o /local/dest/dir
```

**Behaviour and options**:
- Phase 1 still tries a single bulk transfer. On a disconnect (or non-N5 data) it falls back to the
  per-file transfer, retrying each file independently.
- Verification lists the remote tree and compares **every file size** to the local file, re-fetching
  (download) or re-uploading (ingest) any missing or size-mismatched file. This catches a truncated
  large file that a whole-directory retry can miss.
- Pass `--generic` to force the per-file mode (bypass detection). `--generic` and `--setup` are
  mutually exclusive (`--setup` applies to N5 data only).
- File names with spaces are handled. Empty directories are not recreated on download.
- Re-runs are idempotent, so an interrupted transfer can simply be re-run.

## Copying stitching data

`smb_transfer_converted_stitching.py` downloads the input for BigStitcher from a converted-data
folder (usually `2_converted_stitching`) without the full-resolution image data.

**Example**:
```bash
UKON_FOLDER="UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2026\Aleyna\00_LSFM_cosynapse\M_AMD_000117_L"
python /path/to/cochlea-net/scripts/data_transfer/smb_transfer_converted_stitching.py \
    -u <GWDG-username> -p "$UKON_FOLDER" -d 2_converted_stitching -o /local/dest/dir
```

**Behaviour and options**:
- The data lands in `<output_dir>/2_converted_stitching`, with the same layout as on the share.
- All files whose name contains `xml` are copied (`*.xml` and the backups `*.xml~N`). The script
  stops before the image transfer when no `*.xml` file arrives, for example when `-d` names the
  wrong folder.
- `interestpoints.n5` is copied in full, then every file size is compared to the share.
- Every other `*.n5` folder is treated as image data. Only the scale levels from `--min_scale`
  (default 2) onwards are copied, so `s0` and `s1` stay on the share.
- The image transfer is per chunk directory for `s0`-`s3` and per scale for the rest, with the same
  retries as `smb_transfer_resilient.py`.

## Listing raw data

`smb_list_raw.py` writes the file names of a raw-data folder to a JSON file. Nothing is downloaded.
The raw-data folder contains one subfolder per stain.

**Example**:
```bash
python /path/to/cochlea-net/scripts/data_transfer/smb_list_raw.py \
    -u <GWDG-username> -p "$UKON_FOLDER" -d <raw_data_folder> -o raw_files.json
```

The JSON file holds the raw-data folder name and the sorted file names of each stain folder:
```json
{
  "raw_data": "<raw_data_folder>",
  "stain_folders": {
    "20260409_040345_MAMD_118L_Vglut3_488_PELCOfHC2": ["<file 1>", "<file 2>"]
  }
}
```
Files directly in the raw-data folder are not listed. A file in a deeper subfolder keeps its path
relative to the stain folder.

# Data Transfer Huisken

See "Transfer via smbclient" above:
```
smbclient \\\\wfs-biologie-spezial.top.gwdg.de\\UBM1-all\$\\ -U GWDG\\pape41
```
