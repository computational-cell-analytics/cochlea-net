# Data transfer from and to UKON

UKON is the SMB storage of the UMG imaging data. The command `flamingo_tools.transfer_data` and the scripts in
`scripts/data_transfer/` copy data between UKON and a local machine or the cluster with `smbclient`. These tools need
`smbclient` and run on Linux or on the cluster.

| Share | Address |
|---|---|
| UKON100 | `//wfs-medizin.top.gwdg.de/ukon-all$/ukon100` |
| UKON_spezial | `//wfs-medizin-spezial.top.gwdg.de/ukon-all$` |

The tools connect to UKON_spezial. Select another share with `-s`.
Each tool asks for your GWDG password and then starts.

The connection to UKON is unstable. The tools retry each transfer unit after a disconnect, without a new password
prompt.

For the transfer on the cluster and the transfer to the S3 bucket, see the
[data transfer documentation of slurm-workflows](https://github.com/computational-cell-analytics/slurm-workflows/blob/main/doc/hpc_02_data_transfer.md).

## Remote paths

- Put a remote path with backslashes in quotation marks. Without them, the shell removes every backslash:
  `UKON100\archiv\imaging` arrives as `UKON100archivimaging`. The tools warn about a path without a separator before
  they ask for the password.
- Forward slashes need no quotation marks: `-p UKON100/archiv/imaging/...` works too.
- A leading backslash is optional. `\UKON100\archiv` and `UKON100\archiv` name the same directory.
- The tools check the login and the remote directory before any transfer. They stop with an error if either check
  fails.
- A failed login stops the tool at once, without a retry. Repeated failed logins can possibly lock the GWDG account.

## Errors and the log file

The tools append each transfer unit that still fails after all retries to a log file. The default log file is
`transfer_log.txt` in the output directory. Select another file with `-l`.

A tool exits with code 1 when its log file grew: a transfer unit failed, or expected data was not found. Run the same
command again: a repeated transfer overwrites the existing files.

## Copy an N5 dataset, a directory or a file

`flamingo_tools.transfer_data` copies an N5 dataset, any other directory, or a single file. `-p` is the parent directory
on the share, `-d` is the name of the data in it.

**Example** (download an N5 dataset):
```bash
UKON_FOLDER="UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\<year>\<experimenter>\M_AMD_00C202_L\3_fused"
flamingo_tools.transfer_data \
    -u <GWDG-username> -p "$UKON_FOLDER" -d MAMD_C202_PV_CR_Lypd1_fused.n5 -o /local/dest/dir
```

The data lands in `<output_dir>/<name>`. The default output directory is the current directory.

### N5 datasets

An N5 dataset has top-level `setupN` directories. The command detects it automatically.
- Phase 1 copies the dataset in a single `smbclient` session.
- After a disconnect, phase 2 copies each piece on its own: one piece per chunk directory for the scales `s0`-`s3`,
  one piece per scale for the other scales. Each piece is retried.
- At the end, the command fetches each chunk file again that is too small to be valid.

`--setup 0 1` restricts the transfer to `setup0` and `setup1`.

### Other directories

A directory without top-level `setupN` directories, for example a folder with a few very large files, is copied
**one file at a time** after a disconnect. Each file is retried on its own.
- At the end, the command compares the size of each file with the share. It copies each missing file or each file with
  a different size again.
- `--generic` forces this mode for an N5 dataset too. `--generic` and `--setup` are mutually exclusive.
- File names with spaces work. Empty directories are not created again on download.

### Single files

When `-d` names a file, the command copies this file. A download that stops part way through a large file continues
where it stopped (`smbclient reget`), instead of starting again. At the end, the command compares the file size with the
share. `--setup` does not apply to a file.

The download continues an existing local file of the same name without a check of its content. Delete the local file
first if the file on the share changed.

### Upload (ingest)

`--ingest` uploads local data to the share. `-o` is then the local **parent** directory: `<output_dir>/<name>` must
exist.

**Example**:
```bash
UKON_FOLDER="UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\MartinS\forMartin"
flamingo_tools.transfer_data \
    --ingest -u <GWDG-username> -p "$UKON_FOLDER" -d my_cochlea.n5 -o /local/parent/dir
```

- The upload of a directory uses the same two phases as the download. At the end, the command compares the size of
  each file with the local file, and uploads each missing file or each file with a different size again.
- An upload of a single file always starts again after a disconnect. `smbclient` cannot continue an upload.
- `-p` must exist on the share. The command stops otherwise, so a typo cannot create stray folders on the shared server.
  `--create-parents` creates the missing parent directories.
- `--setup` and `--generic` work as for the download.

## Copy the stitching data

`smb_transfer_converted_stitching.py` downloads the input for BigStitcher from a converted-data folder (usually
`2_converted_stitching`) without the full-resolution image data.

**Example**:
```bash
UKON_FOLDER="UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2026\Aleyna\00_LSFM_cosynapse\M_AMD_000117_L"
python /path/to/cochlea-net/scripts/data_transfer/smb_transfer_converted_stitching.py \
    -u <GWDG-username> -p "$UKON_FOLDER" -d 2_converted_stitching -o /local/dest/dir
```

- The data lands in `<output_dir>/2_converted_stitching`, with the same layout as on the share.
- The script copies all files whose name contains `xml` (`*.xml` and the backups `*.xml~N`). It stops before the image
  transfer when no `*.xml` file arrives, for example when `-d` names the wrong folder.
- `interestpoints.n5` is copied in full. Then the script compares the size of each file with the share.
- Every other `*.n5` folder is image data. Only the scale levels from `--min_scale` (default 2) onwards are copied, so
  `s0` and `s1` stay on the share.
- The image transfer copies one piece per chunk directory for `s0`-`s3` and one piece per scale for the other scales,
  with the same retries as `flamingo_tools.transfer_data`.

## List the raw data

`smb_list_raw.py` writes the file names of a raw-data folder to a JSON file. It downloads nothing.
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
Files directly in the raw-data folder are not listed. A file in a deeper subfolder keeps its path relative to the stain
folder.

## Convert raw data over an unstable connection

You do not need to transfer the raw data first in order to convert it. `flamingo_tools.convert_data` reads the tiles
one z-slab at a time, retries every read, and records its progress. A lost connection costs one slab instead of the
whole run.

**From a mounted share** (works on Windows, where `smbclient` is not available):
```bash
flamingo_tools.convert_data -i G301L/ --out_path G_LR_000301_L.n5 --file_ext .raw
```
Run the same command again after an interruption. It continues where it stopped.
- `--dry_run` checks the input first. It reports the file pairing and the tile shapes in seconds.
- `--slab_memory` (GB, default 1.0) limits the memory per read.
- `--stage_dir` copies each file to local storage before the conversion.
- `--restart` discards an existing output.

**Directly from the share with `smbclient`** (Linux or the cluster). This needs `--stage_dir`, and enough free disk
for one raw file:
```bash
flamingo_tools.convert_data -u <GWDG-username> \
    -i "\UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\<...>\G301L" \
    --out_path G_LR_000301_L.n5 --file_ext .raw --stage_dir ./stage
```

The conversion appends failed reads and transfers to `<out_path>.convert_log.txt`.
