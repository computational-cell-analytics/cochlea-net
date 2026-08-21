# Transfer files

Transfer generic file structures by providing a parent directory and a folder/file to transfer. It's important to use quotation marks because otherwise the parent directory is not read correctly.
Replace the placeholder with the gwdg_username and enter your password when prompted.
```bash
python ~/flamingo-tools/scripts/data_transfer/smb_transfer_resilient.py -p "UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\For_CP\Figure_drafts" -d Revision -u <gwdg_username> --generic -o .
```

# Transfer image data

Der Transfer der Bilddaten kann parallel erfolgen, wenn die Skripte einzeln aufgerufen werden. Der Transfer dauert eine Weile.
```bash
# PV
python ~/flamingo-tools/scripts/data_transfer/smb_transfer_resilient.py -d GLR_301L_PV_fused.n5 -p "UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2026\Lennart\G301L\2_converted_stitching\fused" -u <gwdg_username> -o .
# CTBP2
python ~/flamingo-tools/scripts/data_transfer/smb_transfer_resilient.py -d GLR_301L_CTBP2_fused.n5 -p "UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2026\Lennart\G301L\2_converted_stitching\fused" -u <gwdg_username> -o .
# Vglut3
python ~/flamingo-tools/scripts/data_transfer/smb_transfer_resilient.py -d GLR_301L_Vglut3_fused.n5 -p "UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2026\Lennart\G301L\2_converted_stitching\fused" -u <gwdg_username> -o .
```