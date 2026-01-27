#!/bin/bash

helpstr=$(cat <<- EOF
Transfer files from UKON to the GWDG NHR. The password is given via a prompt.

username 			GWDG username, e.g. schilling40
remote_parent_dir	Remote parent directory on the SMB share
remote_file/folder	Remote file/folder on the SMB share
-r output_dir		Output directory. Default: Current directory
EOF
)

OUTPUT_DIR=${PWD}

usage="Usage: $0 [-o output_dir] <username> <remote_parent_directory> <remote_file/folder>"

while getopts "o:h" opt; do
	case $opt in
	o)
		OUTPUT_DIR=$(readlink -f "$OPTARG")
	;;
	h)
		echo "$usage"
		echo
		echo "$helpstr"
		exit 0
	;;
	\?)
		echo "$usage" >&2
		exit 1
	;;
	esac
done

shift $((OPTIND - 1))

if [ $# -lt 3 ] ; then

	echo "$usage" >&2
	exit 1
fi

username=$1
remote_dir=$2
remote_data=$3

# Prompt for password securely
read -r -s -p "Enter password: " password
echo

if [ ! -d "$OUTPUT_DIR" ] ; then
	mkdir -p "$OUTPUT_DIR"
fi

cd "$OUTPUT_DIR" || exit

# Use smbclient to connect and copy the file
echo "Connecting to SMB server and transferring file..."
smbclient "//wfs-medizin.top.gwdg.de/ukon-all$/ukon100" -U "GWDG/$username%$password" << EOF
	cd "$remote_dir"
	ls
	recurse
	prompt
	mget "$remote_data"
	exit
EOF

# Check the exit status
if [ $? -eq 0 ]; then
	echo "File transfer completed successfully."
else
	echo "An error occurred during the file transfer."
fi
