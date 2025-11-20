#!/bin/bash

# Check if dvc is installed
if ! command -v dvc &> /dev/null; then
    echo "Please install dvc using pip3 install dvc[s3]"
    exit 1
fi

# Create .dvc directory if it doesn't exist
mkdir -p .dvc

url=$(grep -E "url = s3://ml-cup-dvc" .dvc/config | xargs | cut -d ' ' -f 3)

if [[ -z "$url" ]]; then
    # Generate random 32-character string (alphanumeric)
    random_key=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
    dvc remote modify yandex-storage url "s3://ml-cup-dvc/$random_key"
fi
