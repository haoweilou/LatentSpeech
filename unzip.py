import tarfile

# Define the path to the input and output files
input_tar_bz2_file  = './LJSpeech-1.1.tar.bz2'
output_directory = './LJSpeech'

with tarfile.open(input_tar_bz2_file, 'r:bz2') as tar:
    # Extract all contents to the specified output directory
    tar.extractall(path=output_directory)

print(f'Extracted {input_tar_bz2_file} to {output_directory}')