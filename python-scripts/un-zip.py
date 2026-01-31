
# Code to unzip a large number of zip files in a folder. 
# Create a new folder for each zip file, with the same name as the zip file.
import os
import zipfile

def unzip_files_in_folder (input_folder_path, output_folder_path):
    # Iterate through all files in the given folder
    for item in os.listdir (input_folder_path):
        if item.endswith('.zip'):
            # print(item)
            file_path = os.path.join(input_folder_path, item)
            # print(file_path)
            # Create a new folder with the same name as the zip file (without .zip extension)
            new_folder_name = os.path.splitext(item)[0]
            # print(new_folder_name)
            new_folder_path = os.path.join(output_folder_path, new_folder_name)
            # print(new_folder_path)
            # os.makedirs (new_folder_path, exist_ok=True) # Creates a double level.
            # Unzip the file into the new folder
            with zipfile.ZipFile (file_path, 'r') as zip_ref:
                zip_ref.extractall(output_folder_path)
            print(f'Unzipped {item} to folder {new_folder_path}')

if __name__ == "__main__":
    folder_to_unzip_from = "C:\\Computer-Vision-Systems\\zip-files" 
    folder_to_unzip_to   = "C:\\Computer-Vision-Systems\\datasets"
    unzip_files_in_folder (folder_to_unzip_from, folder_to_unzip_to)

