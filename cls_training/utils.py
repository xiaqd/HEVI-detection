import os

def scan_files(input_file_path, ext_list):
    file_list = []
    for root, dirs, files in os.walk(input_file_path):
        # scan all files and put it into list
        for f in files:
            if os.path.splitext(f)[1].lower() in ext_list:
                file_list.append(os.path.join(root, f).replace("\\","/").replace(os.path.join(input_file_path, "").replace("\\","/"), "", 1 ))

    return file_list