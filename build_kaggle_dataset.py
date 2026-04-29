import os
import zipfile

def zip_directory(folder_path, zip_file, root_dir):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Exclude __pycache__ and compiled files
            if '__pycache__' in file_path or file.endswith('.pyc') or file.endswith('.pyo'):
                continue
            arcname = os.path.relpath(file_path, root_dir)
            zip_file.write(file_path, arcname)

if __name__ == "__main__":
    output_zip = "haic_kaggle_utils.zip"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    tools_dir = os.path.join(script_dir, "tools")
    viability_dir = os.path.join(script_dir, "viability")
    
    print(f"Creating {output_zip}...")
    with zipfile.ZipFile(os.path.join(script_dir, output_zip), 'w', zipfile.ZIP_DEFLATED) as zipf:
        if os.path.exists(tools_dir):
            print(f"Adding tools directory...")
            zip_directory(tools_dir, zipf, script_dir)
        else:
            print("Warning: tools directory not found.")
            
        if os.path.exists(viability_dir):
            print(f"Adding viability directory...")
            zip_directory(viability_dir, zipf, script_dir)
        else:
            print("Warning: viability directory not found.")
            
    print(f"Successfully packaged {output_zip}")
