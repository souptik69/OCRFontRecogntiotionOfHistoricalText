
import os
import zipfile

# Paths to input and output zip files
input_zip_path = '/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/singlefontdata.zip'
output_zip_path = '/home/hpc/iwi5/iwi5144h/nbb-recipient-detection/SingleMultiple.zip'

# Define the target directories in SingleMultiple.zip
target_dirs = ['test', 'train', 'valid']


def copy_filtered_files(input_zip, output_zip):
    for target_dir in target_dirs:
        target_multiple_dir = os.path.join('SingleMultiple',target_dir, 'multiple')
        os.makedirs(target_multiple_dir, exist_ok=True)

        with zipfile.ZipFile(input_zip, 'r') as input_zip_ref:
            with zipfile.ZipFile(output_zip, 'a') as output_zip_ref:
                jpg_files = [filename for filename in input_zip_ref.namelist() if filename.startswith(target_dir + '/single/') and filename.endswith('.jpg')]
                
                for jpg_file in jpg_files:
                    name, _ = os.path.splitext(jpg_file)
                    # print(name)
                    numerical_name = name.split('/')[-1]
                    # print(numerical_name)
                    txt_path = os.path.join(name + '.txt')
                    cf_path = os.path.join(name + '.cf')
                    pf_path = os.path.join(name + '.pf')
                    
                    # Check if all corresponding files exist before copying
                    if (txt_path in input_zip_ref.namelist() and
                        cf_path in input_zip_ref.namelist() and
                        pf_path in input_zip_ref.namelist() and
                            jpg_file.endswith('.jpg')):
                        output_zip_ref.writestr(os.path.join(target_multiple_dir, numerical_name + '.jpg'), input_zip_ref.read(jpg_file))
                        output_zip_ref.writestr(os.path.join(target_multiple_dir, numerical_name + '.txt'), input_zip_ref.read(txt_path))
                        output_zip_ref.writestr(os.path.join(target_multiple_dir, numerical_name + '.cf'), input_zip_ref.read(cf_path))
                        output_zip_ref.writestr(os.path.join(target_multiple_dir, numerical_name + '.pf'), input_zip_ref.read(pf_path))
# Copy the filtered files to the corresponding directories
copy_filtered_files(input_zip_path, output_zip_path)
