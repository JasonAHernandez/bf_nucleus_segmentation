from ij import IJ, ImagePlus
from ij.plugin import Duplicator
from ij.plugin.filter import BackgroundSubtracter
import os

#remember to change folders at bottom

def process_nd2_files(folder_path, mask_folder, utrack_folder, start_frame=51, end_frame=300,
                      mask_index_formula=lambda x: x * 2):
    # Extract the last folder name from the path
    last_folder_name = os.path.basename(os.path.normpath(folder_path))

    # Create cleaned movies folder
    cleaned_folder = os.path.join(folder_path, "cleaned_movies_{}".format(last_folder_name))
    if not os.path.exists(cleaned_folder):
        os.makedirs(cleaned_folder)

    if not os.path.exists(utrack_folder):
        os.makedirs(utrack_folder)

    files = [f for f in os.listdir(folder_path) if f.endswith('.nd2')]
    for idx, filename in enumerate(files, 1):
        full_path = os.path.join(folder_path, filename)
        imp = IJ.openImage(full_path)

        if imp:
            # Duplicate the image, keeping frames from start_frame to end_frame
            dup = Duplicator().run(imp, start_frame, end_frame)

            # Enhance contrast
            IJ.run(dup, "Enhance Contrast", "saturated=0.35")

            # Subtract background
            bs = BackgroundSubtracter()
            for i in range(1, dup.getStackSize() + 1):
                dup.setSlice(i)
                bs.rollingBallBackground(dup.getProcessor(), 50, False, False, True, False, True)

            # Save intermediate cleaned file
            cleaned_file_path = os.path.join(cleaned_folder, filename.replace('.nd2', '_cleaned.tif'))
            IJ.saveAsTiff(dup, cleaned_file_path)

            # Calculate mask index using the provided formula
            mask_index = mask_index_formula(idx)
            mask_index_str = "{:03d}".format(mask_index)  # Ensure the index is zero-padded to 3 digits
            mask_filename = '_'.join(filename.split('_')[:-1] + [mask_index_str + '.tif'])
            mask_path = os.path.join(mask_folder, mask_filename)

            mask_imp = IJ.openImage(mask_path)

            if mask_imp:
                for i in range(1, dup.getStackSize() + 1):
                    dup.setSlice(i)
                    mask_imp.setSlice(i)
                    ip_original = dup.getProcessor()
                    ip_mask = mask_imp.getProcessor()

                    # Set pixels in the original image to black where the mask is black
                    for x in range(ip_mask.getWidth()):
                        for y in range(ip_mask.getHeight()):
                            if ip_mask.getPixel(x, y) == 0:  # Assuming black in mask is 0
                                ip_original.putPixel(x, y, 0)  # Set to black in original image

                    dup.updateAndDraw()

            # Create folder for each image in utrack folder
            cell_folder = os.path.join(utrack_folder, "{}_cell{}".format(last_folder_name, idx))
            if not os.path.exists(cell_folder):
                os.makedirs(cell_folder)

            # Save as image sequence
            IJ.run(dup, "Image Sequence... ",
                   "select=[{}] dir=[{}] format=TIFF name=image start=1".format(cell_folder, cell_folder))

            print("Processed and saved: {} into {}, cleaned file into {}, and applied mask {}".format(
                filename, cell_folder, cleaned_file_path, mask_filename))
        else:
            print("Could not open: {}".format(filename))
    print("Processing complete")


folder_path1 = r"C:\path\to\movies1"
folder_path2 = r"C:\path\to\movies2"
folder_path3 = r"C:\path\to\movies3"
folder_path4 = r"C:\path\to\movies4"

mask_folder = r"C:\path\to\masks"
utrack_folder = r"C:\path\to\desired\output\for\movie\frames" #for example this can be your current experiment folder that is the exact desired input for u-track
start_frame = 51  # Adjust as needed for the duplicated movie
end_frame = 300  # Adjust as needed for the duplicated movie

mask_index_formula = lambda x: x * 2  # Formula to calculate the mask index
# mask_index_formula = lambda x: ( x - 80 ) * 2 # Formula to calculate the mask index

process_nd2_files(folder_path1, mask_folder, utrack_folder, start_frame, end_frame, mask_index_formula)
process_nd2_files(folder_path2, mask_folder, utrack_folder, start_frame, end_frame, mask_index_formula)
process_nd2_files(folder_path3, mask_folder, utrack_folder, start_frame, end_frame, mask_index_formula)
process_nd2_files(folder_path4, mask_folder, utrack_folder, start_frame, end_frame, mask_index_formula)
