macro "Process ND2 Files" {
    // Set user-defined parameters
    folderPaths = newArray(
        "C:\\path\\to\\raw\\movies\\data\\"
    );   

    startFrame = 25;
    endFrame = 250;

    maskIndexFormulas = newArray(
        "x * 1"
    );

for (f = 0; f < lengthOf(folderPaths); f++) {
    folderPath = folderPaths[f];

    cleanedFolder = folderPath + "cleaned_movies\\";
    File.makeDirectory(cleanedFolder);

    list = getFileList(folderPath);

    for (i = 0; i < list.length; i++) {
        if (endsWith(list[i], ".nd2")) {
            filePath = folderPath + list[i];
            movieName = replace(list[i], ".nd2", "");

            print("Processing file: " + filePath);

            // Open the ND2 file
            run("Bio-Formats Importer", "open=[" + filePath + "] autoscale color_mode=Default view=Hyperstack stack_order=Default");

            if (nImages == 0 || getTitle() == "") {
                print("ERROR: Failed to open file: " + filePath);
                closeAll();
                continue;
            }

            // Duplicate the image (with frame range)
            run("Duplicate...", "duplicate range=" + startFrame + "-" + endFrame);
            wait(500);

            // Get the name of the duplicate
            imageList = getList("image.titles");
            dupTitle = "";

            for (j = 0; j < lengthOf(imageList); j++) {
                if (endsWith(imageList[j], "-1.nd2")) {  // Find the duplicate
                    dupTitle = imageList[j];
                    break;
                }
            }

            if (dupTitle == "") {
                print("ERROR: Could not find the duplicated image.");
                closeAll();
                continue;
            }

            // Select and process the duplicate
            selectWindow(dupTitle);
            print("Using duplicated image: " + dupTitle);

            run("Enhance Contrast", "saturated=0.35");
            run("Subtract Background...", "rolling=50 stack");

            // Save the cleaned file
            cleanedFilePath = cleanedFolder + movieName + "_cleaned.tif";
            saveAs("Tiff", cleanedFilePath);

            print("Saved cleaned file: " + cleanedFilePath);

            // Close all open images to free memory
            closeAll();
        }
    }
}
print("All files processed. Exiting.");


// Function to extract last numeric value
function extractLastNumber(filename) {
    number = -1;
    for (i = lengthOf(filename) - 3; i >= 0; i--) {
        if (i + 3 <= lengthOf(filename)) {
            segment = substring(filename, i, i + 3);
            if (isNumber(segment)) {
                number = parseInt(segment, 10);
                break;
            }
        }
    }
    return number;
}

// Function to format numbers into three-digit format
function formatMaskNumber(num) {
    numStr = "" + num;
    while (lengthOf(numStr) < 3) {
        numStr = "0" + numStr;
    }
    return numStr;
}

// Function to check if a string is a number
function isNumber(str) {
    return !isNaN(parseFloat(str)) && isFinite(str);
}

// Function to close all images
function closeAll() {
    while (nImages > 0) {
        close();
    }
}
