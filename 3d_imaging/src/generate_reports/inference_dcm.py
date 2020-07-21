"""
Here we do inference on a DICOM volume, constructing the volume first, and then sending it to the
clinical archive

This code will do the following:
    1. Identify the series to run HippoCrop.AI algorithm on from a folder containing multiple studies
    2. Construct a NumPy volume from a set of DICOM files
    3. Run inference on the constructed volume
    4. Create report from the inference
    5. Call a shell script to push report to the storage archive
"""

import os
import sys
import datetime
import time
import shutil
import subprocess

import numpy as np
import pydicom

from glob import glob
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from inference.UNetInferenceAgent import UNetInferenceAgent

def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects a Numpy array.
    Assumes that only one series is in the array

    Arguments:
        dcmlist {list of PyDicom objects} -- path to directory

    Returns:
        tuple of (3D volume, header of the 1st image)
    """

    # In the real world you would do a lot of validation here
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]

    # Make sure that you have correctly constructed the volume from your axial slices!
    hdr = dcmlist[0]

    # We return header so that we can inspect metadata properly.
    # Since for our purposes we are interested in "Series" header, we grab header of the
    # first file (assuming that any instance-specific values will be ighored - common approach)
    # We also zero-out Pixel Data since the users of this function are only interested in metadata
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)

def get_predicted_volumes(pred):
    """Gets volumes of two hippocampal structures from the predicted array

    Arguments:
        pred {Numpy array} -- array with labels. Assuming 0 is bg, 1 is anterior, 2 is posterior

    Returns:
        A dictionary with respective volumes
    """
    volume_ant = (pred == 1).astype(int).sum()
    volume_post = (pred == 2).astype(int).sum()
    total_volume = volume_ant + volume_post

    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}


def pad_to_shape(img, shape):
    """Pads a 2D numpy array to the given shape using zero padding"""
    x_gap = shape[0] - img.shape[0]
    y_gap = shape[1] - img.shape[1]
    x_pad = (x_gap // 2, x_gap - (x_gap // 2))
    y_pad = (y_gap // 2, y_gap - (y_gap // 2))
    return np.pad(img, [x_pad, y_pad], mode='constant')


def create_report(inference, header, orig_vol, pred_vol):
    """Generates an image with inference report

    Arguments:
        inference {Dictionary} -- dict containing anterior, posterior and full volume values
        header {PyDicom Dataset} -- DICOM header
        orig_vol {Numpy array} -- original volume
        pred_vol {Numpy array} -- predicted label

    Returns:
        PIL image
    """

    # The code below uses PIL image library to compose an RGB image that will go into the report
    # A standard way of storing measurement data in DICOM archives is creating such report and
    # sending them on as Secondary Capture IODs (http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html)
    # Essentially, our report is just a standard RGB image, with some metadata, packed into 
    # DICOM format. 

    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)

    header_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=40)
    main_font = ImageFont.truetype("assets/Roboto-Regular.ttf", size=20)

    slice_nums = [orig_vol.shape[2] // 3, orig_vol.shape[2] // 2,
                  orig_vol.shape[2] * 3 // 4]  # is there a better choice?

    axial = np.zeros((orig_vol.shape[1], orig_vol.shape[2]))
    for i in range(orig_vol.shape[0]):
        axial += orig_vol[i, :, :]

    axial = ((axial / np.max(axial)) * 0xff).astype(np.uint8)
    axial = pad_to_shape(axial, (64, 64))

    sagittal = np.zeros((orig_vol.shape[0], orig_vol.shape[2]))
    for i in range(orig_vol.shape[1]):
        sagittal += orig_vol[:, i, :]

    sagittal = ((sagittal / np.max(sagittal)) * 0xff).astype(np.uint8)
    sagittal = pad_to_shape(sagittal, (64, 64))

    coronal = np.zeros((orig_vol.shape[0], orig_vol.shape[1]))
    for i in range(orig_vol.shape[2]):
        coronal += orig_vol[:, :, i]

    coronal = ((coronal / np.max(coronal)) * 0xff).astype(np.uint8)
    coronal = pad_to_shape(coronal, (64, 64))

    pred_anterior = (pred_vol == 1).astype(int)
    pred_posterior = (pred_vol == 2).astype(int)

    axial_anterior = np.zeros((pred_vol.shape[1], pred_vol.shape[2]))
    axial_posterior = np.zeros((pred_vol.shape[1], pred_vol.shape[2]))
    for i in range(pred_vol.shape[0]):
        axial_anterior += pred_anterior[i, :, :]
        axial_posterior += pred_posterior[i, :, :]

    axial_preds = np.zeros((64, 64, 3))
    axial_preds[:,:,1] += pad_to_shape(axial_anterior, (64,64))
    axial_preds[:,:,2] += pad_to_shape(axial_posterior, (64,64))
    axial_mask = ((axial_preds / np.max(axial_preds)) * 0xff).astype(np.uint8)
    axial_mask = Image.fromarray(axial_mask, mode='RGB').convert('RGBA')

    sagittal_anterior = np.zeros((pred_vol.shape[0], pred_vol.shape[2]))
    sagittal_posterior = np.zeros((pred_vol.shape[0], pred_vol.shape[2]))
    for i in range(pred_vol.shape[1]):
        sagittal_anterior += pred_anterior[:, i, :]
        sagittal_posterior += pred_posterior[:, i, :]

    sagittal_preds = np.zeros((64, 64, 3))
    sagittal_preds[:,:,1] += pad_to_shape(sagittal_anterior, (64,64))
    sagittal_preds[:,:,2] += pad_to_shape(sagittal_posterior, (64,64))
    sagittal_mask = ((sagittal_preds / np.max(sagittal_preds)) * 0xff).astype(np.uint8)
    sagittal_mask = Image.fromarray(sagittal_mask, mode='RGB').convert('RGBA')

    coronal_anterior = np.zeros((pred_vol.shape[0], pred_vol.shape[1]))
    coronal_posterior = np.zeros((pred_vol.shape[0], pred_vol.shape[1]))
    for i in range(pred_vol.shape[2]):
        coronal_anterior += pred_anterior[:, :, i]
        coronal_posterior += pred_posterior[:, :, i]

    coronal_preds = np.zeros((64, 64, 3))
    coronal_preds[:,:,1] += pad_to_shape(coronal_anterior, (64,64))
    coronal_preds[:,:,2] += pad_to_shape(coronal_posterior, (64,64))
    coronal_mask = ((coronal_preds / np.max(coronal_preds)) * 0xff).astype(np.uint8)
    coronal_mask = Image.fromarray(coronal_mask, mode='RGB').convert('RGBA')

    # TASK: Create the report here and show information that you think would be relevant to
    # clinicians. A sample code is provided below, but feel free to use your creative
    # genius to make if shine. After all, the is the only part of all our machine learning
    # efforts that will be visible to the world. The usefulness of your computations will largely
    # depend on how you present them.

    # SAMPLE CODE BELOW: UNCOMMENT AND CUSTOMIZE
    draw.text((10, 0), "HippoVolume.AI", (255, 255, 255), font=header_font)
    draw.multiline_text((10, 90),
                        f"""
    Patient ID: {header.PatientID}
    Hippocampus Volume
        - Anterior:__{inference['anterior']} mm^3
        - Posterior:_{inference['posterior']} mm^3
        - Total:_____{inference['total']} mm^3
    """,
                        (255, 255, 255), font=main_font)

    axial_img = Image.fromarray(axial, mode='L').convert('RGBA')
    sagittal_img = Image.fromarray(sagittal, mode='L').convert('RGBA')
    coronal_img = Image.fromarray(coronal, mode='L').convert('RGBA')

    axial_overlay = Image.blend(axial_img, axial_mask, 0.4)
    sagittal_overlay = Image.blend(sagittal_img, sagittal_mask, 0.4)
    coronal_overlay = Image.blend(coronal_img, coronal_mask, 0.4)

    draw.text((10, 500), "2D Axial View", (255, 255, 255), font=main_font)
    draw.text((420, 10), "2D Sagittal View", (255, 255, 255), font=main_font)
    draw.text((420, 500), "2D Coronal View", (255, 255, 255), font=main_font)

    pimg.paste(axial_overlay.resize((400, 400)), box=(40, 520))
    pimg.paste(sagittal_overlay.resize((400, 400)), box=(420, 40))
    pimg.paste(coronal_overlay.resize((400, 400)), box=(420, 520))

    # STAND-OUT SUGGESTION:
    # In addition to text data in the snippet above, can you show some images?
    # Think, what would be relevant to show? Can you show an overlay of mask on top of original data?
    # Hint: here's one way to convert a numpy array into a PIL image and draw it inside our pimg object:
    #
    # Create a PIL image from array:
    # Numpy array needs to flipped, transposed and normalized to a matrix of values in the range of [0..255]
    # nd_img = np.flip((slice/np.max(slice))*0xff).T.astype(np.uint8)
    # This is how you create a PIL image from numpy array
    # pil_i = Image.fromarray(nd_img, mode="L").convert("RGBA").resize(<dimensions>)
    # Paste the PIL image into our main report image object (pimg)
    # pimg.paste(pil_i, box=(10, 280))

    return pimg

def save_report_as_dcm(header, report, path):
    """Writes the supplied image as a DICOM Secondary Capture file

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        report {PIL image} -- image representing the report
        path {Where to save the report}

    Returns:
        N/A
    """

    # Code below creates a DICOM Secondary Capture instance that will be correctly
    # interpreted by most imaging viewers including our OHIF
    # The code here is complete as it is unlikely that as a data scientist you will 
    # have to dive that deep into generating DICOMs. However, if you still want to understand
    # the subject, there are some suggestions below

    # Set up DICOM metadata fields. Most of them will be the same as original file header
    out = pydicom.Dataset(header)

    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # STAND OUT SUGGESTION: 
    # If you want to understand better the generation of valid DICOM, remove everything below
    # and try writing your own DICOM generation code from scratch.
    # Refer to this part of the standard to see what are the requirements for the valid
    # Secondary Capture IOD: http://dicom.nema.org/medical/dicom/2019e/output/html/part03.html#sect_A.8
    # The Modules table (A.8-1) contains a list of modules with a notice which ones are mandatory (M)
    # and which ones are conditional (C) and which ones are user-optional (U)
    # Note that we are building an RGB image which would have three 8-bit samples per pixel
    # Also note that writing code that generates valid DICOM has a very calming effect
    # on mind and body :)

    out.is_little_endian = True
    out.is_implicit_VR = False

    # We need to change class to Secondary Capture
    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    # Our report is a separate image series of one image
    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "OT" # Other
    out.SeriesDescription = "HippoVolume.AI"

    out.Rows = report.height
    out.Columns = report.width

    out.ImageType = r"DERIVED\PRIMARY\AXIAL" # We are deriving this image from patient data
    out.SamplesPerPixel = 3 # we are building an RGB image.
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0 # means that bytes encode pixels as R1G1B1R2G2B2... as opposed to R1R2R3...G1G2G3...
    out.BitsAllocated = 8 # we are using 8 bits/pixel
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    # Set time and date
    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm

    out.ImagesInAcquisition = 1

    # We empty these since most viewers will then default to auto W/L
    out.WindowCenter = ""
    out.WindowWidth = ""

    # Data imprinted directly into image pixels is called "burned in annotation"
    out.BurnedInAnnotation = "YES"

    out.PixelData = report.tobytes()

    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)

def get_series_for_inference(path):
    """Reads multiple series from one folder and picks the one
    to run inference on.

    Arguments:
        path {string} -- location of the DICOM files

    Returns:
        Numpy array representing the series
    """

    # Here we are assuming that path is a directory that contains a full study as a collection
    # of files
    # We are reading all files into a list of PyDicom objects so that we can filter them later
    files = glob(os.path.join(path, "*", "*.dcm"))
    dicoms = [pydicom.dcmread(f) for f in files]
    series_for_inference = [dcm for dcm in dicoms if dcm.SeriesDescription == "HippoCrop"]

    # Check if there are more than one series (using set comprehension).
    if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        print("Error: can not figure out what series to run inference on")
        return []

    return series_for_inference

def os_command(command):
    # Comment this if running under Windows
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", command])
    sp.communicate()

    # Uncomment this if running under Windows
    # os.system(command)

if __name__ == "__main__":
    # This code expects a single command line argument with link to the directory containing
    # routed studies
    if len(sys.argv) != 2:
        print("You should supply one command line argument pointing to the routing folder. Exiting.")
        sys.exit()

    # Find all subdirectories within the supplied directory. We assume that 
    # one subdirectory contains a full study
    subdirs = [os.path.join(sys.argv[1], d) for d in os.listdir(sys.argv[1]) if
                os.path.isdir(os.path.join(sys.argv[1], d))]

    # Get the latest directory
    study_dir = sorted(subdirs, key=lambda dir: os.stat(dir).st_mtime, reverse=True)[0]

    print(f"Looking for series to run inference on in directory {study_dir}...")

    # TASK: get_series_for_inference is not complete. Go and complete it
    volume, header = load_dicom_volume_as_numpy_from_list(get_series_for_inference(study_dir))
    print(f"Found series of {volume.shape[2]} axial slices")

    print("HippoVolume.AI: Running inference...")
    # TASK: Use the UNetInferenceAgent class and model parameter file from the previous section
    inference_agent = UNetInferenceAgent(
        device="cpu",
        parameter_file_path=os.path.join(".", "unet", "model.pth")
    )

    # Run inference
    # TASK: single_volume_inference_unpadded takes a volume of arbitrary size 
    # and reshapes y and z dimensions to the patch size used by the model before 
    # running inference. Your job is to implement it.
    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    # TASK: get_predicted_volumes is not complete. Go and complete it
    pred_volumes = get_predicted_volumes(pred_label)

    # Create and save the report
    print("Creating and pushing report...")
    report_save_path = os.path.join("..", "reports", "report.dcm")
    # TASK: create_report is not complete. Go and complete it. 
    # STAND OUT SUGGESTION: save_report_as_dcm has some suggestions if you want to expand your
    # knowledge of DICOM format
    report_img = create_report(pred_volumes, header, volume, pred_label)
    save_report_as_dcm(header, report_img, report_save_path)

    # Send report to our storage archive
    # TASK: Write a command line string that will issue a DICOM C-STORE request to send our report
    # to our Orthanc server (that runs on port 4242 of the local machine), using storescu tool
    os_command("<COMMAND LINE TO SEND REPORT TO ORTHANC>")

    # This line will remove the study dir if run as root user
    # Sleep to let our StoreSCP server process the report (remember - in our setup
    # the main archive is routing everyting that is sent to it, including our freshly generated
    # report) - we want to give it time to save before cleaning it up
    time.sleep(2)
    shutil.rmtree(study_dir, onerror=lambda f, p, e: print(f"Error deleting: {e[1]}"))

    print(f"Inference successful on {header['SOPInstanceUID'].value}, out: {pred_label.shape}",
          f"volume ant: {pred_volumes['anterior']}, ",
          f"volume post: {pred_volumes['posterior']}, total volume: {pred_volumes['total']}")
