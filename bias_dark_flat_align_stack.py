import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import scipy
import scipy.signal
import scipy.ndimage.interpolation as interp
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
import os.path
import os 
import astropy.stats as stat
from astropy.visualization import ZScaleInterval
from astropy.stats import sigma_clipped_stats
from astropy.io import fits 
from astropy import stats
from astropy.stats import mad_std
from astropy.stats import sigma_clip
from PIL import Image

!pip install photutils
from photutils.utils import calc_total_error
from photutils import aperture_photometry, CircularAperture, CircularAnnulus, DAOStarFinder

!pip install astroalign
import astroalign as aa

%matplotlib inline

# Mount the Google drive
from google.colab import drive
drive.mount('/content/drive') 

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def filesorter(filename, dir, foldername):

    '''
    PURPOSE: 
            Checks if the directory (dir + foldername + filename) exists, then creates the directory if it doesn't.
    INPUTS:  
            [filename]:  (string) Name of the file
                 [dir]:  (string) Directory where you want things to be saved
          [foldername]:  (string) Name of the folder you want to check/generate
    RETURNS:  
            Creates the specified folders.
    AUTHOR:
            Connor E. Robinson
    '''

    # The statement checks to see if the file exists.
    if os.path.exists(dir + filename):
        pass
    else:
        print(dir + filename + " does not exist or has already been moved.")
        return

    # If the foldername (input) doesn't exist, then it creates a new directory with the name foldername.
    if os.path.exists(dir + foldername):
        pass
    else:
        print("Making new directory: " + dir + foldername)
        os.mkdir(dir + foldername)

    # Move files to new folder
    print('Moving ' + filename + ' to:  ' + dir + foldername + '/' + filename)
    os.rename(dir + filename, dir + foldername + '/' + filename)

    return

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def mediancombine(first_letter, dir):

    '''
    PURPOSE: 
            Median combine all the images in the directory 'dir' that start with the letter 'first_letter'.
    INPUTS:  
            [first_letter]:  (string)  First letter(s) of the files to be combined
                     [dir]:  (string)  Directory that contains the images to be median combined
    RETURNS:  
               [med_frame]:  [np.array, float] Median combined image
    AUTHOR:
            Julio M. Morales, October 15, 2021
    '''

    # Collect all the frames
    files = glob.glob(dir + first_letter + '*.fit')

    # Number of files we have
    n = len(files)  
  
    # Pull data from the first file
    first_frame_data = fits.getdata(files[0]) 
  
    # Shape of the first file
    imsize_y, imsize_x = first_frame_data.shape
  
    # Makes n amount of arrays of zeros x by y size
    fits_stack = np.zeros((imsize_y, imsize_x, n))   
  
    # True if we are trying to make a master flat image
    if (first_letter == 'dark_subtracted_'):

        for ii in range(0, n):

            # Pull the data from each image
            im = fits.getdata(files[ii])
            # Normalize the image by dividing off the median from each pixel
            norm_im =  im / np.median(im)
            # Replace all zeros in the zero array with the normalized values
            fits_stack[:, :, ii] = norm_im
        
        # Median combine all the normalized frames while maintaining the shape of the array  
        med_frame = np.median(fits_stack, axis = 2)

    # True if we are making a master bias or dark frame
    else:

        for ii in range(0, n):  
    
            # Pull all the data from the files
            im = fits.getdata(files[ii])
            # Replace all zeros in the zero array with the image values
            fits_stack[:, :, ii] = im
        
        # Median combine all the frames while maintaining the shape of the array  
        med_frame = np.median(fits_stack, axis = 2)

    return med_frame

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def bias_subtract(first_letter, master_bias_path, dir):
  
    '''
    PURPOSE: 
            Bias subtract all the images in the directory 'dir' that start with the letter 'first_letter' using the master bias frame.
    INPUTS:
                [first_letter]:  (string)  First letter(s) of the files to be bias subtracted
            [master_bias_path]:  (string)  Pathway to the master bias image
                         [dir]:  (string)  Directory of files to be calibrated
  
    RETURNS:
            Writes .fit file to the specified directory
    AUTHOR:
            Julio M. Morales, October 27, 2021
    '''

    # Collect all the files
    files = glob.glob(dir + first_letter + '*.fit')

    for image in files:

        # Pull the data from the files
        data = fits.getdata(image)

        # Get the header information we want
        header = fits.getheader(image)
                
        # Get the master bias data             
        master_bias_data = fits.getdata(master_bias_path)

        # Subtract the bias from the pulled frames
        bias_subtracted_data = data - master_bias_data      

        # Write to new fits file with the header of the orignal file
        fits.writeto(dir + 'bias_subtracted_' + image.split('/')[-1], bias_subtracted_data, header = header, overwrite = True)

    return

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def dark_subtract(first_letter, master_dark_path, dir):
  
    '''
    PURPOSE: 
            Dark subtract all the images in the directory 'dir' that start with the letter 'first_letter' using the master dark frame.
    INPUTS:  
                [first_letter]:  (string)  First letter(s) of the files to be dark subtracted
            [master_dark_path]:  (string)  Pathway to the master dark image
                         [dir]:  (string)  Directory of files to be calibrated
    RETURNS:
            Writes .fit file to the specified directory
    AUTHOR:
            Julio M. Morales, October 27, 2021
    '''

    # Collect all the files
    files = glob.glob(dir + first_letter + '*.fit')

    for image in files:

        # Pull the data from the files
        data = fits.getdata(image)

        # Get the header information we want
        header = fits.getheader(image)

        # Get the master bias data 
        master_dark_data = fits.getdata(master_dark_path)

        # Subtract the bias from the pulled frames
        dark_subtracted_data = data - master_dark_data

        # Write to new fits file with the header of the orignal file
        fits.writeto(dir + 'dark_subtracted_' + image.split('/')[-1], dark_subtracted_data, header = header, overwrite = True)

    return

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def flat_field(first_letter, master_flat_path, dir):
  
    '''
    PURPOSE: 
            Flat-field all the images in the directory 'dir' that start with the letter 'first_letter' using the master flat frame.
    INPUTS:  
                [first_letter]:  (string)  First letter(s) of the files to be flat-fielded
            [master_flat_path]:  (string)  Pathway to the master flat image
                         [dir]:  (string)  Directory of files to be calibrated
    RETURNS:
            Writes .fit file to the specified directory
    AUTHOR:
            Julio M. Morales, October 27, 2021
    '''

    # Pull all the files to be calibrated
    files = glob.glob(dir + first_letter + '*.fit')

    for image in files:

        # Pull the data from the files
        data = fits.getdata(image)

        # Get the header information we want
        header = fits.getheader(image)

        # Get the master flat data 
        master_flat_image = fits.getdata(master_flat_path)

        # Divide the files by the master_flat image
        flat_fielded_data = data.astype('float') / master_flat_image.astype('float')

        # Write to new fits file with the header of the orignal file
        fits.writeto(dir + 'flat_fielded_' + image.split('/')[-1], flat_fielded_data, header = header, overwrite = True)

    return

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def master_frame_generator(first_letter, dir, filter):

    '''
    PURPOSE:  
            To bias-subtract, dark-subtract, and flat-field all the images in the directory 'dir' that start with the letter 'first_letter'.
    INPUTS:   
            [first_letter]:  (string)  First letter(s) of the files to be reduced
                     [dir]:  (string)  Directory of files to be calibrated
                  [filter]:  (string)  Name of the filter the frames were taken in
    RETURNS:  
            [master_bias_path]:  (string)  Path to the master bias frame
            [master_dark_path]:  (string)  Path to the master dark frame
            [master_flat_path]:  (string)  Path to the master flat frame
    AUTHOR:  
            Julio M. Morales, November 01, 2021
    '''

    # Create master bias frame
    master_bias = mediancombine(first_letter, dir + '/Bias Frame/')
    fits.writeto(dir + 'Bias Frame/Master_Bias.fit', master_bias, overwrite = True)

    # Get the exopsure time of the source frames to know which dark frames we need to use
    science_files = glob.glob(dir + 'Light Frame/source/' + filter + '/*.fit')
    header = fits.getheader(science_files[0])
    exposure = str(header['EXPTIME'])

    # Bias-subtract the dark frames, then create master dark
    bias_subtract(first_letter, dir + 'Bias Frame/Master_Bias.fit', dir + 'Dark Frame/' + exposure + '/')
    master_dark = mediancombine('bias_subtracted_', dir + 'Dark Frame/' + exposure + '/')
    fits.writeto(dir + 'Dark Frame/' + exposure + '/Master_Dark.fit', master_dark, overwrite = True)

    # Bias and dark-subtract the flat frames, then create the master flat
    bias_subtract(first_letter, dir + 'Bias Frame/Master_Bias.fit', dir + 'Flat Field/' + filter + '/')
    dark_subtract('bias_subtracted_', dir + 'Dark Frame/' + exposure + '/Master_Dark.fit', dir + 'Flat Field/' + filter + '/')
    master_flat = mediancombine('dark_subtracted_', dir + 'Flat Field/' + filter + '/')
    fits.writeto(dir + 'Flat Field/' + filter + '/Master_Flat.fit', master_flat, overwrite = True)

    # Define the pathways for each master frame
    master_bias_path = dir + 'Bias Frame/Master_Bias.fit'
    master_dark_path = dir + 'Dark Frame/' + exposure + '/Master_Dark.fit'
    master_flat_path = dir + 'Flat Field/' + filter + '/Master_Flat.fit'

    return [master_bias_path, master_dark_path, master_flat_path]

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def cross_image(im1, im2, **kwargs):

    '''
    PURPOSE: 
            This function performs cross-correlation by slicing each image, and subtracting the 
            mean of each image from itself.  It then performs a fast--fourier transform on the images.  
            It then calculates the shifts required for alignment by comparing peak pixel value locations.
    INPUTS:  
                 [im1]:  (np.array, float)  First image to be cross correlated
                 [im2]:  (np.array, float)  Second image to be cross correlated
            [**kwargs]:  (list)             Variable-length keyword list
    RETURNS:  
            [xshift, yshift]:  (list, float)  x and y shifts required to align the images
    AUTHOR:  
            Connor E. Robinson
    '''

    # The type cast into 'float' is to avoid overflows:
    im1_gray = im1.astype('float')
    im2_gray = im2.astype('float')

    # Enable a trimming capability using keyword argument option.
    if ('boxsize' in kwargs):

        im1_gray = im1_gray[0:kwargs['boxsize'], 0:kwargs['boxsize']]
        im2_gray = im2_gray[0:kwargs['boxsize'], 0:kwargs['boxsize']]

    # Subtract the averages of im1_gray and im2_gray from their respective arrays -- cross-correlation
    # works better that way.
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    # Calculate the correlation image using fast Fourier transform (FFT)
    corr_image = scipy.signal.fftconvolve(im1_gray, im2_gray[::-1, ::-1], mode = 'same')
    
    # Determine the location of the peak value in the cross-correlated image
    peak_corr_index = np.argmax(corr_image)

    # Find the peak signal position in the cross-correlation -- this gives the shift between the images.
    corr_tuple = np.unravel_index(peak_corr_index, corr_image.shape)
    
    # Calculate shifts (not cast to integer, but could be).
    xshift = corr_tuple[0] - corr_image.shape[0]/2.
    yshift = corr_tuple[1] - corr_image.shape[1]/2.

    return xshift, yshift
  
###########################################################################################################################################################################################################################################################################################################################################################################################################################################
  
def shift_image(image, xshift, yshift):

    '''
    PURPOSE: 
            This function takes as input an image, and the x and y-shifts to be executed.  
            It then performs this shift by using the np.roll() function which shifts each pixel in a specifced direction.  
            The amount by which it is shifted depends on the inputs x-shift and y-shift.
    INPUTS:
             [image]:  (np.array, float)  Image to be shifted
            [xshift]:  (float)            Amount that the image will be shifted by in the x-direction
            [yshift]:  (float)            Amount that the image will be shifted by in the y-direction
    RETURNS: 
            [np.array, float]  Rolled image
    AUTHOR:
            Connor E. Robinson
    '''

    # Roll the image using the input shifts wwhile maintaning the shape of the array
    return np.roll(np.roll(image, int(yshift), axis = 1), int(xshift), axis = 0)
  
###########################################################################################################################################################################################################################################################################################################################################################################################################################################
  
def align_N_stack(targname, first_letter, dir, filter):

    '''
    PURPOSE: 
            Align and stack fully reduced images that have been bias and dark-subtracted, and flat-fielded.
    INPUTS:  
                [targname]:  (string)  Name of the object your imaging
            [first_letter]:  (string)  First letter(s) of the files you will be aligning and stacking
                     [dir]:  (string)  Directory where your fully reduced images are
                  [filter]:  (string)  Filter of the images you want to reduce
    RETURNS:
            [median_image]:  (np.array, float)  Aligned and stacked images saved into a new foder named 'stacked' in the directory 'dir'.
    AUTHOR:  
            Connor E. Robinson
    '''

    # Collect all the fully-reduced images
    imlist = glob.glob(dir + filter + '/' + first_letter + '*.fit')

    # Check to make sure that your new list has the right files:
    print("All files to be aligned: \n", imlist)
    print('\n')

    # Open first image = master image; all other images of same target will be aligned to this one.
    im1, hdr1 = fits.getdata(imlist[0], header = True)
    print("Aligning all images to:", imlist[0])
    print('\n')

    xshifts = {}
    yshifts = {}

    for index, filename in enumerate(imlist):

        # Pull the data from the images
        im, hdr = fits.getdata(filename, header = True)
        
        # Shift the images using the cross-correlation method
        xshifts[index], yshifts[index] = cross_image(im1, im, boxsize = 1000)
        print("Shift for image", index, "is", xshifts[index], yshifts[index])

    # Calculate trim edges of new median stacked images so all stacked images of each target have same size 
    max_x_shift = int(np.max([xshifts[x] for x in xshifts.keys()]))
    max_y_shift = int(np.max([yshifts[x] for x in yshifts.keys()]))
    print('   Max x - shift = {0}, max y - shift = {1} (pixels)'.format(max_x_shift,max_y_shift))
    print('\n')

    scilist = []

    # Populates scilist with a list of FITS files matching *only* the selected filter:
    for fitsfile in imlist:
    
        # Pull the header info
        header = fits.getheader(fitsfile)

        # True if the filter of the files is equal to the input filter
        if header['FILTER'] == filter:

            # Add the file to the empty list
            scilist.append(fitsfile)
        
        # True if the filter does not equal the input filter
        else:
            pass

    # Checks that the list of files is not empty 
    if len(scilist) < 1:
        print("Warning! No files in scilist. Your path is likely incorrect.")
  
    # Ensure each scilist entry has the right filter:
    for fitsfile in scilist:
        print('File:  ', fitsfile)
        print('\n') 
        print('Filter:  ', filter)
  
    nfiles = len(scilist)
    print('Stacking ', nfiles, filter, ' science frames')
    print('\n')

    # Define new array with same size as master image
    image_stack = np.zeros([im1.shape[0], im1.shape[1], len(scilist)])

    xshifts_filt = {}
    yshifts_filt = {}

    # This loop completes the shifts for the rest of the images in relation to the master image
    for index, filename in enumerate(scilist):

        # Pull the data and header info
        im, hdr = fits.getdata(filename, header = True)

        # Shift the images using the cross-correlation method
        xshifts_filt[index], yshifts_filt[index] = cross_image(im1, im, boxsize = 1000)

        # Shift the images
        image_stack[:, :, index] = shift_image(im, xshifts_filt[index], yshifts_filt[index])      

    # Take the median of the image stack while maintaing the dimensions of the array
    median_image = np.median(image_stack, axis = -1)

    # Sets the new image boundaries
    if (max_x_shift > 0) & (max_y_shift > 0):
        median_image = median_image[max_x_shift:-max_x_shift, max_y_shift:-max_y_shift]

    # True if the stacked directory does not exist
    if os.path.isdir(dir + 'stacked') == False:

        # Make the stacked directory
        os.mkdir(dir + 'stacked')
        print('\n Making new subdirectory for stacked images:', dir + 'stacked \n')
      
  
    # Save the final stacked images into your new folder
    fits.writeto(dir + '/stacked/' + targname + '_' + filter + ' stack.fits', median_image, header = hdr, overwrite = True)
    print('   Wrote FITS file ', targname + '_' + filter + ' stack.fits', 'in ', dir + 'stacked', '\n')
    print('\n Done stacking!')

    return median_image

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def align_filters(dir, targname, sigma = 4, standard = False):

    '''
    PURPOSE:
            To align images of differing filters to one another and saving the aligned images to a .fits. file.
    INPUTS:
                  [dir]:  (string)  General directory of the folders containing all of the data files
             [targname]:  (string)  Name of the target you are aligning
            [calfilter]:  (string)  Name of the filter you want to use as a reference for alignment
    OPT. INPUTS:
               [sigma]:  (float)    Confidence threshold for detections in the images (default is 4)
            [standard]:  (boolean)  'True' if we are aligning the standard images, 'False' if we are aligningthe primary source of interest
    RETURNS:
            Writes the newly aligned images as a .fits file and saves to a new 'aligned' folder.
    AUTHOR:
            Julio M. Morales, December 10, 2021
    '''

    # True if we are aligning the primary source of interest
    if (standard == False):

        imtype = 'source'

    # True if we are aligning the standard images
    else:

        imtype = 'standard'

    # Collect the files that we will align
    files = glob.glob(dir + 'Light Frame/' + imtype + '/stacked/*stack.fits')

    for image in files:

        # True if the images are in Blue or in Visual Bands
        if (image == dir + 'Light Frame/' + imtype + '/stacked/' + targname + '_Blue stack.fits') or (image == dir + 'Light Frame/' + imtype + '/stacked/' + targname + '_Visual stack.fits'):

            # Retrieve the data for the image that we will align to
            calimage =  fits.getdata(dir + 'Light Frame/' + imtype + '/stacked/' + targname + '_Blue stack.fits')
            print('Aligning ' + image + ' to ' + dir + 'Light Frame/' + imtype + '/stacked/' + targname + '_Blue stack.fits' + '\n')

        # True if the image is in the Red band
        else:

            # Use the Visual filter to align the Red image (I believe this is neccesary since the aligning function can't identify enough similarities in the Blue and Red images to work, but it does
            # work for the Visual and Red bands)
            calimage =  fits.getdata(dir + 'Light Frame/' + imtype + '/aligned/' + targname + '_Visual aligned.fits')
            print('Aligning ' + image + ' to ' + dir + 'Light Frame/' + imtype + '/aligned/' + targname + '_Visual aligned.fits' + '\n')

        # Convert the array from big eldien to little eldien (code breaks without this!)
        calimage =  calimage.byteswap().newbyteorder()

        # Pull the data from the images to be aligned
        image_to_align = fits.getdata(image)

        # Pull the header info
        header = fits.getheader(image)

        # Pull the filter from the header
        filter = header['FILTER']

        # Convert the array from big eldien to little eldien (code breaks without this!)
        image_to_align = image_to_align.byteswap().newbyteorder()
        
        # Execute the alignment (footprint is a boolean mask where True is an unphysical pixel)
        img_aligned, footprint = aa.register(image_to_align, calimage, detection_sigma = sigma)

        # Loop over all the pixels in the aligned image
        for j in range(0, footprint.shape[0]):

            for i in range(0, footprint.shape[1]):

                # True if the pixel is nonphysical
                if (footprint[j][i] == True):

                    # Replace the nonphysical pixel with a NaN
                    img_aligned[j][i] = np.nan

                # True if no NaN's are needed
                else:
                    pass

        # Write to a new fits file and sort into a new 'aligned' folder
        fits.writeto(dir + 'Light Frame/' + imtype + '/' + targname + '_' + filter + ' aligned.fits', img_aligned, header = header, overwrite = True)
        print('Writing the aligned ' + targname + '_' + filter + ' aligned.fits to ' + dir + 'Light Frame/' + imtype + '/aligned/' + '\n')
        filesorter(targname + '_' + filter + ' aligned.fits', dir + 'Light Frame/' + imtype + '/', 'aligned')
        print('\n')

    return

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def science_file_sorter(dir, source_first_letter, standard_first_letter, is_standard = False):

    '''
    PURPOSE: 
            Takes in the directory where all of the files taken reside, it then organizes the files by 
            calibration type, filter, exposure time, and by source vs. standard type.
    INPUTS:  
                            [dir]:  (string)  Directory where all the files reside
            [source_first_letter]:  (string)  The first letter(s) of the files taken for the primary source of interest
          [standard_first_letter]:  (string)  The first letter(s) of the files taken for the standard star
    OPT. INPUTS:
                    [is_standard]:  (boolean)  True if you are trying to organize images of the standard star.  False if for the source.
    RETURNS:
            Sorts all of the files into new directories.
    AUHTOR:
            Julio M. Morales, November 19, 2021
    '''

    # Compile all the files in the dir
    all_files = glob.glob(dir + '*.fit')

    # Organize by image type (bias vs. dark vs. flat vs. light)
    for files in all_files:
        header = fits.getheader(files)
        file_type = header['IMAGETYP']
        filesorter(files.split('/')[-1], dir, file_type)

    # Compile all the dark frames
    dark_files = glob.glob(dir + 'Dark Frame/*.fit')

    # Sort the dark frames by exopsure times
    for files in dark_files:
        header = fits.getheader(files)
        exp_time = header['EXPTIME']
        filesorter(files.split('/')[-1], dir + 'Dark Frame/', str(exp_time))

    # Compile all of the flat frames
    flat_files = glob.glob(dir + 'Flat Field/*.fit')

    # Sort the flat-frames by filter name
    for files in flat_files:
        header = fits.getheader(files)
        filter = header['FILTER']
        filesorter(files.split('/')[-1], dir + 'Flat Field/', filter)

    # Compile science files of the source of interest
    source_files = glob.glob(dir + 'Light Frame/' + source_first_letter + '*.fit')

    # Sort the science frames by filter
    for files in source_files:
        filesorter(files.split('/')[-1], dir + 'Light Frame/', 'source')
        header = fits.getheader(dir + 'Light Frame/source/' + files.split('/')[-1])
        filter = header['FILTER']
        filesorter(files.split('/')[-1], dir + 'Light Frame/source/', filter)

    if (is_standard == True):

        # Compile the science frames of the standard star
        standard_files = glob.glob(dir + 'Light Frame/' + standard_first_letter + '*.fit')

        # Organize the standard files by filter
        for files in standard_files:
            filesorter(files.split('/')[-1], dir + 'Light Frame/', 'standard')
            header = fits.getheader(dir + 'Light Frame/standard/' + files.split('/')[-1])
            filter = header['FILTER']
            filesorter(files.split('/')[-1], dir + 'Light Frame/standard/', filter)

    else:
        pass

    return

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def source_image_reducer(targname, first_letter, dir, filter, is_standard = False):

    '''
    PURPOSE:
            To bias-subtract, dark-subtract, flat-field, align, and stack the raw images of the primary source of interest.
    IMPUTS:
                [targname]:  (string)  Name of the primary source of interest
            [first_letter]:  (string)  First letter(s) of the files to be reduced
                     [dir]:  (string)  Directory of files to be calibrated
                  [filter]:  (string)  Name of the filter the frames were taken in
    OPT. INPUTS:
             [is_standard]:  (boolean) True if you are trying to reduce images of the standard star.  False if for the source.
    RETURNS:
             .fits file written with the fully reduced images that have been bias, dark subtracted, flat-fielded, aligned, and stacked.
    AUTHOR:
            Julio M. Morales, November 01, 2021
    '''

    # Generate master frames
    master_frames = master_frame_generator(first_letter, dir, filter)

    # Bias-subtract the science frames
    bias_subtract(first_letter, master_frames[0], dir + 'Light Frame/source/' + filter + '/')

    # Dark-subtract the bias-subtracted frames
    dark_subtract('bias_subtracted_', master_frames[1], dir + 'Light Frame/source/' + filter + '/')

    # Flat-field the bias and dark-subtracted frames
    flat_field('dark_subtracted_', master_frames[2], dir + 'Light Frame/source/' + filter + '/')

    # Align and stack the reduced images
    align_N_stack(targname, 'flat_fielded_', dir + 'Light Frame/source/', filter)

    return

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def standard_image_reducer(targname, first_letter, dir, filter):

    '''
    PURPOSE:
            To bias-subtract, dark-subtract, flat-field, align, and stack the raw images of the standard star.
    IMPUTS:
                [targname]:  (string)  Name of the standard star
            [first_letter]:  (string)  First letter(s) of the files to be reduced
                     [dir]:  (string)  Directory of files to be calibrated
                  [filter]:  (string)  Name of the filter the frames were taken in
    RETURNS:
            .fits file writen with the fully calibrated images that have been bias, dark subtracted, flat-fielded, aligned, and stacked.
    AUTHOR:
            Julio M. Morales, November 01, 2021
    '''

    # Get the exopsure time of the source frames to know which dark frames we need to use
    science_files = glob.glob(dir + 'Light Frame/standard/' + filter + '/*.fit')
    header = fits.getheader(science_files[0])
    exposure = str(header['EXPTIME'])

    # Generate master frames
    master_bias_path =  dir + 'Bias Frame/Master_Bias.fit'
    master_dark_path =  dir + 'Dark Frame/' + exposure + '/Master_Dark.fit'
    matster_flat_path = dir + 'Flat Field/' + filter + '/Master_Flat.fit'

    # Bias subtract the science frames
    bias_subtract(first_letter, master_bias_path, dir + 'Light Frame/standard/' + filter + '/')

    # Dark-subtract the bias-subtracted frames
    dark_subtract('bias_subtracted_', master_dark_path, dir + 'Light Frame/standard/' + filter + '/')

    # Flat-field the bias and dark-subtracted frames
    flat_field('dark_subtracted_', matster_flat_path, dir + 'Light Frame/standard/' + filter + '/')

    # Align and stack the reduced images
    align_N_stack(targname, 'flat_fielded_', dir + 'Light Frame/standard/', filter)

    return

###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def scale_filter(tmpimg, lowsig = -1. , highsig = 15.):

    '''
    PURPOSE:
            The fuunction scales the filters to appropriate amount for the purpose of stacking the images to
            make a false colored image.
    INPUTS:
            [tmpimg]:  (np.array, float)  Image to be scaled
    
    OPT. INPUTS:
            [lowsig]:  (integer)  Lowest sigma  (default is -1)
           [highsig]:  (integer)  Highest sigma (default is 15)
    RETURNS:
               [IMG]:  (np.array, float)  Scaled image
    AUTHOR:
            Connor E. Robinson
    '''

    # Subtract the median of the image
    tmpimg -= np.nanmedian(tmpimg)
    print('minmax 1: ', np.nanmin(tmpimg), np.nanmax(tmpimg))

    # Sigma clip sources 2 sigma and above
    tmpsig = stats.sigma_clipped_stats(tmpimg, sigma = 2, maxiters = 5)[2]
    print('std: ', tmpsig)
    print("lowsig, highsig: ", lowsig, highsig)
    print('cuts: ', lowsig*tmpsig, highsig*tmpsig)

    # Create histogram for the data
    image_hist = plt.hist(tmpimg.flatten(), 1000, range = [-100, 100])
 
    # Apply thresholding
    tmpimg[np.where(tmpimg < lowsig*tmpsig)] = lowsig*tmpsig
    tmpimg[np.where(tmpimg > highsig*tmpsig)] = highsig*tmpsig
    print('minmax 2: ', np.nanmin(tmpimg),np.nanmax(tmpimg))

    # Double hyperbolic arcsin scaling
    tmpimg = np.arcsinh(tmpimg)
    print('minmax 3: ', np.nanmin(tmpimg),np.nanmax(tmpimg))

    # Scale to [0,255]
    tmpimg += np.nanmin(tmpimg)
    tmpimg *= 255./np.nanmax(tmpimg)
    tmpimg[np.where(tmpimg < 0.)] = 0.
    print('minmax 4: ', np.nanmin(tmpimg),np.nanmax(tmpimg))
    
    # Recast as unsigned integers for jpeg writer
    IMG = Image.fromarray(np.uint8(tmpimg))
    
    print("")
    
    return IMG
