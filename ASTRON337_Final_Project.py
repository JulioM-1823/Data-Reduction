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

%matplotlib inline

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

 ###########################################################################################################################################################################################################################################################################################################################################################################################################################################

def bg_error_estimate(dir, targname, filter, gain = 1.4, sigma = 4.0, standard = False):

    '''
    PURPOSE:
            To bias-subtract, dark-subtract, flat-field, align, and stack the raw images of the primary source of interest.

    IMPUTS:
                 [dir]:  (string)  Directory of files to be calibrated
            [targname]:  (string)  Name of the primary source of interest
              [filter]:  (string)  Name of the filter the frames were taken in

    OPT. INPUTS:
                [gain]:  (float)   Gain of the detector (default is 1.4)
               [sigma]:  (float)   Threshold of the detections (default is 4.0)
            [standard]:  (boolean) True if you are working with images of the standard star.  False if for the source.


    RETURNS:
            [error_image]:  (np.array, float)  Sigma clipped error image

    AUTHOR:
            Julio M. Morales, December 01, 2021
    '''

    # True if we are aligning the primary source of interest
    if (standard == False):

        imtype = 'source'

    # True if we are aligning the standard images
    else:

        imtype = 'standard'

    # Pull the data and header from the fully reduced images
    fitsdata = fits.getdata  (dir + 'Light Frame/' + imtype + '/aligned/' + targname + '_' + filter + ' aligned.fits')
    header =   fits.getheader(dir + 'Light Frame/' + imtype + '/aligned/' + targname + '_' + filter + ' aligned.fits')
    
    # Remove pixels from the data that are above sigma times the standard deviation of the background
    filtered_data = sigma_clip(fitsdata, sigma = sigma, copy = False)

    # Take the filtered values and fill them with NaN's
    bkg_values_nan = filtered_data.filled(fill_value = np.nan)

    # Find the variance of all the remaining values (the values that have not been replaced with NaN's)
    bkg_error = np.sqrt(bkg_values_nan)

    # Find the median of the varience (found in the previous line)
    bkg_error[np.isnan(bkg_error)] = np.nanmedian(bkg_error)
    print("Writing the background-only error image: ", dir + 'Light Frame/' + imtype + '/aligned/' + targname + '_' + filter + ' aligned.fits')
    print('\n')
    fits.writeto(dir + 'Light Frame/' + imtype + '/aligned/' + filter + '*aligned_bgerror.fit', bkg_error, header, overwrite = True)
    
    # Calculate the error image and write to a new file
    error_image = calc_total_error(fitsdata, bkg_error, gain)  
    print("Writing the total error image: ", dir + 'Light Frame/' + imtype + '/aligned/' + targname + '_' + filter + ' aligned_error.fit')
    print('\n')
    fits.writeto(dir + 'Light Frame/' + imtype + '/aligned/' + targname + '_' + filter + ' aligned_error.fit', error_image, header, overwrite = True)
    
    return error_image
  
###########################################################################################################################################################################################################################################################################################################################################################################################################################################
  
def star_extractor(dir, targname, filter = 'Blue', fwhm = 10.0, sigma = 4.0, standard = False):

    '''
    PURPOSE:
            Extract the positions of all the sources in the image that resides in 'dir'.

    IMPUTS:
                 [dir]:  (string)  Directory of files to be calibrated
            [targname]:  (string)  Name of the primary source of interest

    OPT. INPUTS:
                 [fwhm]:  (float)  FWHM of the sources (default is 10.0)
                [sigma]:  (float)  Threshold of the detections (default is 4.0)
               [filter]:  (string) Name of the filter the frames were taken in (default is Blue since this shouldn't matter)

    RETURNS:
             [star_pos]:  (list, float)  x and y-centroids of every source in the given image

    AUTHOR:
            Julio M. Morales, December 01, 2021
    '''
    
    # True if we are aligning the primary source of interest
    if (standard == False):

        imtype = 'source'

    # True if we are aligning the standard images
    else:

        imtype = 'standard'

    # Pull the data from the images
    image = fits.getdata(dir + 'Light Frame/' + imtype + '/aligned/' + targname + '_' + filter + ' aligned.fits')

    # Estimate the background as the Mean Absolute Deviation (MAD)
    bkg_mad = pd.DataFrame(np.hstack(image)).mad(skipna = True)[0]

    # Detect all of the sources in the images
    daofind = DAOStarFinder(fwhm = fwhm, threshold = sigma * bkg_mad)
    sources = daofind(image)

    # Positions of source
    xpos = np.array(sources['xcentroid'])           
    ypos = np.array(sources['ycentroid'])

    # Define the file that contains the circled sources
    f = open(dir + 'Light Frame/' + imtype + '/aligned/' + targname + '_green_cricles.reg', 'w') 
    for i in range(0, len(xpos)):
        # Write the circled reigons to the file
        f.write('circle ' + str(xpos[i]) + ' ' + str(ypos[i]) + ' ' + str(fwhm) + '\n')
    # Close the file
    f.close()

    # Pair up the positions as a tupple and store them as a list
    star_pos = list(zip(xpos, ypos))

    return star_pos
  
###########################################################################################################################################################################################################################################################################################################################################################################################################################################
  
def measure_photometry(dir, targname, filter, star_pos, aperture_radius, error_array, standard = False):

    '''
    PURPOSE:
            Extract the positions of all the sources in the image that resides in 'dir'.

    IMPUTS:
                    [dir]:  (string)          Directory of files to be calibrated
               [targname]:  (string)          Name of the primary source of interest
                 [filter]:  (string)          Name of the filter the frames were taken in.
               [star_pos]:  (list, float)     x and y-centroids of every source in the given image
        [aperture_radius]:  (float)           Radius of the aperture to extract photometry
            [error_array]:  (np.array, float) Error image
    
    OPT. INPUTS:
               [standard]:  (boolean) True if you are working with images of the standard star.  False if for the source.

    RETURNS:
             [phot_table]:  (np.array, float)  Dataframe containing all of the photometric data.

    AUTHOR:
            Julio M. Morales, December 01, 2021
    '''

    # True if we are aligning the primary source of interest
    if (standard == False):

        imtype = 'source'

    # True if we are aligning the standard images
    else:

        imtype = 'standard'

    # Pull the data from the image
    image = fits.getdata(dir + 'Light Frame/' + imtype + '/aligned/' + targname + '_' + filter + ' aligned.fits')

    # Define inner and outer sky radii in terms of aperture radius
    sky_inner = aperture_radius + 5
    sky_outer = sky_inner + 10
    
    # Define a circular aperture and anulus
    starapertures = CircularAperture(star_pos, r = aperture_radius)
    skyannuli =     CircularAnnulus(star_pos, r_in = sky_inner, r_out = sky_outer)
    phot_apers =    [starapertures, skyannuli]
    
    # Define the dataframe containing the photometric data
    phot_table = aperture_photometry(image, phot_apers, error = error_array)
        
    # Calculate mean background in annulus and subtract from aperture flux
    bkg_mean =       phot_table['aperture_sum_1'] / skyannuli.area
    bkg_starap_sum = bkg_mean * starapertures.area
    final_sum =      phot_table['aperture_sum_0'] - bkg_starap_sum

    # Append the data to the dataframe
    phot_table['bg_subtracted_star_counts'] = final_sum
    
    # Calculate the error on the photometry
    bkg_mean_err = phot_table['aperture_sum_err_1'] / skyannuli.area
    bkg_sum_err =  bkg_mean_err * starapertures.area

    # Append the data to the dataframe
    phot_table['bg_sub_star_cts_err'] = np.sqrt((phot_table['aperture_sum_err_0']**2) + (bkg_sum_err**2)) 

    return phot_table
  
###########################################################################################################################################################################################################################################################################################################################################################################################################################################
  
def standard_photometry(dir, targname, filter_list, index, bm, ubm, vm, uvm, rm, urm):

    '''
    PURPOSE:
            Calibrate the photometry for the standard star.

    IMPUTS:
                    [dir]:  (string)        Directory of files to be calibrated
               [targname]:  (string)        Name of the standard star
            [filter_list]:  (list, string)  List of the names of the filters the frames were taken in.
                  [index]:  (integer)       Index of the standard star in the dataframe
                     [bm]:  (float)         True B-band magnitude of the standard star
                    [ubm]:  (float)         Uncertainity in B-band magnitude of the standard star
                     [vm]:  (float)         True V-band magnitude of the standard star
                    [uvm]:  (float)         Uncertainity in V-band magnitude of the standard star
                     [rm]:  (float)         True R-band magnitude of the standard star
                    [urm]:  (float)         Uncertainity in R-band magnitude of the standard star
    
    RETURNS:
          [std_fluxtable]:  (np.array, float)  Dataframe containing all of the calibrated photometric data for the standard star.

    AUTHOR:
            Julio M. Morales, December 01, 2021
    '''

    # Loop over the filters
    for filter in filter_list:

        # Pull the data from the image of a given filter
        image = fits.getdata(dir + 'Light Frame/standard/aligned/' + targname + '_' + filter + ' aligned.fits')

        # Call on the background estimator function to calculate the background error
        bkg_error = bg_error_estimate(dir, targname, filter, standard = True)

        # Call on the star extractor to calculate the source centroids
        star_pos =  star_extractor(dir, targname, standard = True)

        # Call on the measure_photometry to extract photometry from the standard star
        phot_table = measure_photometry(dir, targname, filter, star_pos, aperture_radius = 19.5, error_array = bkg_error, standard = True)

        # True if the filter is Blue
        if   (filter == filter_list[0]):
            std_fluxtable_1 = phot_table

        # True if the filter is Visual
        elif (filter == filter_list[1]):
            std_fluxtable_2 = phot_table

        # True if the filter is Red
        elif (filter == filter_list[2]):
            std_fluxtable_3 = phot_table

    
    # Define the columns of the dataframe to be made
    columns = ['id', 'xcenter', 'ycenter', filter_list[0] + 'flux', filter_list[0] + 'fluxerr', filter_list[1] + 'flux', filter_list[1] + 'fluxerr', filter_list[2] + 'flux', filter_list[2] + 'fluxerr']
    
    # Define the dataframe containing all of the photometry
    std_fluxtable = pd.DataFrame(
    {'id'                      : std_fluxtable_1['id'],
     'xcenter'                 : std_fluxtable_1['xcenter'],
     'ycenter'                 : std_fluxtable_1['ycenter'],
     filter_list[0] + 'flux'   : std_fluxtable_1['bg_subtracted_star_counts'],
     filter_list[0] + 'fluxerr': std_fluxtable_1['bg_sub_star_cts_err'], 
     filter_list[1] + 'flux'   : std_fluxtable_2['bg_subtracted_star_counts'],
     filter_list[1] + 'fluxerr': std_fluxtable_2['bg_sub_star_cts_err'],
     filter_list[2] + 'flux'   : std_fluxtable_3['bg_subtracted_star_counts'],
     filter_list[2] + 'fluxerr': std_fluxtable_3['bg_sub_star_cts_err']}, columns = columns)
    
    # Loop through all of the filters
    for filter in filter_list:

        # Pull the header from the images
        std_header = fits.getheader(dir + 'Light Frame/standard/aligned/' + targname + '_' + filter + ' aligned.fits')

        # Pull the exposure time from the images header
        std_exptime = std_header['EXPTIME']

        # True if the filter is Blue
        if (filter == filter_list[0]):
            
            # Calculate the standard stars flux/1 sec
            std_flux_1sec =     std_fluxtable_1[index]['bg_subtracted_star_counts']  / std_exptime
            std_flux_1sec_err = std_fluxtable_1[index]['bg_sub_star_cts_err'] / std_exptime

            # Append the data to the dataframe
            std_fluxtable[filter + 'flux_1sec'] =     std_flux_1sec
            std_fluxtable[filter + 'flux_1sec_err'] = std_flux_1sec_err

            # Calculate the standard star instrumental magnitude and append to the dataframe
            std_flux_inst =  -2.5 * np.log10(std_fluxtable[filter + 'flux_1sec'])
            std_fluxtable[filter + '_inst'] = std_flux_inst
            std_flux_inst_err = 2.5 * 0.434 * (std_flux_1sec_err / std_flux_1sec)
            std_fluxtable[filter + '_inst_err'] = std_flux_inst_err

            m =  bm
            um = ubm

            # Calculate the zeropoint magnitudes and append to the dataframe
            magzp = m - std_fluxtable[filter + '_inst']
            std_fluxtable[filter + '_magzp'] = magzp
            magzp_error = np.sqrt((std_fluxtable[filter + '_inst'])**2 + um**2)
            std_fluxtable[filter + '_magzp_err'] = magzp_error

        # True if the filter is Visual
        elif (filter == filter_list[1]):

            # Calculate the standard stars flux/1 sec
            std_flux_1sec =     std_fluxtable_2[index]['bg_subtracted_star_counts']  / std_exptime
            std_flux_1sec_err = std_fluxtable_2[index]['bg_sub_star_cts_err'] / std_exptime

            # Append the data to the dataframe
            std_fluxtable[filter + 'flux_1sec'] =     std_flux_1sec
            std_fluxtable[filter + 'flux_1sec_err'] = std_flux_1sec_err

            # Calculate the standard star instrumental magnitude and append to the dataframe
            std_flux_inst =  -2.5 * np.log10(std_fluxtable[filter + 'flux_1sec'])
            std_fluxtable[filter + '_inst'] = std_flux_inst
            std_flux_inst_err = 2.5 * 0.434 * (std_flux_1sec_err / std_flux_1sec)
            std_fluxtable[filter + '_inst_err'] = std_flux_inst_err

            m =  vm
            um = uvm

            # Calculate the zeropoint magnitudes and append to the dataframe
            magzp = m - std_fluxtable[filter + '_inst']
            std_fluxtable[filter + '_magzp'] = magzp
            magzp_error = np.sqrt((std_fluxtable[filter + '_inst'])**2 + um**2)
            std_fluxtable[filter + '_magzp_err'] = magzp_error

        # True if the filter is Red
        elif (filter == filter_list[2]):

            # Calculate the standard stars flux/1 sec
            std_flux_1sec =     std_fluxtable_3[index]['bg_subtracted_star_counts']  / std_exptime
            std_flux_1sec_err = std_fluxtable_3[index]['bg_sub_star_cts_err'] / std_exptime

            # Append the data to the dataframe
            std_fluxtable[filter + 'flux_1sec'] =     std_flux_1sec
            std_fluxtable[filter + 'flux_1sec_err'] = std_flux_1sec_err

            # Calculate the standard star instrumental magnitude and append to the dataframe
            std_flux_inst =  -2.5 * np.log10(std_fluxtable[filter + 'flux_1sec'])
            std_fluxtable[filter + '_inst'] = std_flux_inst
            std_flux_inst_err = 2.5 * 0.434 * (std_flux_1sec_err / std_flux_1sec)
            std_fluxtable[filter + '_inst_err'] = std_flux_inst_err

            m =  rm
            um = urm

            # Calculate the standard star instrumental magnitude and append to the dataframe
            magzp = m - std_fluxtable[filter + '_inst']
            std_fluxtable[filter + '_magzp'] = magzp
            magzp_error = np.sqrt((std_fluxtable[filter + '_inst'])**2 + um**2)
            std_fluxtable[filter + '_magzp_err'] = magzp_error


    return std_fluxtable
  
 ###########################################################################################################################################################################################################################################################################################################################################################################################################################################
  
 def source_photometry(dir, targname, std_name, filter_list, standard_index, bm, ubm, vm, uvm, rm, urm):

    '''
    PURPOSE:
            Calibrate the photometry for the primary source of interest.

    IMPUTS:
                  [dir]:  (string)        Directory of files to be calibrated
             [targname]:  (string)        Name of the primary source of interest
             [std_name]:  (string)        Name of the standard star
          [filter_list]:  (list, string)  List of the names of the filters the frames were taken in.
       [standard_index]:  (integer)       Index of the standard star in the standard star dataframe
                   [bm]:  (float)         True B-band magnitude of the standard star
                  [ubm]:  (float)         Uncertainity in B-band magnitude of the standard star
                   [vm]:  (float)         True V-band magnitude of the standard star
                  [uvm]:  (float)         Uncertainity in V-band magnitude of the standard star
                   [rm]:  (float)         True R-band magnitude of the standard star
                  [urm]:  (float)         Uncertainity in R-band magnitude of the standard star
    
    RETURNS:
            [fluxtable]:  (np.array, float)  Dataframe containing all of the calibrated photometric data for the source of interest.

    AUTHOR:
            Julio M. Morales, December 01, 2021
    '''

    # Loop over the filters
    for filter in filter_list:

        # Pull the data from the image of a given filter
        image = fits.getdata(dir + 'Light Frame/source/aligned/' + targname + '_' + filter + ' aligned.fits')

        # Call on the background estimator function to calculate the background error
        bkg_error = bg_error_estimate(dir, targname, filter)

        # Call on the star extractor to calculate the source centroids
        star_pos =  star_extractor(dir, targname)

        # Call on the measure_photometry to extract photometry from the standard star
        phot_table = measure_photometry(dir, targname, filter, star_pos, aperture_radius = 19.5, error_array = bkg_error)

        # True if the filter is Blue
        if  (filter == filter_list[0]):
            fluxtable_1 = phot_table

        # True if the filter is Visual
        elif (filter == filter_list[1]):
            fluxtable_2 = phot_table

        # True if the filter is Red
        elif (filter == filter_list[2]):
            fluxtable_3 = phot_table

    # Define the columns of the dataframe to be made
    columns = ['id', 'xcenter', 'ycenter', filter_list[0] + 'flux', filter_list[0] + 'fluxerr', filter_list[1] + 'flux', filter_list[1] + 'fluxerr', filter_list[2] + 'flux', filter_list[2] + 'fluxerr']

    # Define the dataframe containing all of the photometry
    fluxtable = pd.DataFrame(
    {'id'                      : fluxtable_1['id'],
     'xcenter'                 : fluxtable_1['xcenter'],
     'ycenter'                 : fluxtable_1['ycenter'],
     filter_list[0] + 'flux'   : fluxtable_1['bg_subtracted_star_counts'],
     filter_list[0] + 'fluxerr': fluxtable_1['bg_sub_star_cts_err'], 
     filter_list[1] + 'flux'   : fluxtable_2['bg_subtracted_star_counts'],
     filter_list[1] + 'fluxerr': fluxtable_2['bg_sub_star_cts_err'],
     filter_list[2] + 'flux'   : fluxtable_3['bg_subtracted_star_counts'],
     filter_list[2] + 'fluxerr': fluxtable_3['bg_sub_star_cts_err']}, columns = columns)
    
    # Call on the standard star photometry
    std_fluxtable = standard_photometry(dir, std_name, filter_list, standard_index, bm, ubm, vm, uvm, rm, urm)

    # Loop over the filters
    for filter in filter_list:

        # Pull the header and it's exposure time
        header = fits.getheader(dir + 'Light Frame/source/aligned/' + targname + '_' + filter + ' aligned.fits')
        exptime = header['EXPTIME']

        # True if the filter is Blue
        if (filter == filter_list[0]):
            
            # Calculate the sources flux/1 sec
            flux_1sec =     fluxtable_1['bg_subtracted_star_counts']  / exptime
            flux_1sec_err = fluxtable_1['bg_sub_star_cts_err'] / exptime

            # Append the data to the dataframe
            fluxtable[filter + 'flux_1sec'] =     flux_1sec
            fluxtable[filter + 'flux_1sec_err'] = flux_1sec_err

            # Calculate the sources instrumental magnitude and append to the dataframe
            flux_inst =  -2.5 * np.log10(fluxtable[filter + 'flux_1sec'])
            fluxtable[filter + '_inst'] = flux_inst
            flux_inst_err = 2.5 * 0.434 * (flux_1sec_err / flux_1sec)
            fluxtable[filter + '_inst_err'] = flux_inst_err

            # Calculate the zeropoint magnitudes and append to the dataframe
            magzp =     std_fluxtable[filter + '_magzp']
            magzp_err = std_fluxtable[filter + '_magzp_err']

            # Calculate the calibrated magnitude for the sources
            mcal = magzp + fluxtable[filter + '_inst']
            fluxtable[filter + 'mag'] = mcal
            mcal_err = np.sqrt(magzp_err**2 + (fluxtable[filter + '_inst_err'])**2)
            fluxtable[filter + 'mag_err'] = mcal_err

        # True if the filter is Visual
        elif (filter == filter_list[1]):

            # Calculate the sources flux/1 sec
            flux_1sec =     fluxtable_2['bg_subtracted_star_counts']  / exptime
            flux_1sec_err = fluxtable_2['bg_sub_star_cts_err'] / exptime

            # Append the data to the dataframe
            fluxtable[filter + 'flux_1sec'] =     flux_1sec
            fluxtable[filter + 'flux_1sec_err'] = flux_1sec_err

            # Calculate the sources instrumental magnitude and append to the dataframe
            flux_inst =  -2.5 * np.log10(fluxtable[filter + 'flux_1sec'])
            fluxtable[filter + '_inst'] = flux_inst
            flux_inst_err = 2.5 * 0.434 * (flux_1sec_err / flux_1sec)
            fluxtable[filter + '_inst_err'] = flux_inst_err

            # Calculate the zeropoint magnitudes and append to the dataframe
            magzp =     std_fluxtable[filter + '_magzp']
            magzp_err = std_fluxtable[filter + '_magzp_err']

            # Calculate the calibrated magnitude for the sources
            mcal = magzp + fluxtable[filter + '_inst']
            fluxtable[filter + 'mag'] = mcal
            mcal_err = np.sqrt(magzp_err**2 + (fluxtable[filter + '_inst_err'])**2)
            fluxtable[filter + 'mag_err'] = mcal_err

        # True if the filter is Red
        elif (filter == filter_list[2]):

            # Calculate the sources flux/1 sec
            flux_1sec =     fluxtable_3['bg_subtracted_star_counts']  / exptime
            flux_1sec_err = fluxtable_3['bg_sub_star_cts_err'] / exptime

            # Append the data to the dataframe
            fluxtable[filter + 'flux_1sec'] =     flux_1sec
            fluxtable[filter + 'flux_1sec_err'] = flux_1sec_err

            # Calculate the sources instrumental magnitude and append to the dataframe
            flux_inst =  -2.5 * np.log10(fluxtable[filter + 'flux_1sec'])
            fluxtable[filter + '_inst'] = flux_inst
            flux_inst_err = 2.5 * 0.434 * (flux_1sec_err / flux_1sec)
            fluxtable[filter + '_inst_err'] = flux_inst_err

            # Calculate the zeropoint magnitudes and append to the dataframe
            magzp =     std_fluxtable[filter + '_magzp']
            magzp_err = std_fluxtable[filter + '_magzp_err']

            # Calculate the calibrated magnitude for the sources
            mcal = magzp + fluxtable[filter + '_inst']
            fluxtable[filter + 'mag'] = mcal
            mcal_err = np.sqrt(magzp_err**2 + (fluxtable[filter + '_inst_err'])**2)
            fluxtable[filter + 'mag_err'] = mcal_err

    # Calculate the colors
    color_1 = fluxtable[filter_list[0] + 'mag'] - fluxtable[filter_list[1] + 'mag']
    color_err_1 = np.sqrt((fluxtable[filter_list[0] + 'mag_err'])**2 + (fluxtable[filter_list[1] + 'mag_err'])**2)

    color_2 = fluxtable[filter_list[0] + 'mag'] - fluxtable[filter_list[2] + 'mag']
    color_err_2 = np.sqrt((fluxtable[filter_list[0] + 'mag_err'])**2 + (fluxtable[filter_list[2] + 'mag_err'])**2)

    color_3 = fluxtable[filter_list[1] + 'mag'] - fluxtable[filter_list[2] + 'mag']
    color_err_3 = np.sqrt((fluxtable[filter_list[1] + 'mag_err'])**2 + (fluxtable[filter_list[2] + 'mag_err'])**2)

    # Apend the color data to the dataframe
    fluxtable[filter_list[0] + '_' + filter_list[1] + '_color'] =     color_1
    fluxtable[filter_list[0] + '_' + filter_list[1] + '_color_err'] = color_err_1

    fluxtable[filter_list[0] + '_' + filter_list[2] + '_color'] =     color_2
    fluxtable[filter_list[0] + '_' + filter_list[2] + '_color_err'] = color_err_2

    fluxtable[filter_list[1] + '_' + filter_list[2] + '_color'] =     color_3
    fluxtable[filter_list[1] + '_' + filter_list[2] + '_color_err'] = color_err_3

    return fluxtable
 
###########################################################################################################################################################################################################################################################################################################################################################################################################################################
 
def linear_func(x, m, b):

    '''
    PURPOSE:
            Function to calculate a straight line.
    
    INPUTS:
            [x]: (np.array, float) The positions at which to evaluate the function
            [m]: (float)           The slope of the straight line
            [b]: (float)           The intercept of the straight line
        
    RETURNS:
            [y]: (np.array, float) The values predicted by our model at positions x
    
    AUTHOR:
            Julio M. Morales
    '''

    y = m*x + b

    return y
