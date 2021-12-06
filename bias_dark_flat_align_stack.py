import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.signal
import os 
import glob 
import scipy.ndimage.interpolation as interp
from astropy.visualization import ZScaleInterval
from astropy.stats import sigma_clipped_stats
from scipy.ndimage.filters import gaussian_filter
from astropy.io import fits 

# Mount the Google drive
from google.colab import drive
drive.mount('/content/drive')

##############################################################################################################################################################################################

def filesorter(filename, dir, foldername):

  '''
  PURPOSE: 
          Checks if the directory  (dir + foldername + filename) exists, then creates the directory if it doesn't.
  INPUTS:  
          filename:   [string] Name of the file
          dir:        [string] Directory where you want things to be saved
          foldername: [string] Name of the folder you want to check/generate
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

#######################################################################################################################################################################################################################################################################

def mediancombine(first_letter, dir):

  '''
  PURPOSE: 
          Median combine all the images in the list fed to the function (ONLY COMBINE ONE SET OF FRAMES AT A TIME)
  INPUTS:  
          first_letter:  [string]  First letter(s) of the files to be combined
          dir:           [string]  Directory that contains the images to be median combined
  RETURNS:  
          med_frame:  [np.array, float] Median combined image
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
  fits_stack = np.zeros((imsize_y, imsize_x , n))

  if (first_letter == 'dark_subtracted_'):

    for ii in range(0, n):

        im = fits.getdata(filelist[ii])
        norm_im =  im/np.median(im) # finish new line here to normalize flats
        fits_stack[:,:,ii] = norm_im
        
    # Add your own comment describing this step    
    med_frame = np.median(fits_stack, axis = 2)

  else:

    # Iterate through n and assign image to the array
    for ii in range(0, n):  
    
        im = fits.getdata(files[ii])
        fits_stack[:,:,ii] = im
        
    # Takes the median of the stacked files and combines them
    med_frame = np.median(fits_stack, axis = 2)

  return med_frame

#######################################################################################################################################################################################################################################################################

def bias_subtract(first_letter, master_bias_path, dir):
  
  '''
  PURPOSE: 
          This function takes science images and subtracts the master bias image from them, and writes the result to a new fits file.
  INPUTS:
          filename:          [string]  Name of the files to be bias subtracted
          master_bias_path:  [string]  Pathway to the master bias image
          dir:               [string]  Directory of files to be calibrated
  
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

#######################################################################################################################################################################################################################################################################

def dark_subtract(first_letter, master_dark_path, dir):
  
  '''
  PURPOSE: 
          This function takes science images, subtracts the master dark image from them, and writes the result to a new fits file.
  INPUTS:  
          filename:          [string]  Name of the files to be dark subtracted
          master_dark_path:  [string]  Pathway to the master dark image
          dir:               [string]  Directory of files to be calibrated
  
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

#######################################################################################################################################################################################################################################################################

def flat_field(first_letter, master_flat_path, dir):
  
  '''
  PURPOSE: 
          This function takes science images, divides them by the master flat image, and writes the result to a new fits file.
  INPUTS:  
          filename:          [string]  Name of the files to be flat-fielded
          master_flat_path:  [string]  Pathway to the master flat image
          dir:               [string]  Directory of files to be calibrated
  
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
    flat_fielded_data = data.astype('float')/master_flat_image.astype('float')

    # Write to new fits file with the header of the orignal file
    fits.writeto(dir + 'flat_fielded_' + image.split('/')[-1], flat_fielded_data, header = header, overwrite = True)

  return

#######################################################################################################################################################################################################################################################################

def master_frame_generator(first_letter, dir, filter):

  '''
  PURPOSE:  
          To bias subtract, dark-subtract, and flatfield a set of images in a given filter all in one step.
  INPUTS:   
          first_letter:      [string]  First letter(s) of the files to be reduced
          dir:               [string]  Directory of files to be calibrated
          filter:            [string]  Name of the filter the frames were taken in
  RETURNS:  
          master_bias_path:  [string]  Path to the master bias frame
          master_dark_path:  [string]  Path to the master dark frame
          master_flat_path:  [string]  Path to the master flat frame
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

  # Bias subtract the dark frames, then create master dark
  bias_subtract(first_letter, dir + 'Bias Frame/Master_Bias.fit', dir + 'Dark Frame/' + exposure + '/')
  master_dark = mediancombine('bias_subtracted_', dir + 'Dark Frame/' + exposure + '/')
  fits.writeto(dir + 'Dark Frame/' + exposure + '/Master_Dark.fit', master_dark, overwrite = True)

  # Bias and dark subtract the flat frames, then create the master flat
  bias_subtract(first_letter, dir + 'Bias Frame/Master_Bias.fit', dir + 'Flat Field/' + filter + '/')
  dark_subtract('bias_subtracted_', dir + 'Dark Frame/' + exposure + '/Master_Dark.fit', dir + 'Flat Field/' + filter + '/')
  master_flat = mediancombine('dark_subtracted_', dir + 'Flat Field/' + filter + '/')
  fits.writeto(dir + 'Flat Field/' + filter + '/Master_Flat.fit', master_flat, overwrite = True)

  # Define the pathways for each master frame
  master_bias_path = dir + 'Bias Frame/Master_Bias.fit'
  master_dark_path = dir + 'Dark Frame/' + exposure + '/Master_Dark.fit'
  master_flat_path = dir + 'Flat Field/' + filter + '/Master_Flat.fit'

  return [master_bias_path, master_dark_path, master_flat_path]

#######################################################################################################################################################################################################################################################################

def cross_image(im1, im2, **kwargs):

    """
    PURPOSE: 
            This function performs cross-correlation by slicing each image, and subtracting the 
            mean of each image from itself.  It then performs a fast--fourier transform on the images.  
            It then calculates the shifts required for alignment by comparing peak pixel value locations.
    INPUTS:  
            im1:             [np.array, float]  First image to be cross correlated
            im2:             [np.array, float]  Second image to be cross correlated
            **kwargs:        [list]             Variable-length keyword list
    RETURNS:  
            xshift, yshift:  [list, float]  x and y shifts required to align the images
    AUTHOR:  
            Connor E. Robinson
    """

    # The type cast into 'float' is to avoid overflows:
    im1_gray = im1.astype('float')
    im2_gray = im2.astype('float')

    # Enable a trimming capability using keyword argument option.
    if 'boxsize' in kwargs:
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
  
#######################################################################################################################################################################################################################################################################
  
def shift_image(image, xshift, yshift):

    '''
    PURPOSE: 
            This function takes as input an image, and the x and y-shifts to be executed.  
            It then performs this shift by using the np.roll() function which shifts each pixel in a specifced direction.  
            The amount by which it is shifted depends on the inputs x-shift and y-shift.
    INPUTS:
            image:  [np.array, float]  Image to be shifted
            xshift: [float]            Amount that the image will be shifted by in the x-direction
            yshift: [float]            Amount that the image will be shifted by in the y-direction
    RETURNS: 
            [np.array, float]  Rolled image
    AUTHOR:
            Connor E. Robinson
    '''

    return np.roll(np.roll(image, int(yshift), axis = 1), int(xshift), axis = 0)
  
#######################################################################################################################################################################################################################################################################
  
def align_N_stack(targname, first_letter, dir, filter):

  '''
  PURPOSE: 
          Align and stack fully reduced images that have been bias, dark, subtracted, and flat-fielded.
  INPUTS:  
          targname:      [string]  Name of the object your imaging
          first_letter:  [string]  First letter(s) of the files you will be aligning and stacking
          dir:           [string]  Directory where your fully reduced images are
          filter:        [string]  Filter of the images you want to reduce
  RETURNS:
          median_image:  [np.array, float]  Aligned and stacked images saved into a new foder named 'stacked' in the directory 'dir'
  AUTHOR:  
          Connor E. Robinson
  '''

  # Using glob, make list of all reduced images of current target in all filters.
  # Complete the following line to create a list of the correct images to be shifted (use wildcards!):
  imlist = glob.glob(dir + filter + '/' + first_letter + '*.fit')

  # Check to make sure that your new list has the right files:
  print("All files to be aligned: \n", imlist)
  print('\n') # adding some space to the print statements, '/n' means new line
  
  # Open first image = master image; all other images of same target will be aligned to this one.
  im1, hdr1 = fits.getdata(imlist[0], header = True)
  print("Aligning all images to:", imlist[0])
  
  print('\n') # adding some space to the print statements

  # What is the following for loop doing?
  # Your answer:  The loop takes every pixel in a given indexed location and shifts them for the master image
  
  xshifts = {}
  yshifts = {}
  for index, filename in enumerate(imlist):
      im, hdr = fits.getdata(filename, header = True)
      xshifts[index], yshifts[index] = cross_image(im1, im, boxsize = 1000)
      print("Shift for image", index, "is", xshifts[index], yshifts[index])

  # Calculate trim edges of new median stacked images so all stacked images of each target have same size 
  max_x_shift = int(np.max([xshifts[x] for x in xshifts.keys()]))
  max_y_shift = int(np.max([yshifts[x] for x in yshifts.keys()]))

  print('   Max x - shift = {0}, max y - shift = {1} (pixels)'.format(max_x_shift,max_y_shift))
  

  scilist = []

  # Populates scilist with a list of FITS files matching *only* the selected filter:
  for fitsfile in imlist:
    
    header = fits.getheader(fitsfile)

    if header['FILTER'] == filter:
      scilist.append(fitsfile)
    else:
      pass

  if len(scilist) < 1:
    print("Warning! No files in scilist. Your path is likely incorrect.")
  
  # Ensure each scilist entry has the right filter:
  for fitsfile in scilist:
      print('File:  ', fitsfile)
      print('\n') 
      print('Filter:  ', filter)
  
  nfiles = len(scilist)
  print('Stacking ', nfiles, filter, ' science frames')

  # Define new array with same size as master image
  image_stack = np.zeros([im1.shape[0], im1.shape[1], len(scilist)])

  xshifts_filt = {}
  yshifts_filt = {}

  # This loop completes the shifts for the rest of the images in relation to the master image
  for index, filename in enumerate(scilist):

      im, hdr = fits.getdata(filename, header = True)
      xshifts_filt[index], yshifts_filt[index] = cross_image(im1, im, boxsize = 1000)
      image_stack[:, :, index] = shift_image(im, xshifts_filt[index], yshifts_filt[index])      

  # Take the median of the image stack while maintaing the dimensions of the array
  median_image = np.median(image_stack, axis = -1)

  # Sets the new image boundaries
  if (max_x_shift > 0) & (max_y_shift > 0):
      median_image = median_image[max_x_shift:-max_x_shift, max_y_shift:-max_y_shift]

  # Make a new directory in your dir for the new stacked fits files
  if os.path.isdir(dir + 'stacked') == False:
      os.mkdir(dir + 'stacked')
      print('\n Making new subdirectory for stacked images:', dir + 'stacked \n')
      
  
  # Save the final stacked images into your new folder
  fits.writeto(dir + '/stacked/' + targname + '_' + filter + 'stack.fits', median_image, overwrite = True)
  print('   Wrote FITS file ', targname + '_' + filter + 'stack.fits', 'in ', dir + 'stacked', '\n')
  print('\n Done stacking!')

  return median_image

##################################################################################################################################################################################

def science_file_sorter(dir, source_first_letter, standard_first_letter, is_standard = False):

  '''
  PURPOSE: 
          Takes in the directory where all of the files taken reside, it then organizes the files by 
          calibration type, filter, exposure time, and by source vs. standard type.
  INPUTS:  
          dir:                    [string]  Directory where all the files reside
          source_first_letter:    [string]  The first letter(s) of the files taken for the primary source of interest
          standard_first_letter:  [string]  The first letter(s) of the files taken for the standard star
  
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

#######################################################################################################################################################################################################################################################################

def source_image_calibrator(targname, first_letter, dir, filter, is_standard = False):

  '''
  PURPOSE:
          To bias subtract, dark-subtract, and flatfield a set of images in a given filter all in one step.
  IMPUTS:
          first_letter:  [string]  First letter(s) of the files to be reduced
          dir:           [string]  Directory of files to be calibrated
          filter:        [string]  Name of the filter the frames were taken in
  RETURNS:
          final_image:   [np.array, float]  Fully calibrated images that have been bias, dark subtracted, flat-fielded, aligned, and stacked.
  AUTHOR:
          Julio M. Morales, November 01, 2021
  '''

  # Generate master frames
  master_frames = master_frame_generator(first_letter, dir, filter)

  # Bias subtract the science frames
  bias_subtract(first_letter, master_frames[0], dir + 'Light Frame/source/' + filter + '/')

  # Dark subtract the bias subtracted frames
  dark_subtract('bias_subtracted_', master_frames[1], dir + 'Light Frame/source/' + filter + '/')

  # Flat-field the bias and dark subtracted frames
  flat_field('dark_subtracted_', master_frames[2], dir + 'Light Frame/source/' + filter + '/')

  # Align and stack the reduced images
  fully_reduced_image = align_N_stack(targname, 'flat_fielded_', dir + 'Light Frame/source/', filter)
  
  return fully_reduced_image

#######################################################################################################################################################################################################################################################################

def standard_image_calibrator(targname, first_letter, dir, filter):

  '''
  PURPOSE:  
          To bias subtract, dark-subtract, and flat-field a set of images in a given filter all in one step.
  IMPUTS:   
          first_letter:  [string]  First letter(s) of the files to be reduced
          dir:           [string]  Directory of files to be calibrated
          filter:        [string]  Name of the filter the frames were taken in
  RETURNS:   
          final_image:   [np.array, float]  Fully calibrated images that have been bias, dark subtracted, flat-fielded, aligned, and stacked.
  AUTHOR:  
          Julio M. Morales, November 01, 2021
  '''

  # Get the exopsure time of the source frames to know which dark frames we need to use
  science_files = glob.glob(dir + 'Light Frame/source/' + filter + '/*.fit')
  header = fits.getheader(science_files[0])
  exposure = str(header['EXPTIME'])

  # Generate master frames
  master_bias_path = dir + 'Bias Frame/Master_Bias.fit'
  master_dark_path = dir + 'Dark Frame/' + exposure + '/Master_Dark.fit'
  matster_flat_path = dir + 'Flat Field/' + filter + '/Master_Flat.fit'

  # Bias subtract the science frames
  bias_subtract(first_letter, master_bias_path, dir + 'Light Frame/standard/' + filter + '/')

  # Dark subtract the bias subtracted frames
  dark_subtract('bias_subtracted_', master_dark_path, dir + 'Light Frame/standard/' + filter + '/')

  # Flat-field the bias and dark subtracted frames
  flat_field('dark_subtracted_', matster_flat_path, dir + 'Light Frame/standard/' + filter + '/')

  # Align and stack the reduced images
  fully_reduced_standard_image = align_N_stack(targname, 'flat_fielded_', dir + 'Light Frame/standard/', filter)

  return fully_reduced_standard_image

def align(targname, first_letter, dir):

  '''
  PURPOSE: 
          Align and stack fully reduced images that have been bias, dark, subtracted, and flat-fielded.
  INPUTS:  
          targname:      [string]  Name of the object your imaging
          first_letter:  [string]  First letter(s) of the files you will be aligning and stacking
          dir:           [string]  Directory where your fully reduced images are
          filter:        [string]  Filter of the images you want to reduce
  RETURNS:
          median_image:  [np.array, float]  Aligned and stacked images saved into a new foder named 'stacked' in the directory 'dir'
  AUTHOR:  
          Connor E. Robinson
  '''

  # Using glob, make list of all reduced images of current target in all filters.
  # Complete the following line to create a list of the correct images to be shifted (use wildcards!):
  imlist = glob.glob(dir + 'Light Frame/source/stacked/*.fits')

  # Check to make sure that your new list has the right files:
  print("All files to be aligned: \n", imlist)
  print('\n') # adding some space to the print statements, '/n' means new line
  
  # Open first image = master image; all other images of same target will be aligned to this one.
  im1, hdr1 = fits.getdata(imlist[0], header = True)
  print("Aligning all images to:", imlist[0])
  
  print('\n') # adding some space to the print statements

  # What is the following for loop doing?
  # Your answer:  The loop takes every pixel in a given indexed location and shifts them for the master image
  
  xshifts = {}
  yshifts = {}

  for index, filename in enumerate(imlist):
      im, hdr = fits.getdata(filename, header = True)
      xshifts[index], yshifts[index] = cross_image(im1, im, boxsize = 1000)
      print("Shift for image", index, "is", xshifts[index], yshifts[index])

  # Calculate trim edges of new median stacked images so all stacked images of each target have same size 
  max_x_shift = int(np.max([xshifts[x] for x in xshifts.keys()]))
  max_y_shift = int(np.max([yshifts[x] for x in yshifts.keys()]))

  print('   Max x - shift = {0}, max y - shift = {1} (pixels)'.format(max_x_shift,max_y_shift))
  
  scilist = imlist

  if len(scilist) < 1:
    print("Warning! No files in scilist. Your path is likely incorrect.")
  
  
  nfiles = len(scilist)
  print('Aligning ', nfiles, ' science frames')

  # Define new array with same size as master image
  image_stack = np.zeros([im1.shape[0], im1.shape[1], len(scilist)])

  xshifts_filt = {}
  yshifts_filt = {}

  # This loop completes the shifts for the rest of the images in relation to the master image
  for index, filename in enumerate(scilist):

    im, hdr = fits.getdata(filename, header = True)

    filter = hdr['FILTER']
    xshifts_filt[index], yshifts_filt[index] = cross_image(im1, im, boxsize = 1000)
    image_stack[:, :, index] = shift_image(im, xshifts_filt[index], yshifts_filt[index])   

    # Save the final stacked images into your new folder
    fits.writeto(dir + targname + '_' + filter + ' aligned.fits', image_stack[:, :, index], overwrite = True)
    print('   Wrote FITS file ', targname + '_' + filter + 'aligned.fits', 'in ', dir, '\n')
    print('\n Done stacking!')
    
  return
