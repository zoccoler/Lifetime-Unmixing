import napari
import tifffile
import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle
import dask.array as da
from pathlib import Path
from skimage.measure import label, regionprops_table

from napari_flim_phasor_plotter.filters import apply_binning
from napari_flim_phasor_plotter._reader import read_stack
from napari_flim_phasor_plotter._widget import make_flim_phasor_plot, manual_label_extract, smooth_cluster_mask

from utilities import ellipse_vertices, format_metadata, set_plot_zoom_position, add_segmentation_metadata
import time
import warnings
from watermark import watermark
warnings.filterwarnings("ignore")
# Print package versions and machine info
print(watermark(packages="numpy,pandas,napari,napari_flim_phasor_plotter,skimage,dask,pyclesperanto_prototype,tifffile,xmltodict", machine=True, python=True, gpu=True))

start_time = time.time()
# Inputs

# Main folder path (each folder contains a zarr file and an xml file)
main_folder_path = Path("/home/pol_haase/mazo260d/Data/I227_Lifetime_Unmixing_of_Dyes_with_Overlapping_Sprectra/Batch_Processing")

# Additional metadata in case the xml file is not available
z_pixel_size = 0.5*1e-6 # in m
pixel_size_unit = 'm'
time_resolution_per_slice = 0.663 # in seconds
time_unit = 's'
channel_names = ['Centrosomes', 'Membrane and Chromatin']

# Center, major, minor, and angle of the ellipse
center = (0.880, 0.300)
major = 0.055
minor = 0.033
angle = 2

# Number of bins in 2D histogram
bins = 400

# Axis limits for 2D histogram
x_lim = (-0.05, 1.05)
y_lim = (0.0, 0.55)

# Manual threshold value for phasor plot
threshold = 15

# Median filter iteration number
median_filter_iteration = 3

# Harmonic number
harmonic = 1

# Binning settings
binning = True
binning_kernel_size = 3
binning_3D = False

# minimal area in pixels of holes to be filled
fill_area_px = 64
# radius for morphological closing and opening (objects smaller than this will be removed, gaps smaller than this will be connected)
smooth_radius = 1
# minimal object size
min_allowed_object_size = 900 # total number of pixels
# store phasor mask
store_phasor_mask = False

processed_samples = []

# Iterate throgh all the folders in the main folder
for folder in main_folder_path.iterdir():
    new_sample_time = time.time()
    if folder.is_dir():

        print(f"Processing folder: {folder}")
        sample_name = folder.stem
        xml_path = None
        # Create output folders
        output_path = Path(folder) / 'Outputs'
        output_path.mkdir(exist_ok=True)
        segmentation_output_path = output_path / 'Segmentation'
        segmentation_output_path.mkdir(exist_ok=True)
        phasor_plots_output_path = output_path / 'Phasor Plots'
        phasor_plots_output_path.mkdir(exist_ok=True)
        screenshots_output_path = output_path / 'Screenshots'
        screenshots_output_path.mkdir(exist_ok=True)
        omero_output_path = output_path / 'OME-TIFFs'
        omero_output_path.mkdir(exist_ok=True)
        for file in folder.iterdir():
            # Read the zarr and a single ptu file
            if file.suffix == '.zarr':
                zarr_data_path = file
            if file.suffix == '.xml':
                xml_path = file
        
        # Read the zarr file
        data, flim_metadata = read_stack(zarr_data_path)
        summed_intensity_stack = da.sum(data, axis=1).astype(np.uint32)
        print("Extracting metadata...")
        if xml_path is None:
            print("No xml file found. Using additional metadata.")
        else:
            if not xml_path.exists():
                print("The xml file does not exist. Using additional metadata.")
            else:
                print(f"Reading metadata from: {xml_path} and zarr file.")
        # Extract metadata
        metadata_timelapse, metadata_single_timepoint = format_metadata(
            flim_metadata,
            xml_path, 
            data.shape,
            z_pixel_size,
            pixel_size_unit,
            time_resolution_per_slice,
            time_unit,
            channel_names)

        laser_frequency = flim_metadata[0]['frequency']/1e6 # in MHz
        print(f'Laser frequency: {laser_frequency} MHz')
        # Split channels
        image_raw_FLIM_channel_0 = data[0] # first channel
        image_raw_FLIM_channel_1 = data[1] # second channel

        master_table = pd.DataFrame()
        cluster_labels_image_timelapse_post_processed = np.zeros((
            image_raw_FLIM_channel_1.shape[1],
            image_raw_FLIM_channel_1.shape[2],
            image_raw_FLIM_channel_1.shape[3],
            image_raw_FLIM_channel_1.shape[4]), dtype=np.uint8)
        cluster_mask_timelapse = np.zeros((
            image_raw_FLIM_channel_1.shape[1],
            image_raw_FLIM_channel_1.shape[2],
            image_raw_FLIM_channel_1.shape[3],
            image_raw_FLIM_channel_1.shape[4]), dtype=np.uint8)
        

        for t in range(image_raw_FLIM_channel_0.shape[1]):
            print(f"Processing timepoint: {t}")
            print(f"Saving OME-TIF...")
            t_string = str(t).zfill(len(str(data.shape[2])))
            output_file_name =  sample_name + f'_t{t_string}.ome.tif'
            with tifffile.TiffWriter(omero_output_path / output_file_name, ome=True) as tif:
                tif.write(data[:,:,t], metadata=metadata_single_timepoint, compression='zlib')
            # Get current timepoint
            image_raw_FLIM_channel_0_current_t = image_raw_FLIM_channel_0[:, t, ...]
            image_raw_FLIM_channel_1_current_t = image_raw_FLIM_channel_1[:, t, ...]

            # Reshape to make it compatible with plugin standards
            image_raw_FLIM_channel_0_current_t = image_raw_FLIM_channel_0_current_t[:, np.newaxis, ...]
            image_raw_FLIM_channel_1_current_t = image_raw_FLIM_channel_1_current_t[:, np.newaxis, ...]

            # Convert to numpy array
            # Use channle 1 for embryos data, use channel 0 for worms data
            image_raw_FLIM_channel_1_current_t = np.asarray(image_raw_FLIM_channel_1_current_t)
            # Apply binning (if True)
            if binning:
                image_raw_FLIM_channel_1_current_t = apply_binning(image_raw_FLIM_channel_1_current_t,
                                                                    bin_size=binning_kernel_size,
                                                                    binning_3D=binning_3D)

            # Generate intensity image from raw FLIM image
            image_intensity_channel_0 = np.sum(image_raw_FLIM_channel_0_current_t, axis=0)
            image_intensity_channel_1 = np.sum(image_raw_FLIM_channel_1_current_t, axis=0)

            print(f"{sample_name} max intensity: {image_intensity_channel_1.max()}")
            print(f"{sample_name} average intensity: {image_intensity_channel_1.mean()}")

            # Add images to napari viewer
            viewer = napari.Viewer()
            viewer.add_image(image_raw_FLIM_channel_1_current_t, name=(sample_name + ' raw channel 1'), blending='additive', colormap='magenta')
            viewer.add_image(image_intensity_channel_0, name=(sample_name + ' intensity channel 0'), colormap='green')
            viewer.add_image(image_intensity_channel_1, name=(sample_name + ' intensity channel 1'), colormap='magenta', blending='additive')
            
            # Adjust viewer visualization (3D display, first microtime)
            viewer.dims.ndisplay = 3
            viewer.dims.current_step = (0, 0, 21, 128, 128)

            print('Napari viewer initialized')

            plot_maker_widget = make_flim_phasor_plot()
            # Generate phasor plot
            phasor_plot_widget, labels_layer = plot_maker_widget(image_layer = viewer.layers[(sample_name + ' raw channel 1')],
                                    laser_frequency = laser_frequency,
                                    harmonic = harmonic,
                                    threshold = threshold,
                                    apply_median = True,
                                    median_n = median_filter_iteration,
                                    napari_viewer = viewer)
            
            print('Phasor plot generated')
            
            # Optimize phasor visualization and phasor plot position
            phasor_plot_widget.bin_auto.setChecked(False)
            phasor_plot_widget.bin_number_spinner.setValue(bins)
            phasor_plot_widget.bin_number_set.clicked.emit(True)
            set_plot_zoom_position(phasor_plot_widget, x_lim, y_lim)

            # Save screenshot
            screenshot = viewer.screenshot(canvas_only=False)
            tifffile.imwrite(screenshots_output_path / f"{sample_name}_t{t}_screenshot.png", screenshot)
            # Save phasor plot before clustering
            phasor_plot_widget.graphics_widget.axes.figure.savefig(phasor_plots_output_path / f'{sample_name}_t{t}_phasor_plot.png', dpi=300)

            # Get labels layer with labelled pixels (labels)
            for choice in phasor_plot_widget.layer_select.choices:
                if choice.name.startswith('Labelled_pixels_from_'):
                    viewer.layers.selection.active = choice
                    # phasor_plot_widget.layer_select.value = choice
                    break
            
            # Set center, major axis, minor axis length and angle of ellipse
            vertices = ellipse_vertices(center, major, minor, angle)
            # Select vertices
            phasor_plot_widget.graphics_widget.selector.onselect(vertices)

            print('Cluster selected from ellipse!')

            # Draw ellipse on phasor plot
            x_values = [x[0] for x in vertices]
            y_values = [x[1] for x in vertices]
            phasor_plot_widget.graphics_widget.axes.plot(x_values, y_values, 'w-', linewidth=1)
            set_plot_zoom_position(phasor_plot_widget, x_lim, y_lim)
            # Save phasor plot after cluster selection
            phasor_plot_widget.graphics_widget.axes.figure.savefig(phasor_plots_output_path / f'{sample_name}_t{t}_phasor_plot_after_cluster_selection.png', dpi=300)
            # Save phasor data as a table
            labels_layer.features.to_csv(phasor_plots_output_path / f'{sample_name}_t{t}_phasors_table.csv', index=False)
            
            # Extract cluster mask from phasor plot
            extraced_labels_layer = manual_label_extract(
                cluster_labels_layer=phasor_plot_widget.visualized_layer, 
                label_number=2)
            cluster_mask_image = extraced_labels_layer.data

            # Save extracted mask
            cluster_mask_timelapse[t,] = np.squeeze(cluster_mask_image)

            # Smooth cluster mask
            cluster_mask_layer_smoothed = smooth_cluster_mask(
                cluster_mask_layer = extraced_labels_layer,
                fill_area_px = fill_area_px,
                smooth_radius = smooth_radius)
            
            # Instance Segmentation
            cluster_labels_image = label(cluster_mask_layer_smoothed.data)

            # Exclude small labels
            cluster_labels_image_filtered = cle.exclude_small_labels(cluster_labels_image, maximum_size=min_allowed_object_size)
            viewer.add_labels(cluster_labels_image_filtered, name=(sample_name + ' cluster labels'), opacity=0.6)
            # Save screenshot
            screenshot = viewer.screenshot(canvas_only=False)
            tifffile.imwrite(screenshots_output_path / f"{sample_name}_t{t}_post_segmentation_screenshot.png", screenshot)

            cluster_labels_image_timelapse_post_processed[t,] = np.squeeze(cluster_labels_image_filtered)

            print('Post-processing done! Extracting Features...')

            # Feature extraction
            table = pd.DataFrame(regionprops_table(cluster_labels_image_filtered, properties=('label', 'area', 'centroid')))
            table['frame'] = t
            table['sample'] = sample_name
            master_table = pd.concat([master_table, table])

            print('Found {} objects'.format(len(table)))

            # Close napari viewer
            viewer.close()
            print('Finished timepoint!')

        # save segmentation results with tifffile
        tifffile.imsave(segmentation_output_path / f'{sample_name}_cluster_mask_timelapse.tiff', 
                        cluster_mask_timelapse, imagej=True)
        tifffile.imsave(segmentation_output_path / f'{sample_name}_cluster_labels_timelapse_post_processed.tiff', 
                        cluster_labels_image_timelapse_post_processed, imagej=True)
        
        print('Saving timelapse OME-TIFF with segmentation results...')
        # Save summed intensity timelapse stack with segmentation result as OME-TIFF
        # Add a new axis to the segmentation result image
        cluster_labels_image_timelapse_post_processed = np.expand_dims(cluster_labels_image_timelapse_post_processed.astype(summed_intensity_stack.dtype), axis=0)
        # Convert the segmentation result to a Dask array
        chunk_shape = summed_intensity_stack.chunksize
        cluster_labels_image_timelapse_dask = da.from_array(cluster_labels_image_timelapse_post_processed, chunks=chunk_shape)
        
        if store_phasor_mask:
            cluster_mask_timelapse = np.expand_dims(cluster_mask_timelapse.astype(summed_intensity_stack.dtype), axis=0)
            cluster_mask_timelapse_dask = da.from_array(cluster_mask_timelapse, chunks=chunk_shape)
            metadata_timelapse = add_segmentation_metadata(metadata_timelapse, channel_name='Phasor Mask')
            # Add mask and segmentation result to the summed intensity stack
            summed_intensity_stack = da.concatenate([summed_intensity_stack, cluster_mask_timelapse_dask, cluster_labels_image_timelapse_dask], axis=0)
        else:
            # Add segmentation result to the summed intensity stack
            summed_intensity_stack = da.concatenate([summed_intensity_stack, cluster_labels_image_timelapse_dask], axis=0)
        # Add metadata to 'segmentation' channel(s): Phasor Mask and Segmentation results after post-processing
        metadata_timelapse = add_segmentation_metadata(metadata_timelapse)
        output_file_name = sample_name + '_summed_intensity_with_segmentation.ome.tif'
        with tifffile.TiffWriter(omero_output_path / output_file_name, ome=True) as tif:
            tif.write(summed_intensity_stack[:], metadata=metadata_timelapse, compression='zlib')

        # Save table with measurements
        master_table.to_csv(output_path / f'{sample_name}_master_table.csv', index=False)

        # File path for the config file
        file_path = output_path / f'{sample_name}_config.txt'

        # Writing to the file
        with open(file_path, 'w') as file:
            file.write(f"sample_name = {sample_name}\n")
            file.write(f"center = {center}\n")
            file.write(f"major-axis = {major}\n")
            file.write(f"minor-axis = {minor}\n")
            file.write(f"angle = {angle}\n")
            file.write(f"bins = {bins}\n")
            file.write(f"x_lim = {x_lim}\n")
            file.write(f"y_lim = {y_lim}\n")
            file.write(f"threshold = {threshold}\n")
            file.write(f"median_filter_iteration = {median_filter_iteration}\n")
            file.write(f"harmonic = {harmonic}\n")
            file.write(f"binning = {binning}\n")
            file.write(f"binning_kernel_size = {binning_kernel_size}\n")
            file.write(f"binning_3D = {binning_3D}\n")
            file.write(f"fill_area_px = {fill_area_px}\n")
            file.write(f"smooth_radius = {smooth_radius}\n")
            file.write(f"min_allowed_object_size = {min_allowed_object_size}\n")
        
    print(f"Finished processing sample: {sample_name}!")
    print(f"Time taken for processing sample: %.2f seconds" % (time.time() - new_sample_time))
    processed_samples.append(sample_name)
    print(f"Processed samples: {processed_samples}\n")

print(f"Total time taken for processing all samples: %.2f seconds" % (time.time() - start_time))
        