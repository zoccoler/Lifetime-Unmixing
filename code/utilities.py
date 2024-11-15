def set_plot_zoom_position(widget, xlim, ylim):
    """Set axes limits of the plot widget to the given values."""
    widget.graphics_widget.axes.set_xlim(xlim[0], xlim[1])
    widget.graphics_widget.axes.set_ylim(ylim[0], ylim[1])
    widget.graphics_widget.axes.figure.canvas.draw()

def circle_vertices(center, radius, num_vertices=100):
    """Generates vertices of a circle given its center, radius, and the desired number of vertices.

    Parameters
    ----------
    center : Tuple[float, float]
        The center of the circle.
    radius : float
        The radius of the circle.
    num_vertices : int, optional
        The number of vertices to generate, by default 100

    Returns
    -------
    List[Tuple[float, float]]
        A list of tuples representing the vertices of the circle.
    """
    import math

    vertices = []
    for i in range(num_vertices):
        angle = 2 * math.pi * i / num_vertices
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))

    return vertices

def ellipse_vertices(center, a, b, angle, num_vertices=100):
    """Generates vertices of a ellipse given its center, major radius (a), minor radius (b),
    rotation angle, and the desired number of vertices.

    Parameters
    ----------
    center : Tuple[float, float]
        The center of the ellipse.
    a : float
        The major radius of the ellipse (radius along the x-axis).
    b : float
        The minor radius of the ellipse (radius along the y-axis).
    angle : float
        The rotation angle of the ellipse in radians.
    num_vertices : int, optional
        The number of vertices to generate, by default 100

    Returns
    -------
    List[Tuple[float, float]]
        A list of tuples representing the vertices of the ellipse.
    """
    import math

    vertices = []
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    
    for i in range(num_vertices):
        # Angle for the ellipse calculation (not the rotation)
        ellipse_angle = 2 * math.pi * i / num_vertices
        # Ellipse vertex before rotation
        x_ellipse = a * math.cos(ellipse_angle)
        y_ellipse = b * math.sin(ellipse_angle)
        # Apply rotation
        x_rotated = center[0] + x_ellipse * cos_theta - y_ellipse * sin_theta
        y_rotated = center[1] + x_ellipse * sin_theta + y_ellipse * cos_theta
        vertices.append((x_rotated, y_rotated))

    return vertices

def format_metadata(flim_metadata, xml_path=None, stack_shape=None, z_pixel_size = 0.5, pixel_size_unit = 'µm', time_resolution_per_slice = 0.663, time_unit = 's', channel_names = ['0', '1'], axes='CTZYX', timelapse=True):
    import xmltodict
    if xml_path is None:
        xml_exist = False
    else:
        xml_exist = xml_path.exists()
    if xml_exist:
        # Read and parse the XML file
        with open(xml_path, 'r') as file:
            xml_data = file.read()
        metadata_dict = xmltodict.parse(xml_data)

        # Extract the metadata
        image_desc = metadata_dict['Data']['Image']['ImageDescription']
        dimensions = {dim['@DimID']: dim for dim in image_desc['Dimensions']['DimensionDescription']}
        detectors =  metadata_dict['Data']['Image']['Attachment'][4]['ATLConfocalSettingDefinition']['DetectorList']['Detector']
        spectro = metadata_dict['Data']['Image']['Attachment'][4]['ATLConfocalSettingDefinition']['Spectro']['MultiBand']

        # Extract additional metadata fields
        acquisition_time = image_desc.get('StartTime', '')
        channel_metadata = {
            'Name': [],
            'EmissionWavelength': [],
            'EmissionWavelengthUnit':  [],
            'ExcitationWavelength': [],
            'ExcitationWavelengthUnit': []
        }

        # Extract channel metadata based on active detectors
        for detector in detectors:
            if detector['@IsActive'] == '1':
                channel_metadata['Name'].append(detector['@Name'])
                channel_metadata['ExcitationWavelength'].append(float(detector['DetectionReferenceLine']['@LaserWavelength']))
                channel_metadata['ExcitationWavelengthUnit'].append('nm')
                # Find the corresponding filter for the emission wavelength
                for band in spectro:
                    if band['@ChannelName'] == detector['@ChannelName']:
                        target_wavelength_begin = float(band['@TargetWaveLengthBegin'])
                        target_wavelength_end = float(band['@TargetWaveLengthEnd'])
                        emission_wavelength = (target_wavelength_begin + target_wavelength_end) / 2
                        channel_metadata['EmissionWavelength'].append(emission_wavelength)
                        channel_metadata['EmissionWavelengthUnit'].append('nm')
                        break

        # Prepare metadata for OME-TIFF
        metadata_timelapse = dict()
        metadata_timelapse['axes'] = axes
        metadata_timelapse['PhysicalSizeX'] = float(dimensions['X']['@Voxel'])
        metadata_timelapse['PhysicalSizeXUnit'] = dimensions['X']['@Unit']
        metadata_timelapse['PhysicalSizeY'] = float(dimensions['Y']['@Voxel'])
        metadata_timelapse['PhysicalSizeYUnit'] = dimensions['Y']['@Unit']
        metadata_timelapse['PhysicalSizeZ'] = float(dimensions['Z']['@Voxel'])
        metadata_timelapse['PhysicalSizeZUnit'] = dimensions['Z']['@Unit']
        if timelapse:
            metadata_timelapse['TimeIncrement'] = float(dimensions['T']['@Voxel'].split()[0])
            metadata_timelapse['TimeIncrementUnit'] = dimensions['T']['@Voxel'].split()[1] if len(dimensions['T']['@Voxel'].split()) > 1 else 's'
        metadata_timelapse['AcquisitionDate'] = acquisition_time
        if 'C' in axes:
            metadata_timelapse['Channel'] = channel_metadata

    else:
        if stack_shape is None:
            raise ValueError('stack_shape must be provided if no XML file is provided')
        # If no XML file is provided, use the metadata from the Zarr file along with some manual inputs
        # The time resolution must be calculated depending on the number of z-slices
        time_resolution = time_resolution_per_slice * stack_shape[-3] 
        metadata_timelapse = dict()
        metadata_timelapse['axes'] = axes
        metadata_timelapse['PhysicalSizeX'] = flim_metadata[0]['x_pixel_size']
        metadata_timelapse['PhysicalSizeXUnit'] = pixel_size_unit
        metadata_timelapse['PhysicalSizeY'] = flim_metadata[0]['y_pixel_size']
        metadata_timelapse['PhysicalSizeYUnit'] = pixel_size_unit
        metadata_timelapse['PhysicalSizeZ'] = z_pixel_size
        metadata_timelapse['PhysicalSizeZUnit'] = pixel_size_unit
        if timelapse:
            metadata_timelapse['TimeIncrement'] = time_resolution
            metadata_timelapse['TimeIncrementUnit'] = time_unit
        metadata_timelapse['AcquisitionDate'] = acquisition_time
        if 'C' in axes:
            metadata_timelapse['Channel'] = dict()
            metadata_timelapse['Channel']['Name'] = channel_names
        

    metadata_single_timepoint = metadata_timelapse.copy()
    metadata_single_timepoint['TimeIncrement'] = flim_metadata[0]['tcspc_resolution'] * 1e12
    metadata_single_timepoint['TimeIncrementUnit'] = 'ps'

    return metadata_timelapse, metadata_single_timepoint

def add_segmentation_metadata(metadata_timelapse, channel_name='Segmentation'):
    for channel_key, channel_values in metadata_timelapse['Channel'].items():
        if channel_key == 'Name':
            metadata_timelapse['Channel'][channel_key].append(channel_name)
        elif channel_key == 'ExcitationWavelengthUnit' or channel_key == 'EmissionWavelengthUnit':
            metadata_timelapse['Channel'][channel_key].append('nm')
        elif channel_key == 'ExcitationWavelength' or channel_key == 'EmissionWavelength':
            metadata_timelapse['Channel'][channel_key].append(1)
        else:
            metadata_timelapse['Channel'][channel_key].append('')
    return metadata_timelapse