# SmartAUV Summer Project 2024


### Description of the Smart AUV Project (main initiative) and this Summer Sub-Project

This repository is part of the SmartAUVs summer research project under the UiO:Energy and Environment initiative for 2024. The project is a collaborative effort involving the University of Oslo (UiO), Norwegian Geotechnical Institute, Plymouth Marine Laboratory (PML), and other partners. The SmartAUVs project aims to enhance the monitoring capabilities of Autonomous Underwater Vehicles (AUVs) by incorporating artificial intelligence (AI) and specialized signal processing techniques to enable real-time decision-making based on sensor input.

The primary goal of SmartAUVs is to develop tools that offshore operators and regulators can use to monitor oceanic and benthic environments near industrial activities such as oil and gas extraction and CO2 storage. The project involves simulating gas emission scenarios in a representative ocean environment at high spatial and temporal resolution. These simulations, provided by PML, will help understand plume characteristics and development, which are crucial for determining optimal AUV behavior.

The high-resolution oceanographic simulations will cover approximately 250x250 meters, with a resolution down to one meter for gas concentration and centimeters for bubbles, and will include a simulated period of around 12 hours to capture two tidal cycles. The simulations will provide detailed insights into the horizontal and vertical displacement of plumes, including dilution rates, bubble rise heights, dissolution, and localized concentration distributions of dissolved gases.

As part of the summer project, I will develop a Python utility library for handling and visualizing the spatiotemporal output from these simulations. The library includes functionalities for:

- Extracting and interpolating bubble data and dissolved gas concentrations and other oceanographic parameters.
- Filtering and analyzing bubble sizes and distributions.
- Visualizing gas concentrations and bubble distributions in 2D, 3D, and 4D.

The developed library will play a crucial role in the SmartAUVs project, acting as an interface between oceanographic simulations and algorithms for gas leakage detection and autonomous AUV behavior.

For more information on the UiO:Energy and Environment summer research projects, visit: [UiO Summer Research Projects 2024](https://www.uio.no/english/research/strategic-research-areas/uio-energy-and-environment/funding/summer-research-projects-2024/Summerprojects.html)

For more details on the SmartAUVs project, visit: [SmartAUVs Project Description](https://prosjektbanken.forskningsradet.no/project/FORISS/333872)


## Repository Contents

- `bubble_utils.py`: Scripts for handling and processing bubble data.
- `chem_utils.py`: Scripts for handling and processing chemical data.
- `generate_colormap.py`: Script for generating custom colormaps for visualizations.
- `transform_to_paraview.py`: Script for transforming data to VTK format for Paraview visualization.

## Descriptions of Scripts

### bubble_utils.py

This script is designed for processing and visualizing bubble data in a marine environment. It includes functions for loading bubble data from a file, computing tetrahedral matrices to check if a point is within a tetrahedron or prism, and plotting the results in a 3D space.

- #### Functions

    - **`load_bubble_dataset(bubble_data_path)`**: Loads bubble data from a specified file path and returns a DataFrame containing the processed bubble data.
    - **`compute_tetrahedron_matrix(vertex1, vertex2, vertex3, vertex4)`**: Computes the transformation matrix for a tetrahedron.
    - **`is_point_in_tetrahedron(vertex1, vertex2, vertex3, vertex4, point)`**: Determines if a point is inside a tetrahedron.
    - **`is_point_in_prism(point, prism_vertices)`**: Checks if a point is inside any of the tetrahedra forming a trapezoidal prism.
    - **`compute_trapezoidal_prisms(origin, direction_theta_deg, direction_phi_deg, line_length, num_planes, prism_angle_width_deg, prism_angle_height_deg)`**: Computes vertices of multiple trapezoidal prisms along a specified direction.
    - **`plot_bubbles_and_prisms(prisms_array, inside_bubble_points, outside_bubble_points, origin, bubble_counts, all_bubble_points)`**: Plots trapezoidal prisms and bubbles, with different colors for bubbles inside and outside the prisms.
    - **`get_bubble_counts_for_prisms(dataset, prisms_array, specific_time)`**: Calculates bubble counts for given prisms at a specific time and classifies points.


### chem_utils.py

This script is designed for processing and analyzing chemical data from a NetCDF file. It includes functions for loading chemical datasets, extracting chemical data within a specified volume, and computing average values.

- ### Functions

    - **`load_chemical_dataset(chem_data_path)`**: Loads a chemical dataset from a NetCDF file, fixes the time issue, and returns the corrected dataset.
    - **`extract_chemical_data_for_volume(dataset, metadata, data_variable)`**: Extracts chemical data within a specified spherical volume and computes the average data value.

### generate_colormap.py

This script provides utility functions for generating and displaying custom colormaps for visualizations. It includes functions for converting hex colors to RGB, creating continuous colormaps, and plotting color tables.

- #### Functions

    - **`hex_to_rgb(hex_value)`**: Converts a hex color value to RGB.
    - **`rgb_to_dec(rgb_value)`**: Converts RGB values to decimal.
    - **`get_continuous_cmap(hex_list, float_list=None, reverse=False)`**: Creates and returns a continuous color map that can be used in heat map figures.
    - **`plot_color_table(hex_list)`**: Displays a color table with the provided hex colors, showing the HEX and RGB values.
    - **`example_colormap()`**: Creates an example plot using both the default and a custom color map.


### transform_to_paraview.py

This script transforms chemical data from a NetCDF file to VTK format suitable for volume rendering. It reads the NetCDF file, interpolates the data onto a regular grid, and saves it in VTK format for each time step.

- #### Functions

    - **`transform_netCDF_to_vtk(data_path, data_variable)`**: Transforms chemical data from a NetCDF file to VTK format suitable for volume rendering.


## Usage

1. **Clone the repository:**
   ```sh
   git clone https://github.com/JouvalSomer/SmartAUV_summer_project_2024.git
   cd SmartAUV_summer_project_2024
   ```
2. **Install Required Dependencies:**
   ```sh
    pip install -r requirements.txt
   ```
3. **Run Scarips You Wish:grinning:**

## References

- The `get_continuous_cmap` function is adapted from the following GitHub repository: [KerryHalupka/custom_colormap](https://github.com/KerryHalupka/custom_colormap)
- The functions `compute_tetrahedron_matrix`, `is_point_in_tetrahedron`, and `is_point_in_prism` are adapted and modified from a solution by Dorian on Stack Overflow: [How to check whether the point is in the tetrahedron or not](https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not)

## Supervisors

- **Kai Olav Ellefsen**, Associate Professor, IFI/UiO.
- **Ivar-Kristian Waarum**, Senior Engineer at Norwegian Geotechnical Institute and PhD student at UiO.
