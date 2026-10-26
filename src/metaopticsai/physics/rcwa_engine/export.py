"""
Export module for metabox.

This module contains functions to generate GDSII and other visualization files.
Uses gdsfactory 9.28.1+ for GDS operations.

Compatible with Python 3.11.9+
"""
from __future__ import annotations
import uuid
import os
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
import sys

from . import assembly 
from . import expansion

if TYPE_CHECKING:
    from . import rcwa

# Optional import for gdsfactory
try:
    import gdsfactory as gf
    HAS_GDSFACTORY = True
    try:
        gf.gpdk.PDK.activate()
    except Exception:
        pass
except ImportError:
    HAS_GDSFACTORY = False


def unit_cell_to_gds_shape(
    cell: "rcwa.UnitCell",
    layer: int = 0,
    gds_layer: tuple[int, int] = (1, 0),
) -> gf.Component:
    """
    Converts a unit cell to a gdsfactory Component.
    
    Args:
        cell: The unit cell to convert
        layer: Index of the layer within the unit cell
        gds_layer: Tuple of (layer, datatype) for GDS export
        
    Returns:
        A gdsfactory Component containing the shapes
    """
    if not HAS_GDSFACTORY:
        raise ImportError(
            "gdsfactory is required for GDS export. Install with: pip install gdsfactory>=9.28.1"
        )
    
    component = gf.Component()
    
    for shape in cell.layers[layer].shapes:
        vertices = []
        for vertex in shape.get_vertices():
            vx, vy = vertex[0], vertex[1]
            if hasattr(vx, "numpy"): vx = float(vx.numpy())
            else: vx = float(vx)
            if hasattr(vy, "numpy"): vy = float(vy.numpy())
            else: vy = float(vy)
            vertices.append((vx * 1e6, vy * 1e6))
        
        # Create polygon from vertices
        polygon = gf.kdb.DPolygon([gf.kdb.DPoint(x, y) for x, y in vertices])
        component.add_polygon(polygon, layer=gds_layer)
    
    # Apply 4-fold symmetry if required
    if cell.layers[layer].enforce_4fold_symmetry:
        component = gds_shape_force_4fold_symmetry(component, gds_layer)
    
    return component


def gds_shape_force_4fold_symmetry(
    component: gf.Component,
    gds_layer: tuple[int, int] = (1, 0),
) -> gf.Component:
    """
    Generates a 4-fold symmetric shape from a component.
    
    Args:
        component: The input component
        gds_layer: Tuple of (layer, datatype) for operations
        
    Returns:
        A new component with 4-fold symmetry
    """
    if not HAS_GDSFACTORY:
        raise ImportError("gdsfactory is required")
    
    # Create rotated copies and combine using boolean OR
    result = gf.Component()
    
    # Add original
    ref1 = result.add_ref(component)
    
    # Add 90° rotation
    rotated_90 = gf.Component()
    ref_90 = rotated_90.add_ref(component)
    ref_90.drotate(90)
    
    # Add 180° rotation  
    rotated_180 = gf.Component()
    ref_180 = rotated_180.add_ref(component)
    ref_180.drotate(180)
    
    # Add 270° rotation
    rotated_270 = gf.Component()
    ref_270 = rotated_270.add_ref(component)
    ref_270.drotate(270)
    
    # Boolean OR all rotations
    combined = gf.boolean(
        component, rotated_90, operation="or", layer=gds_layer
    )
    if combined is not None:
        combined = gf.boolean(
            combined, rotated_180, operation="or", layer=gds_layer
        )
    if combined is not None:
        combined = gf.boolean(
            combined, rotated_270, operation="or", layer=gds_layer
        )
    
    if combined is None:
        return component
    
    # Mirror for full symmetry
    mirrored = gf.Component()
    ref_mirror = mirrored.add_ref(combined)
    ref_mirror.dmirror_y()
    
    final = gf.boolean(combined, mirrored, operation="or", layer=gds_layer)
    
    return final if final is not None else combined


def get_loc_along_circle(
    radius_in_meters: float,
    radius_size: int,
    query_radius: int,
    period: float,
    r2c_basis: tf.Tensor,
) -> np.ndarray:
    """
    Gets locations along a circle for placing unit cells.
    
    Args:
        radius_in_meters: Radius of the metasurface in meters
        radius_size: Number of radial pixels
        query_radius: Index of the radius to query
        period: Unit cell periodicity in microns
        r2c_basis: Radius to circle expansion basis tensor
        
    Returns:
        Array of (x, y) coordinates in microns
    """
    radius_selection = np.zeros(radius_size)
    radius_selection[query_radius] = 1
    tf_radius = tf.reshape(radius_selection, [-1, radius_size])
    circle = tf.sparse.sparse_dense_matmul(tf_radius, r2c_basis)
    circle = tf.reshape(circle, [radius_size * 2, radius_size * 2])
    location_index_arr = np.argwhere(circle == 1)
    hp = period / 2
    radius_in_micron = radius_in_meters * 1e6
    x_coor = np.linspace(-radius_in_micron + hp, radius_in_micron - hp, radius_size * 2)
    y_coor = np.linspace(-radius_in_micron + hp, radius_in_micron - hp, radius_size * 2)
    x_locations = x_coor.take(location_index_arr[:, 0])
    y_locations = y_coor.take(location_index_arr[:, 1])
    return np.stack([x_locations, y_locations])


def generate_noncircular_gds(
    metasurface: "assembly.Metasurface",
    layer: int,
    export_name: str,
    export_directory: str | None = None,
    inverted: bool = False,
    gds_layer: tuple[int, int] = (1, 0),
) -> None:
    """
    Generates a GDSII file for a noncircular (rectangular grid) metasurface.
    
    Args:
        metasurface: The metasurface object to export
        layer: Index of the layer to export
        export_name: Name for the output GDS file (without extension)
        export_directory: Directory to save the file (default: ./gds_export/)
        inverted: If True, invert the pattern (subtract from unit cell)
        gds_layer: Tuple of (layer, datatype) for GDS export
    """
    if not HAS_GDSFACTORY:
        raise ImportError(
            "gdsfactory is required for GDS export. Install with: pip install gdsfactory>=9.28.1"
        )
    
    if export_directory is None:
        export_directory = "./gds_export/"

    # Create main device component
    device = gf.Component(name=f"metasurface_{uuid.uuid4().hex[:8]}")

    periodicity = metasurface.atom_2d.period * 1e6  # Convert to microns
    atom_positions = metasurface.get_atom_positions().reshape([-1, 2])
    
    if metasurface.atom_2d.use_mmodel:
        cell_array = metasurface.atom_2d.mmodel.protocell.generate_cells_from_parameter_tensor(
            metasurface.atom_2d.tensor
        )
    else:
        cell_array = metasurface.atom_2d.proto_unit_cell.generate_cells_from_parameter_tensor(
            metasurface.atom_2d.tensor
        )
    
    for name_index, (this_cell, xy_position) in enumerate(zip(cell_array, atom_positions)):
        polygon_component = unit_cell_to_gds_shape(this_cell, layer=layer, gds_layer=gds_layer)
        
        if inverted:
            box_hw = periodicity / 2.0
            box = gf.components.rectangle(
                size=(periodicity, periodicity),
                layer=gds_layer,
                centered=True,
            )
            polygon_component = gf.boolean(
                box, polygon_component, operation="not", layer=gds_layer
            )
        
        if polygon_component is None:
            continue
            
        # Create a named cell for this unit
        cell = gf.Component(name=f"cell_{name_index}_{uuid.uuid4().hex[:8]}")
        cell.add_ref(polygon_component)
        
        # Add reference at the correct position
        loc_x, loc_y = xy_position * 1e6  # Convert to microns
        ref = device.add_ref(cell)
        ref.dmove((loc_x, -loc_y))

    # Ensure export directory exists
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
    
    # Write GDS file
    device.write_gds(f"{export_directory}/{export_name}.gds")


def generate_gds(
    metasurface: "assembly.Metasurface",
    layer: int,
    export_name: str,
    export_directory: str | None = None,
    inverted: bool = False,
    gds_layer: tuple[int, int] = (1, 0),
) -> None:
    """
    Generates a GDSII file for a metasurface.
    
    This function handles both circular (radially symmetric) and rectangular
    grid metasurface layouts.
    
    Args:
        metasurface: The metasurface object to export
        layer: Index of the layer to export
        export_name: Name for the output GDS file (without extension)
        export_directory: Directory to save the file (default: ./gds_export/)
        inverted: If True, invert the pattern (subtract from unit cell)
        gds_layer: Tuple of (layer, datatype) for GDS export
        
    Example:
        >>> from metabox import assembly, export
        >>> # Create your metasurface...
        >>> export.generate_gds(
        ...     metasurface=my_metalens,
        ...     layer=0,
        ...     export_name="my_metalens_design",
        ...     export_directory="./output/",
        ... )
    """
    if not HAS_GDSFACTORY:
        raise ImportError(
            "gdsfactory is required for GDS export. Install with: pip install gdsfactory>=9.28.1"
        )
    
    # Use non-circular export for rectangular grids
    if not metasurface.use_circular_expansions:
        generate_noncircular_gds(
            metasurface=metasurface,
            layer=layer,
            export_name=export_name,
            export_directory=export_directory,
            inverted=inverted,
            gds_layer=gds_layer,
        )
        return

    if export_directory is None:
        export_directory = "./gds_export/"

    # Create main device component
    device = gf.Component(name=f"metasurface_{uuid.uuid4().hex[:8]}")

    periodicity = metasurface.atom_1d.period * 1e6  # Convert to microns
    n_pixels_radial = metasurface.n_pixels_radial
    radius_in_meters = metasurface.diameter / 2.0
    r2c_basis = expansion.radius_to_circle_basis(n_pixels_radial)
    r2c_basis = tf.cast(r2c_basis, tf.float64)
    
    if metasurface.atom_1d.use_mmodel:
        cell_array = metasurface.atom_1d.mmodel.protocell.generate_cells_from_parameter_tensor(
            metasurface.atom_1d.tensor
        )
    else:
        cell_array = metasurface.atom_1d.proto_unit_cell.generate_cells_from_parameter_tensor(
            metasurface.atom_1d.tensor
        )
    
    for radial_ix, this_cell in enumerate(cell_array):
        polygon_component = unit_cell_to_gds_shape(this_cell, layer=layer, gds_layer=gds_layer)
        
        if inverted:
            box_hw = periodicity / 2.0
            box = gf.components.rectangle(
                size=(periodicity, periodicity),
                layer=gds_layer,
                centered=True,
            )
            polygon_component = gf.boolean(
                box, polygon_component, operation="not", layer=gds_layer
            )
        
        if polygon_component is None:
            continue
            
        # Create a named cell for this radial position
        cell = gf.Component(name=f"radial_{radial_ix}_{uuid.uuid4().hex[:8]}")
        cell.add_ref(polygon_component)
        
        # Get all positions along this radius circle
        locations_along_circle = get_loc_along_circle(
            radius_in_meters,
            n_pixels_radial,
            radial_ix,
            periodicity,
            r2c_basis,
        )
        
        # Place unit cells at each position
        for index in range(locations_along_circle.shape[1]):
            loc_x, loc_y = locations_along_circle[:, index]
            ref = device.add_ref(cell)
            ref.dmove((loc_x, loc_y))

    # Ensure export directory exists
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
    
    # Write GDS file
    device.write_gds(f"{export_directory}/{export_name}.gds")


def export_unit_cell_to_gds(
    unit_cell: "rcwa.UnitCell",
    export_name: str,
    export_directory: str | None = None,
    layer: int = 0,
    gds_layer: tuple[int, int] = (1, 0),
) -> None:
    """
    Export a single unit cell to a GDS file.
    
    Args:
        unit_cell: The unit cell to export
        export_name: Name for the output GDS file (without extension)
        export_directory: Directory to save the file (default: ./gds_export/)
        layer: Index of the layer within the unit cell to export
        gds_layer: Tuple of (layer, datatype) for GDS export
        
    Example:
        >>> from metabox import rcwa, export
        >>> # Create your unit cell...
        >>> export.export_unit_cell_to_gds(
        ...     unit_cell=my_cell,
        ...     export_name="single_pillar",
        ... )
    """
    if not HAS_GDSFACTORY:
        raise ImportError(
            "gdsfactory is required for GDS export. Install with: pip install gdsfactory>=9.28.1"
        )
    
    if export_directory is None:
        export_directory = "./gds_export/"
    
    component = unit_cell_to_gds_shape(unit_cell, layer=layer, gds_layer=gds_layer)
    
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
    
    component.write_gds(f"{export_directory}/{export_name}.gds")


def export_array_to_gds(
    unit_cells: list["rcwa.UnitCell"],
    positions: np.ndarray,
    export_name: str,
    export_directory: str | None = None,
    layer: int = 0,
    gds_layer: tuple[int, int] = (1, 0),
) -> None:
    """
    Export an array of unit cells at specified positions to a GDS file.
    
    Args:
        unit_cells: List of unit cells to export
        positions: Array of (x, y) positions in meters, shape (N, 2)
        export_name: Name for the output GDS file (without extension)
        export_directory: Directory to save the file (default: ./gds_export/)
        layer: Index of the layer within each unit cell to export
        gds_layer: Tuple of (layer, datatype) for GDS export
    """
    if not HAS_GDSFACTORY:
        raise ImportError(
            "gdsfactory is required for GDS export. Install with: pip install gdsfactory>=9.28.1"
        )
    
    if export_directory is None:
        export_directory = "./gds_export/"
    
    device = gf.Component(name=f"array_{uuid.uuid4().hex[:8]}")
    
    for idx, (cell, pos) in enumerate(zip(unit_cells, positions)):
        component = unit_cell_to_gds_shape(cell, layer=layer, gds_layer=gds_layer)
        cell_component = gf.Component(name=f"cell_{idx}_{uuid.uuid4().hex[:8]}")
        cell_component.add_ref(component)
        
        # Convert position to microns
        loc_x, loc_y = pos * 1e6
        ref = device.add_ref(cell_component)
        ref.dmove((loc_x, loc_y))
    
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)
    
    device.write_gds(f"{export_directory}/{export_name}.gds")