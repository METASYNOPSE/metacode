"""
Assembly module for metabox.

Defines a lens assembly and functionalities for simulating the performances 
of the lens assembly.

Compatible with Python 3.11.9+
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import itertools
import logging
import os
import re
from typing import Any

import dill
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
from matplotlib.ticker import EngFormatter
import sys

from . import expansion, metrics, modeling, propagation, rcwa 
from . import utils
from . import Incidence

# Suppress tensorflow warnings
tf.get_logger().setLevel(logging.ERROR)


@dataclasses.dataclass
class AtomArray2D:
    """Class to store the 2D atom array data and its metadata."""
    tensor: tf.Tensor
    period: float
    mmodel: modeling.Metamodel | None = None
    proto_unit_cell: rcwa.ProtoUnitCell | None = None
    cached_fields: list[tf.Tensor] | None = None

    def __post_init__(self):
        has_mmodel = self.mmodel is not None
        has_unit_cell = self.proto_unit_cell is not None
        if has_mmodel and has_unit_cell:
            raise ValueError("Cannot have both mmodel and parameterized unit cell.")
        if not has_mmodel and not has_unit_cell:
            raise ValueError("Must have either mmodel or parameterized unit cell.")
        self.use_mmodel = has_mmodel

    def find_feature_index(self, feature_str: str) -> int:
        """Returns the index of the feature in the structure tensor."""
        if self.use_mmodel:
            all_features = copy.deepcopy(self.mmodel.protocell.features)
        else:
            all_features = copy.deepcopy(self.proto_unit_cell.features)
        if feature_str not in [f.name for f in all_features]:
            raise ValueError(f"Feature {feature_str} not found in the atom array.")
        for i, feature in enumerate(all_features):
            if feature.name == feature_str:
                return i
        return 0

    def get_atom_array(self, incidence: Incidence) -> list[rcwa.UnitCell]:
        return AtomArray1D(self.tensor, self.period, self.mmodel).get_atom_array(incidence)

    def get_feature_map(self, feature: str) -> tf.Tensor:
        """Returns the structure of the atom array."""
        if self.use_mmodel:
            index = self.find_feature_index(feature)
        else:
            index = self.proto_unit_cell.find_feature_index(feature)
        matrix_width = int(np.sqrt(self.tensor.shape[-1]))
        return tf.reshape(self.tensor[index], [matrix_width, matrix_width])

    def set_feature_map(self, feature: str, new_values: np.ndarray | tf.Tensor) -> tf.Tensor:
        """Change the structure feature of the atom array."""
        if isinstance(self.tensor, tf.Variable):
            raise NotImplementedError("Changing feature map of tf.Variable not implemented.")

        index = self.find_feature_index(feature)
        matrix_width = int(np.sqrt(self.tensor.shape[-1]))

        if np.shape(new_values) != (matrix_width, matrix_width):
            raise ValueError(f"New values must have shape ({matrix_width}, {matrix_width}).")

        tsnp = self.tensor.numpy()
        tsnp[index] = new_values.flatten()
        self.tensor = tf.convert_to_tensor(tsnp)

    def show_feature_map(self, only_feature: str | None = None):
        """Shows the structure of the atom array."""
        n_pixels = int(np.sqrt(self.tensor.shape[-1]))
        diameter = self.period * n_pixels
        radius = diameter / 2.0
        features_with_wavelength = copy.deepcopy(self.mmodel.protocell.features)
        features_with_wavelength = [f.name for f in features_with_wavelength]
        all_features = list(features_with_wavelength)

        if only_feature is not None:
            if only_feature not in features_with_wavelength:
                raise ValueError(f"Feature {only_feature} not found.")
            all_features = [only_feature]

        for feature_name in all_features:
            feature_array = self.get_feature_map(feature_name)
            complex_str = ""
            if np.iscomplexobj(feature_array):
                complex_str = " (Real Part)"
                feature_array = np.real(feature_array)
            plt.figure(figsize=(5, 5), dpi=100)
            ax = plt.axes([0, 0.05, 0.9, 0.9])
            im = ax.imshow(feature_array, extent=[-radius, radius, -radius, radius])
            formatter0 = EngFormatter(unit="m")
            ax.xaxis.set_major_formatter(formatter0)
            ax.yaxis.set_major_formatter(formatter0)
            plt.locator_params(axis="y", nbins=3)
            plt.locator_params(axis="x", nbins=3)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(False)
            ax.set_title(f"Feature{complex_str}: {feature_name}")
            cax = plt.axes([0.95, 0.05, 0.05, 0.9])
            plt.colorbar(mappable=im, cax=cax)
            plt.show()

    def set_to_use_rcwa(self):
        """Skips the metamodel and directly simulate using RCWA."""
        if not self.use_mmodel:
            print("Already using RCWA simulation directly.")
            return
        metamodel = self.mmodel
        protocell = metamodel.protocell
        self.proto_unit_cell = protocell
        self.use_mmodel = False
        self.cached_fields = None
        self.sim_config = metamodel.sim_config


@dataclasses.dataclass
class AtomArray1D:
    """Class to store the 1D atom array data and its metadata."""
    tensor: tf.Tensor
    period: float
    mmodel: modeling.Metamodel | None = None
    proto_unit_cell: rcwa.ProtoUnitCell | None = None

    def __post_init__(self):
        has_mmodel = self.mmodel is not None
        has_unit_cell = self.proto_unit_cell is not None
        if has_mmodel and has_unit_cell:
            raise ValueError("Cannot have both mmodel and parameterized unit cell.")
        if not has_mmodel and not has_unit_cell:
            raise ValueError("Must have either mmodel or parameterized unit cell.")
        self.use_mmodel = has_mmodel
        self.cached_fields = None

    def find_feature_index(self, feature_str: str) -> int:
        """Returns the index of the feature in the structure tensor."""
        if self.use_mmodel:
            all_features = copy.deepcopy(self.mmodel.protocell.features)
        else:
            all_features = copy.deepcopy(self.proto_unit_cell.features)
        if feature_str not in [f.name for f in all_features]:
            raise ValueError(f"Feature {feature_str} not found in the atom array.")
        for i, feature in enumerate(all_features):
            if feature.name == feature_str:
                return i
        return 0

    def expand_to_2d(self, basis_dir: str = "basis_data") -> AtomArray2D:
        """Expand a 1d atom array to a 2d atom array."""
        new_tensor = expansion.expand_to_2d(self.tensor, basis_dir)
        new_shape = list(new_tensor.shape)
        new_shape = new_shape[:-2] + [-1]
        new_tensor = tf.reshape(new_tensor, new_shape)
        return AtomArray2D(new_tensor, self.period, self.mmodel)

    def get_atom_array(self, incidence: Incidence) -> list[rcwa.UnitCell]:
        """Returns the batched atom array."""
        if self.use_mmodel:
            return self.mmodel.protocell.generate_cells_from_parameter_tensor(self.tensor)
        return self.proto_unit_cell.generate_cells_from_parameter_tensor(self.tensor)

    def get_feature_map(self, feature: str) -> np.ndarray:
        """Returns the 2D feature array."""
        return self.expand_to_2d().get_feature_map(feature)

    def get_feature_map_1d(self, feature_str: str) -> np.ndarray:
        """Returns the 1D feature array."""
        if self.use_mmodel:
            index = self.find_feature_index(feature_str)
        else:
            index = self.proto_unit_cell.find_feature_index(feature_str)
        return self.tensor[index, :].numpy()

    def set_feature_map(self, feature: str, feature_array: np.ndarray):
        """Sets the 2D feature array."""
        index = self.find_feature_index(feature)
        tsnp = self.tensor.numpy()
        tsnp[index, :] = feature_array
        self.tensor = tf.convert_to_tensor(tsnp)

    def show_feature_map(self, only_feature: str | None = None):
        """Shows the structure of the atom array."""
        self.expand_to_2d().show_feature_map(only_feature)

    def set_to_use_rcwa(self):
        """Skips the metamodel and directly simulate using RCWA."""
        if not self.use_mmodel:
            print("Already using RCWA simulation directly.")
            return
        metamodel = self.mmodel
        protocell = metamodel.protocell
        self.proto_unit_cell = protocell
        self.use_mmodel = False
        self.cached_fields = None
        self.sim_config = metamodel.sim_config


@dataclasses.dataclass
class Surface:
    """Defines an optical surface."""
    diameter: float
    refractive_index: float
    thickness: float

    def optimizer_hook(self):
        """Hook for the optimizer to modify the surface."""
        pass

    def get_penalty(self) -> float:
        """Returns the penalty of the surface."""
        return 0.0


@dataclasses.dataclass
class Aperture(Surface):
    """Defines an aperture."""
    periodicity: float
    enable_propagator_cache: bool = False
    store_end_field: bool = False

    def __post_init__(self):
        self.n_pixels_radial = int(self.diameter / 2 / self.periodicity)
        self.propagator_cache = (None, None)

        def create_circular_mask(h, w, center=None, radius=None) -> np.ndarray:
            if center is None:
                center = (int(w / 2), int(h / 2))
            if radius is None:
                radius = min(center[0], center[1], w - center[0], h - center[1])
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            return dist_from_center <= radius

        width = self.n_pixels_radial * 2
        mask = create_circular_mask(width, width)
        self.mask = tf.cast(mask, tf.complex64)
        self.mask = tf.expand_dims(self.mask, axis=0)

    def optimizer_hook(self):
        pass

    def get_modulation_2d(self, incidence: Incidence) -> propagation.Field2D:
        batch_size = len(incidence.wavelength) * len(incidence.theta) * len(incidence.phi)
        return tf.repeat(self.mask, batch_size, axis=0)

    def get_end_field(
        self,
        incidence: Incidence,
        incident_field: propagation.Field2D,
        previous_refractive_index: float,
        lateral_shift: tuple[float, float] | None = None,
        use_padding: bool = True,
        use_x_pol: bool = True,
    ) -> propagation.Field2D:
        mod_tensor = self.get_modulation_2d(incidence)
        mod_field = propagation.Field2D(
            tensor=mod_tensor,
            period=self.periodicity,
            n_pixels=mod_tensor.shape[-1],
            wavelength=incidence.wavelength,
            theta=incidence.theta,
            phi=incidence.phi,
            upsampling=1,
            use_antialiasing=True,
            use_padding=use_padding,
        )
        mod_field = mod_field.modulated_by(incident_field)

        if self.thickness == 0:
            if self.store_end_field:
                self.end_field = mod_field
            return mod_field

        if self.enable_propagator_cache:
            if np.any(incidence != self.propagator_cache[0]):
                propagator = propagation.get_transfer_function(
                    field_like=mod_field,
                    ref_idx=self.refractive_index,
                    prop_dist=self.thickness,
                    lateral_shift=lateral_shift,
                )
                self.propagator_cache = (incidence, propagator)
            propagator = self.propagator_cache[1]
        else:
            propagator = propagation.get_transfer_function(
                field_like=mod_field,
                ref_idx=self.refractive_index,
                prop_dist=self.thickness,
                lateral_shift=lateral_shift,
            )
        end_field = propagation.propagate(mod_field, propagator)
        if self.store_end_field:
            self.end_field = end_field

        return end_field


@enum.unique
class FigureOfMerit(enum.Enum):
    """Defines the types of figure of merit functions."""
    STREHL_RATIO = 1
    LOG_STREHL_RATIO = 2
    MAX_INTENSITY = 3
    LOG_MAX_INTENSITY = 4
    CENTER_INTENSITY = 5
    LOG_CENTER_INTENSITY = 6


@dataclasses.dataclass
class CustomFigureOfMerit:
    """A data class for custom figure of merit functions."""
    expression: str
    data: dict[str, tf.Tensor] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.expression is None:
            raise ValueError("The expression cannot be None.")
        validation_errors = self.get_validation_errors()
        if validation_errors:
            raise ValueError(f"Expression validation failed:\n{validation_errors}")

    def get_validation_errors(self) -> str:
        allowed_keywords = [
            "psf", "strehl_ratio", "max_intensity", "center_intensity", "ideal_mtf", "dist",
        ]
        validation_errors = []
        allowed_keywords.extend(self.data.keys())

        for keyword in allowed_keywords:
            if keyword in self.expression and not re.search(rf"\b{keyword}\b", self.expression):
                validation_errors.append(
                    f"'{keyword}' should not be part of another word."
                )

        if not self.is_valid_expression(self.expression):
            validation_errors.append(
                f"Ensure using allowed operators: +, -, *, /, (, ), :.\n"
                f"And functions: {utils.TF_FUNCTIONS + ['log']}.\n"
                "And variables: psf, strehl_ratio, max_intensity, center_intensity, ideal_mtf."
            )

        return "\n".join(validation_errors)

    def is_valid_expression(self, user_expression: str) -> bool:
        allowed_patterns = [
            r"\bpsf\b", r"\bstrehl_ratio\b", r"\bmax_intensity\b",
            r"\bcenter_intensity\b", r"\bideal_mtf\b", r"\blog\b",
            r"\+", r"\-", r"\/", r"\(", r"\)", r"\:", r"\...",
            r"\d+(\.\d+)?",
        ]
        allowed_patterns.extend(self.data.keys())
        allowed_patterns.extend(utils.TF_FUNCTIONS)

        pattern = "|".join(allowed_patterns)
        elements = re.split(r"\s|(?<=[\(\)\+\-\*/])", user_expression)

        for element in elements:
            if element and not re.match(pattern, element):
                return False
        return True


@dataclasses.dataclass
class Metasurface(Surface):
    """Defines a metasurface."""
    metamodel: modeling.Metamodel | None = None
    proto_unit_cell: rcwa.ProtoUnitCell | None = None
    use_circular_expansions: bool = True
    enable_propagator_cache: bool = False
    set_structures_variable: bool = False
    store_end_field: bool = False
    xy_harmonics: tuple[int, int] = (3, 3)
    unit_cell_spatial_res: int = 128
    minibatch_size: int = 100

    def __post_init__(self):
        has_metamodel = self.metamodel is not None
        has_proto_unit_cell = self.proto_unit_cell is not None
        if not has_metamodel and not has_proto_unit_cell:
            raise ValueError("Either metamodel or proto_unit_cell must be specified.")
        if has_metamodel and has_proto_unit_cell:
            raise ValueError("Only ONE of metamodel or proto_unit_cell must be specified.")
        if has_proto_unit_cell:
            periodicity_tuple = self.proto_unit_cell.proto_unit_cell.periodicity
            if periodicity_tuple[0] != periodicity_tuple[1]:
                raise NotImplementedError("Non-square unit cell not implemented yet.")

        self.sim_config = rcwa.SimConfig(
            xy_harmonics=self.xy_harmonics,
            resolution=self.unit_cell_spatial_res,
            minibatch_size=self.minibatch_size,
            return_tensor=True,
            return_zeroth_order=True,
            use_transmission=True,
            include_z_comp=False,
        )
        self.use_metamodel = has_metamodel
        if self.use_metamodel:
            self.periodicity = self.metamodel.protocell.proto_unit_cell.periodicity[0]
        else:
            self.periodicity = self.proto_unit_cell.proto_unit_cell.periodicity[0]
        self.n_pixels_radial = int(self.diameter / 2 / self.periodicity)

        if self.use_metamodel:
            args = (self.n_pixels_radial, self.periodicity, self.metamodel, self.set_structures_variable)
            if self.use_circular_expansions:
                self.atom_1d = initialize_1d_atom_array_metamodel(*args)
            else:
                self.atom_2d = initialize_2d_atom_array_metamodel(*args)
        else:
            args = (self.n_pixels_radial, self.proto_unit_cell, self.set_structures_variable)
            if self.use_circular_expansions:
                self.atom_1d = initialize_1d_atom_array_proto_unit_cell(*args)
                self.atom_1d.sim_config = self.sim_config
            else:
                self.atom_2d = initialize_2d_atom_array_proto_unit_cell(*args)
                self.atom_2d.sim_config = self.sim_config

        self.propagator_cache = (None, None)
        self.variables = []
        if self.set_structures_variable:
            if self.use_circular_expansions:
                self.variables.append(self.atom_1d.tensor)
            else:
                self.variables.append(self.atom_2d.tensor)

    def get_atom_positions(self) -> np.ndarray:
        x_pos = y_pos = np.linspace(-self.diameter / 2, self.diameter / 2, 2 * self.n_pixels_radial)
        xx_pos, yy_pos = np.meshgrid(x_pos, y_pos)
        return np.stack([xx_pos, yy_pos], axis=-1)

    def get_modulation_2d(
        self,
        incidence: Incidence,
        use_padding: bool = True,
        use_x_pol: bool = True,
    ) -> propagation.Field2D:
        if self.use_circular_expansions:
            if self.atom_1d.cached_fields is None:
                field_1d_x, field_1d_y = structure_to_field_1d(
                    self.atom_1d, incidence, use_padding=use_padding
                )
                self.atom_1d.cached_fields = field_1d_x, field_1d_y

            field_1d_x, field_1d_y = self.atom_1d.cached_fields
            output_field = field_1d_x.expand_to_2d() if use_x_pol else field_1d_y.expand_to_2d()
        else:
            if self.atom_2d.cached_fields is None:
                self.atom_2d.cached_fields = structure_to_field_2d(
                    self.atom_2d, incidence, use_padding=use_padding
                )
            output_field = self.atom_2d.cached_fields[0] if use_x_pol else self.atom_2d.cached_fields[1]

        return output_field

    def get_end_field(
        self,
        incidence: Incidence,
        incident_field: propagation.Field2D,
        previous_refractive_index: float,
        lateral_shift: tuple[float, float] | None = None,
        use_padding: bool = True,
        use_x_pol: bool = True,
    ) -> propagation.Field2D:
        field_2d = self.get_modulation_2d(incidence, use_padding=use_padding, use_x_pol=use_x_pol)
        field_2d = field_2d.modulated_by(incident_field)
        if self.thickness == 0:
            if self.store_end_field:
                self.end_field = field_2d
            return field_2d

        if self.enable_propagator_cache:
            if np.any(incidence != self.propagator_cache[0]):
                propagator = propagation.get_transfer_function(
                    field_like=field_2d,
                    ref_idx=self.refractive_index,
                    prop_dist=self.thickness,
                    lateral_shift=lateral_shift,
                )
                self.propagator_cache = (incidence, propagator)
            propagator = self.propagator_cache[1]
        else:
            propagator = propagation.get_transfer_function(
                field_like=field_2d,
                ref_idx=self.refractive_index,
                prop_dist=self.thickness,
                lateral_shift=lateral_shift,
            )
        field_2d = propagation.propagate(field_2d, propagator)
        if self.store_end_field:
            self.end_field = field_2d

        return field_2d

    def get_feature_map(self):
        if self.use_circular_expansions:
            return self.atom_1d.get_feature_map()
        else:
            return self.atom_2d.get_feature_map()

    def set_feature_map(self, feature_str: str, new_value: np.ndarray | tf.Tensor):
        if self.use_circular_expansions:
            self.atom_1d.set_feature_map(feature_str, new_value)
        else:
            self.atom_2d.set_feature_map(feature_str, new_value)

    def show_feature_map(self, only_feature: str | None = None):
        if self.use_circular_expansions:
            self.atom_1d.show_feature_map(only_feature=only_feature)
        else:
            self.atom_2d.show_feature_map(only_feature=only_feature)

    def optimizer_hook(self):
        self.clear_cache()

    def set_to_use_rcwa(self):
        if self.use_circular_expansions:
            self.atom_1d.set_to_use_rcwa()
        else:
            self.atom_2d.set_to_use_rcwa()

    def clear_cache(self):
        if self.use_circular_expansions:
            self.atom_1d.cached_fields = None
        else:
            self.atom_2d.cached_fields = None


@dataclasses.dataclass
class LensAssembly:
    """Defines a lens assembly."""
    surfaces: list[Surface]
    incidence: Incidence
    aperture_stop_index: int = -1
    figure_of_merit: FigureOfMerit | CustomFigureOfMerit | None = None
    use_antialiasing: bool = True
    use_padding: bool = True
    use_x_pol: bool = True

    def __post_init__(self):
        self.upsampling = 1
        focal_length = sum(s.thickness for s in self.surfaces[self.aperture_stop_index:])

        ref_surface = self.surfaces[self.aperture_stop_index]
        n_pixels = ref_surface.n_pixels_radial * 2

        self.field_properties = propagation.FieldProperties(
            n_pixels=n_pixels,
            wavelength=list(self.incidence.wavelength),
            theta=list(self.incidence.theta),
            phi=list(self.incidence.phi),
            period=ref_surface.periodicity,
            upsampling=self.upsampling,
            use_antialiasing=self.use_antialiasing,
            use_padding=self.use_padding,
        )

        if self.figure_of_merit is not None:
            self.ideal_mtf = metrics.get_ideal_mtf_volume(
                field_props=self.field_properties,
                focal_length=focal_length,
            )

    def compute_field_on_sensor(self) -> propagation.Field2D:
        current_field = propagation.get_incident_field_2d(self.field_properties)
        for idx, surface in enumerate(self.surfaces):
            lateral_shift = None if idx == len(self.surfaces) - 1 else (0, 0)
            previous_refractive_index = 1.0 if idx == 0 else self.surfaces[idx - 1].refractive_index

            current_field = surface.get_end_field(
                incidence=self.incidence,
                incident_field=current_field,
                previous_refractive_index=previous_refractive_index,
                lateral_shift=lateral_shift,
                use_padding=self.use_padding,
                use_x_pol=self.use_x_pol,
            )
        return current_field

    def show_psf(self, use_wavelength_average: bool = False, crop_factor: float = 1.0) -> None:
        if use_wavelength_average:
            self.compute_field_on_sensor().wavelength_average().show_intensity(crop_factor=crop_factor)
        else:
            self.compute_field_on_sensor().show_intensity(crop_factor=crop_factor)
        self.clear_cache()

    def show_color_psf(self, crop_factor: float = 1.0) -> None:
        self.compute_field_on_sensor().show_color_intensity(crop_factor=crop_factor)
        self.clear_cache()

    def compute_strehl_ratio(self):
        field = self.compute_field_on_sensor()
        return metrics.get_mtf_volume(field) / self.ideal_mtf[:, tf.newaxis]

    def compute_max_intensity(self):
        field = self.compute_field_on_sensor()
        return metrics.get_max_intensity(field)

    def compute_center_intensity(self):
        field = self.compute_field_on_sensor()
        return metrics.get_center_intensity(field)

    def get_variables(self) -> list[tf.Variable]:
        variables = []
        for surface in self.surfaces:
            variables += getattr(surface, 'variables', [])
        return variables

    def compute_FOM(self) -> tf.Tensor:
        if self.figure_of_merit is None:
            raise ValueError("No figure of merit defined.")
        elif isinstance(self.figure_of_merit, CustomFigureOfMerit):
            return self.compute_custom_FOM(self.figure_of_merit)
        elif self.figure_of_merit not in FigureOfMerit:
            raise ValueError(f"Invalid figure of merit {self.figure_of_merit}.")

        if self.figure_of_merit == FigureOfMerit.STREHL_RATIO:
            return tf.reduce_mean(self.compute_strehl_ratio())
        elif self.figure_of_merit == FigureOfMerit.LOG_STREHL_RATIO:
            return tf.reduce_mean(tf.math.log(self.compute_strehl_ratio()))
        elif self.figure_of_merit == FigureOfMerit.MAX_INTENSITY:
            return tf.reduce_mean(self.compute_max_intensity())
        elif self.figure_of_merit == FigureOfMerit.LOG_MAX_INTENSITY:
            return tf.reduce_mean(tf.math.log(self.compute_max_intensity()))
        elif self.figure_of_merit == FigureOfMerit.CENTER_INTENSITY:
            return tf.reduce_mean(self.compute_center_intensity())
        elif self.figure_of_merit == FigureOfMerit.LOG_CENTER_INTENSITY:
            return tf.reduce_mean(tf.math.log(self.compute_center_intensity()))
        else:
            raise ValueError("Invalid figure of merit.")

    def compute_custom_FOM(self, custom_FOM: CustomFigureOfMerit) -> tf.Tensor:
        tf_functions = utils.TF_FUNCTIONS
        user_expression = custom_FOM.expression

        replacements = {r"\*": " * ", r"\/": " / ", r"\+": " + ", r"\-": " - "}
        for pattern, replacement in replacements.items():
            user_expression = re.sub(pattern, replacement, user_expression)

        psf = None
        if "psf" in user_expression:
            psf = self.compute_field_on_sensor()
        if "strehl_ratio" in user_expression:
            if psf is None:
                psf = self.compute_field_on_sensor()
            strehl_ratio = metrics.get_mtf_volume(psf) / self.ideal_mtf
        if "max_intensity" in user_expression:
            if psf is None:
                psf = self.compute_field_on_sensor()
            max_intensity = metrics.get_max_intensity(psf)
        if "center_intensity" in user_expression:
            if psf is None:
                psf = self.compute_field_on_sensor()
            center_intensity = metrics.get_center_intensity(psf)
        if psf is not None:
            psf_tensor = tf.math.abs(psf.tensor) ** 2

        replacements = {"ideal_mtf": "self.ideal_mtf", "psf": "psf_tensor", "log": "tf.math.log"}
        for old, new in replacements.items():
            user_expression = user_expression.replace(old, new)

        for func in tf_functions:
            user_expression = user_expression.replace(func, f"tf.{func}")

        if custom_FOM.data:
            for key in custom_FOM.data.keys():
                user_expression = user_expression.replace(key, f"custom_FOM.data['{key}']")

        return eval(user_expression)

    def compute_penalty(self) -> tf.Tensor:
        penalty = 0
        for surface in self.surfaces:
            penalty += surface.get_penalty()
        return penalty

    def copy(self) -> LensAssembly:
        return copy_lens_assembly(self)

    def save(self, name: str, save_dir: str = "./saved_lens_assemblies", overwrite: bool = False):
        save_lens_assembly(self, name, save_dir, overwrite)

    def optimizer_hook(self):
        for surface in self.surfaces:
            surface.optimizer_hook()

    def set_to_use_rcwa(self):
        for surface in self.surfaces:
            if isinstance(surface, Metasurface):
                surface.set_to_use_rcwa()

    def clear_cache(self):
        for surface in self.surfaces:
            if isinstance(surface, Metasurface):
                surface.clear_cache()


# Helper functions

def structure_to_field_1d(
    structure: AtomArray1D,
    incidence: Incidence,
    feature_order: list[str] | None = None,
    use_padding: bool = True,
) -> tuple[propagation.Field1D, propagation.Field1D]:
    """Converts a structure to a 1D field."""
    if structure.use_mmodel:
        return structure_to_field_1d_mmodel(structure, incidence, feature_order, use_padding)
    else:
        return structure_to_field_1d_proto_unit_cell(structure, incidence, feature_order, use_padding)


def structure_to_field_1d_proto_unit_cell(
    structure: AtomArray1D,
    incidence: Incidence,
    feature_order: list[str] | None = None,
    use_padding: bool = True,
) -> tuple[propagation.Field1D, propagation.Field1D]:
    """Converts a structure to a 1D field using proto unit cell."""
    structure_n_features = structure.tensor.shape[0]
    proto_uc_n_features = len(structure.proto_unit_cell.features)
    if structure_n_features != proto_uc_n_features:
        raise ValueError("Number of features mismatch between structure and proto_unit_cell.")

    fields_1d = rcwa.simulate_parameterized_unit_cells(
        parameter_tensor=structure.tensor,
        proto_cell=structure.proto_unit_cell,
        incidence=incidence,
        sim_config=structure.sim_config,
    )

    radius_size = fields_1d.shape[1]
    field_x = propagation.Field1D(
        tensor=fields_1d[..., 0],
        n_pixels=radius_size * 2,
        wavelength=list(incidence.wavelength),
        theta=list(incidence.theta),
        phi=list(incidence.phi),
        period=structure.period,
        upsampling=1,
        use_padding=use_padding,
        use_antialiasing=True,
    )
    field_y = propagation.Field1D(
        tensor=fields_1d[..., 1],
        n_pixels=radius_size * 2,
        wavelength=list(incidence.wavelength),
        theta=list(incidence.theta),
        phi=list(incidence.phi),
        period=structure.period,
        upsampling=1,
        use_padding=use_padding,
        use_antialiasing=True,
    )

    return field_x, field_y


def structure_to_field_1d_mmodel(
    structure: AtomArray1D,
    incidence: Incidence,
    feature_order: list[str] | None = None,
    use_padding: bool = True,
) -> tuple[propagation.Field1D, propagation.Field1D]:
    """Converts a structure to a 1D field using metamodel."""
    structure_n_features = structure.tensor.shape[0]
    metamodel_n_features = len(structure.mmodel.protocell.features)
    if structure_n_features != metamodel_n_features:
        raise ValueError("Number of features mismatch between structure and metamodel.")

    if feature_order is None:
        the_features = structure.mmodel.protocell.features.copy()
        feature_order = [f.name for f in the_features]
    else:
        feature_order = feature_order.copy()

    radius_size = structure.tensor.shape[-1]
    angles = len(incidence.theta) * len(incidence.phi)
    batch_number = len(incidence.wavelength) * angles

    lambda_base = tf.cast(incidence.wavelength, tf.float32)
    wave_repeated = tf.repeat(lambda_base, radius_size)
    wave_angle_repeated = tf.repeat(wave_repeated, [angles])

    wave_angle_repeated = tf.expand_dims(wave_angle_repeated, axis=0)
    structure_var_tiled = tf.tile(structure.tensor, [1, batch_number])

    inputs = tf.concat([wave_angle_repeated, structure_var_tiled], 0)
    inputs = tf.math.real(inputs)
    inputs = tf.cast(inputs, tf.float32)
    inputs = tf.transpose(inputs)
    outputs = structure.mmodel.model(inputs)
    outputs = tf.transpose(outputs)

    x_vec = tf.cast([[1.0], [0.0]], tf.complex64)
    y_vec = tf.cast([[0.0], [1.0]], tf.complex64)
    tx = tf.reduce_sum(outputs * x_vec, axis=0)
    ty = tf.reduce_sum(outputs * y_vec, axis=0)
    tx = tf.reshape(tx, [batch_number, radius_size])
    ty = tf.reshape(ty, [batch_number, radius_size])

    field_x = propagation.Field1D(
        tensor=tx,
        n_pixels=radius_size * 2,
        wavelength=list(incidence.wavelength),
        theta=list(incidence.theta),
        phi=list(incidence.phi),
        period=structure.period,
        upsampling=1,
        use_padding=use_padding,
        use_antialiasing=True,
    )
    field_y = propagation.Field1D(
        tensor=ty,
        n_pixels=radius_size * 2,
        wavelength=list(incidence.wavelength),
        theta=list(incidence.theta),
        phi=list(incidence.phi),
        period=structure.period,
        upsampling=1,
        use_padding=use_padding,
        use_antialiasing=True,
    )

    return field_x, field_y


def structure_to_field_2d(
    structure: AtomArray2D,
    incidence: Incidence,
    feature_order: list[str] | None = None,
    use_padding: bool = True,
) -> list[propagation.Field2D]:
    """Converts a structure to a 2D field."""
    dummy_field_x, dummy_field_y = structure_to_field_1d(
        structure=structure,
        incidence=incidence,
        feature_order=feature_order,
        use_padding=use_padding,
    )

    fields_rtn = []
    for dummy_field in [dummy_field_x, dummy_field_y]:
        dummy_tensor = dummy_field.tensor
        ts_shape = list(dummy_tensor.shape)
        n_pixels = int(np.sqrt(ts_shape[-1]))
        ts_shape.pop(-1)
        ts_shape.extend([n_pixels, n_pixels])
        dummy_tensor = tf.reshape(dummy_tensor, ts_shape)
        dummy_tensor = tf.cast(dummy_tensor, tf.complex64)

        fields_rtn.append(
            propagation.Field2D(
                tensor=dummy_tensor,
                n_pixels=n_pixels,
                wavelength=dummy_field.wavelength,
                theta=dummy_field.theta,
                phi=dummy_field.phi,
                period=structure.period,
                upsampling=1,
                use_padding=use_padding,
                use_antialiasing=True,
            )
        )

    return fields_rtn


def initialize_1d_atom_array_proto_unit_cell(
    n_pixels_radial: int,
    proto_unit_cell: rcwa.ProtoUnitCell,
    set_structures_variable: bool = False,
) -> AtomArray1D:
    """Initializes a 1D atom array from proto unit cell."""
    periodicity_xy = proto_unit_cell.proto_unit_cell.periodicity
    if periodicity_xy[0] != periodicity_xy[1]:
        raise ValueError("x and y periodicity must be equal.")

    variables = proto_unit_cell.generate_initial_variables(n_pixels_radial)
    if not set_structures_variable:
        variables = tf.constant(variables)

    return AtomArray1D(
        tensor=variables,
        period=periodicity_xy[0],
        proto_unit_cell=proto_unit_cell,
    )


def initialize_2d_atom_array_proto_unit_cell(
    n_pixels_radial: int,
    proto_unit_cell: rcwa.ProtoUnitCell,
    set_structures_variable: bool = False,
) -> AtomArray2D:
    """Initializes a 2D atom array from proto unit cell."""
    dummy_atom_array = initialize_1d_atom_array_proto_unit_cell(
        n_pixels_radial=(n_pixels_radial * 2) ** 2,
        proto_unit_cell=proto_unit_cell,
        set_structures_variable=set_structures_variable,
    )

    return AtomArray2D(
        tensor=dummy_atom_array.tensor,
        period=proto_unit_cell.period,
        proto_unit_cell=proto_unit_cell,
    )


def initialize_1d_atom_array_metamodel(
    n_pixels_radial: int,
    period: float,
    mmodel: modeling.Metamodel,
    set_structures_variable: bool = False,
) -> AtomArray1D:
    """Initializes a 1D atom array from metamodel."""
    tensor_columns = []
    clip_value_min = []
    clip_value_max = []
    for feature in mmodel.protocell.features:
        vmin = [feature.vmin]
        vmax = [feature.vmax]
        clip_value_min.append(vmin)
        clip_value_max.append(vmax)
        tensor_columns.append(tf.random.uniform([n_pixels_radial], vmin, vmax))
    tensor = tf.stack(tensor_columns, axis=0)
    constraint_func = lambda x: tf.clip_by_value(x, clip_value_min, clip_value_max)

    if set_structures_variable:
        tensor = tf.Variable(tensor, constraint=constraint_func)

    return AtomArray1D(
        tensor=tensor,
        period=period,
        mmodel=mmodel,
    )


def initialize_2d_atom_array_metamodel(
    n_pixels_radial: int,
    period: float,
    mmodel: modeling.Metamodel,
    set_structures_variable: bool = False,
) -> AtomArray2D:
    """Initializes a 2D atom array from metamodel."""
    dummy_atom_array = initialize_1d_atom_array_metamodel(
        n_pixels_radial=(n_pixels_radial * 2) ** 2,
        period=period,
        mmodel=mmodel,
        set_structures_variable=set_structures_variable,
    )

    return AtomArray2D(
        tensor=dummy_atom_array.tensor,
        period=period,
        mmodel=mmodel,
    )


def copy_lens_assembly(lens_assembly: LensAssembly) -> LensAssembly:
    """Returns a copy of the lens assembly."""
    with utils.suppress_stdout_stderr():
        save_lens_assembly(lens_assembly, "temp", "./", overwrite=True)
        return load_lens_assembly("temp", "./")


def save_lens_assembly(
    lens_assembly: LensAssembly,
    name: str,
    save_dir: str = "./saved_lens_assemblies",
    overwrite: bool = False,
) -> None:
    """Saves the lens assembly to disk."""
    save_path = os.path.join(save_dir, name)
    if os.path.exists(save_path):
        if not overwrite:
            raise ValueError(f"Lens assembly {name} already exists. Set overwrite=True.")
    else:
        os.mkdir(save_path)

    save_path_pkl = os.path.join(save_path, "lens_assembly.pkl")
    with utils.suppress_stdout_stderr():
        new_self = copy.deepcopy(lens_assembly)
    for surface in new_self.surfaces:
        if not isinstance(surface, Metasurface):
            continue
        if not surface.use_metamodel:
            continue
        del surface.metamodel
        surface.propagator_cache = (None, None)
        if surface.use_circular_expansions:
            del surface.atom_1d.mmodel

    with utils.suppress_stdout_stderr():
        with open(save_path_pkl, "wb") as f:
            dill.dump(new_self, f)
        for i, surface in enumerate(lens_assembly.surfaces):
            if not isinstance(surface, Metasurface):
                continue
            if not surface.use_metamodel:
                continue
            surface.metamodel.save(f"surface_{i}_metamodel", save_path, overwrite)


def load_lens_assembly(name: str, save_dir: str = "./saved_lens_assemblies") -> LensAssembly:
    """Loads a lens assembly from disk."""
    lens_assembly = dill.load(
        open(os.path.join(save_dir, name, "lens_assembly.pkl"), "rb")
    )
    for i, surface in enumerate(lens_assembly.surfaces):
        if not isinstance(surface, Metasurface):
            continue
        if not surface.use_metamodel:
            continue
        surface.metamodel = modeling.load_metamodel(
            f"surface_{i}_metamodel",
            save_dir=os.path.join(save_dir, name),
        )
        if surface.use_circular_expansions:
            surface.atom_1d.mmodel = surface.metamodel
    return lens_assembly


def optimize_single_lens_assembly(
    lens_assembly: LensAssembly,
    optimizer: tf.keras.optimizers.Optimizer,
    n_iter: int,
    verbose: int = 0,
    keep_best: bool = True,
) -> list[float]:
    """Optimizes a single lens assembly."""
    variables = lens_assembly.get_variables()
    loss_history = []
    lowest_loss = np.inf
    best_lens_assembly_vars = lens_assembly.get_variables()
    
    tr = range(n_iter) if verbose <= 0 else tqdm.trange(n_iter, desc="Optimizing", leave=True)
    
    for _ in tr:
        with tf.GradientTape() as tape:
            loss = -lens_assembly.compute_FOM()
            loss += lens_assembly.compute_penalty()
        if keep_best and loss < lowest_loss:
            lowest_loss = loss
            best_lens_assembly_vars = lens_assembly.get_variables().copy()
        grads = tape.gradient(loss, variables)
        for grad, variable in zip(grads, variables):
            grad = tf.math.real(grad)
            optimizer.apply_gradients([(grad, variable)])

        loss_history.append(-loss.numpy())
        if verbose > 0:
            tr.set_description(f"Loss: {loss.numpy():.6F}")
        lens_assembly.optimizer_hook()

    if keep_best:
        for variable_ts, best_variable_ts in zip(lens_assembly.get_variables(), best_lens_assembly_vars):
            variable_ts.assign(best_variable_ts)
    return loss_history