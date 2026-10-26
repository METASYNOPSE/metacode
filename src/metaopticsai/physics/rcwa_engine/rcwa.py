"""
RCWA simulation module for metabox.

This module provides classes and functions for defining unit cells and 
running RCWA simulations.

Compatible with Python 3.11.9+
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import gc
import glob
import logging
import os
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
from scipy import interpolate
import sys

from . import raster, utils 
from . import rcwa_tf
from . import CoordType, Feature, Incidence, ParameterType

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_ROOT = os.path.abspath(os.path.dirname(__file__))

# Suppress tensorflow warnings
tf.get_logger().setLevel(logging.ERROR)


def _get_features(parameter) -> list[Feature]:
    """Returns the features of the shape."""
    if isinstance(parameter, Feature):
        yield parameter
    elif isinstance(parameter, (list, tuple, set)):
        for item in parameter:
            yield from _get_features(item)


class Parameterizable:
    """Defines a parameterizable object."""

    def __init__(self):
        self.unique_features = self.get_unique_features()

    def get_unique_features(self) -> list[Feature]:
        """Returns the unique features of the shape (non-recursively)."""
        return list(set(self.get_features()))

    def get_features(self) -> list[Feature]:
        """Returns the features of the shape."""
        for field in dataclasses.fields(self):
            parameter = getattr(self, field.name)
            yield from _get_features(parameter)

    def initialize_values(
        self,
        value_assignment: tuple[list[Feature], list[float]] | None = None,
    ) -> None:
        """Initializes the variables."""
        if value_assignment is not None:
            if len(value_assignment) != 2:
                raise ValueError("value_assignment must be a tuple of length 2.")
            elif len(value_assignment[0]) != len(value_assignment[1]):
                raise ValueError("value_assignment must have lists of equal length.")

            features, values = value_assignment
            for feature, value in zip(features, values):
                feature.initial_value = value

        for feature in self.unique_features:
            feature.initialize_value()

    def replace_feature_with_value(self) -> None:
        for field in dataclasses.fields(self):
            parameter = getattr(self, field.name)
            if isinstance(parameter, Feature):
                setattr(self, field.name, parameter.value)
            elif isinstance(parameter, (list, tuple, set)):
                new_parameter = []
                for item in parameter:
                    if isinstance(item, Feature):
                        new_parameter.append(item.value)
                    else:
                        new_parameter.append(item)
                setattr(self, field.name, new_parameter)

    def get_variables(self) -> list[tf.Variable]:
        """Returns the variables of the shape."""
        variables = []
        for feature in self.unique_features:
            if feature.value is not None:
                if isinstance(feature.value, tf.Variable):
                    variables.append(feature.value)
        return variables


@dataclasses.dataclass
class Shape(Parameterizable):
    """Defines a shape."""
    material: ParameterType | None

    def __post_init__(self):
        return super().__init__()


@dataclasses.dataclass
class Polygon(Shape):
    """Defines a polygon."""
    vertices: list[CoordType]

    def __post_init__(self):
        if len(self.vertices) < 3:
            raise ValueError("A polygon must have at least 3 vertices.")
        for vertex in self.vertices:
            if len(vertex) != 2:
                raise ValueError("Each vertex must be a tuple of (x, y) coordinates.")
        return super().__post_init__()

    def get_shape(self, wavelength: float | None = None):
        if isinstance(self.material, Material):
            if wavelength is None:
                raise ValueError("Wavelength required for Material index.")
            value = self.material.index_at(wavelength)
        else:
            value = self.material
        return raster.Polygon(value=value, points=self.vertices)

    def get_vertices(self):
        return self.vertices


@dataclasses.dataclass
class Rectangle(Shape):
    """Defines a rectangle."""
    x_width: ParameterType
    y_width: ParameterType
    x_pos: ParameterType = 0
    y_pos: ParameterType = 0
    rotation_deg: ParameterType = 0

    def __post_init__(self):
        return super().__post_init__()

    def get_shape(self, wavelength: float | None = None):
        if isinstance(self.material, Material):
            if wavelength is None:
                raise ValueError("Wavelength required for Material index.")
            value = self.material.index_at(wavelength)
        else:
            value = self.material
        return raster.Rectangle(
            value=value,
            center=(self.x_pos, self.y_pos),
            x_width=self.x_width,
            y_width=self.y_width,
            rotation_deg=self.rotation_deg,
        )

    def get_vertices(self):
        return raster.rectangle_to_vertices(
            center=(self.x_pos, self.y_pos),
            x_width=self.x_width,
            y_width=self.y_width,
            rotation_deg=self.rotation_deg,
        )


@dataclasses.dataclass
class Circle(Shape):
    """Defines a circle."""
    radius: ParameterType
    x_pos: ParameterType = 0
    y_pos: ParameterType = 0

    def __post_init__(self):
        return super().__post_init__()

    def get_shape(self, wavelength: float | None = None):
        if isinstance(self.material, Material):
            if wavelength is None:
                raise ValueError("Wavelength required for Material index.")
            value = self.material.index_at(wavelength)
        else:
            value = self.material
        return raster.Circle(
            value=value,
            center=(self.x_pos, self.y_pos),
            radius=self.radius,
        )

    def get_vertices(self, num_of_vertices: int = 21):
        vertices = []
        for i in range(num_of_vertices):
            angle = i * 2 * np.pi / num_of_vertices
            x = self.x_pos + self.radius * np.cos(angle)
            y = self.y_pos + self.radius * np.sin(angle)
            vertices.append((x, y))
        return vertices


def duplicate_shape(shape: Shape, num_of_duplicates: int) -> list[Shape]:
    """Generates a list of duplicate parameterized shapes."""
    shapes = []
    for i in range(num_of_duplicates):
        new_shape = copy.deepcopy(shape)
        features = new_shape.unique_features
        for feature in features:
            feature.name = f"{feature.name}~{i}"
        shapes.append(new_shape)
    return shapes


@dataclasses.dataclass
class Layer(Parameterizable):
    """Defines a layer."""
    material: Feature | float | Material
    thickness: Feature | float
    shapes: tuple[Shape, ...] = ()
    enforce_4fold_symmetry: bool = False

    def __post_init__(self):
        Parameterizable.__init__(self)

    def get_shapes(self, wavelength: float | None = None):
        shapes = []
        for shape in self.shapes:
            shapes.append(shape.get_shape(wavelength))
        return shapes

    def initialize_values(
        self,
        value_assignment: tuple[list[Feature], list[float]] | None = None,
    ) -> None:
        super().initialize_values(value_assignment)
        for shape in self.shapes:
            shape.initialize_values(value_assignment)

    def get_layer_unique_features(self) -> list[Feature]:
        all_features = copy.deepcopy(self.unique_features)
        for shape in self.shapes:
            all_features.extend(shape.unique_features)
        return list(set(all_features))

    def get_variables(self) -> list[tf.Variable]:
        all_features = copy.deepcopy(self.unique_features)
        for shape in self.shapes:
            all_features.extend(shape.unique_features)
        unique_features = list(set(all_features))
        variables = []
        for feature in unique_features:
            if feature.value is not None:
                if isinstance(feature.value, tf.Variable):
                    variables.append(feature.value)
        return variables


@dataclasses.dataclass
class UnitCell(Parameterizable):
    """Defines a unit cell."""
    layers: list[Layer]
    periodicity: tuple[ParameterType, ParameterType]
    refl_index: ParameterType = 1.0
    tran_index: ParameterType = 1.0

    def __post_init__(self):
        super().__init__()
        if len(self.periodicity) != 2:
            raise ValueError("Periodicity must be a tuple of (x, y) in meters.")

    def initialize_values(
        self,
        value_assignment: tuple[list[Feature], list[float]] | None = None,
    ) -> None:
        super().initialize_values(value_assignment)
        for layer in self.layers:
            layer.initialize_values(value_assignment)

    def replace_features(self):
        _replace_feature_with_value_in_dataclass(self)

    def get_cell_unique_features(self) -> list[Feature]:
        all_features = copy.deepcopy(self.unique_features)
        for layer in self.layers:
            all_features.extend(layer.get_layer_unique_features())
        return list(set(all_features))

    def get_variables(self) -> list[tf.Variable]:
        unique_features = self.get_cell_unique_features()
        variables = []
        for feature in unique_features:
            if feature.value is not None:
                if isinstance(feature.value, tf.Variable):
                    variables.append(feature.value)
        return variables

    def get_epsilon(self, x_resolution: int, wavelength: float) -> tf.Tensor:
        pixel_density = self.periodicity[0] / float(x_resolution)
        epsilon_all = []
        for layer in self.layers:
            epsilon_layer = (
                _rasterize_layer(
                    layer=layer,
                    periodicity=self.periodicity,
                    pixel_density=pixel_density,
                    enforce_4fold_symmetry=layer.enforce_4fold_symmetry,
                    wavelength=wavelength,
                )
                ** 2
            )
            epsilon_all.append(tf.cast(epsilon_layer, tf.complex64))
        epsilon_all = tf.cast(epsilon_all, tf.complex64)
        return tf.stack(epsilon_all, axis=0)

    def get_thickness(self) -> tf.Tensor:
        return tf.cast(
            [tf.math.real(layer.thickness) for layer in self.layers],
            tf.float32,
        )

    def find_feature_index(self, feature_str):
        for i, feature in enumerate(self.unique_features):
            if feature.name == feature_str:
                return i
        raise ValueError("Feature not found.")


def _replace_this_feature_with_value_recursively(parent: Any, child_field: Any) -> None:
    """Recursively replaces the features with their values."""
    parent_is_list_or_tuple = isinstance(parent, (list, tuple))
    parent_is_tuple = isinstance(parent, tuple)
    if parent_is_list_or_tuple:
        field_content = child_field
    else:
        field_content = getattr(parent, child_field.name)

    if isinstance(field_content, Feature):
        if field_content.value is None:
            field_content.initialize_value()
        if parent_is_list_or_tuple:
            if parent_is_tuple:
                parent = list(parent)
            idx = parent.index(child_field)
            parent[idx] = field_content.value
            if parent_is_tuple:
                parent = tuple(parent)
        else:
            setattr(parent, child_field.name, field_content.value)
    elif isinstance(field_content, (list, tuple)):
        for child_field_child_field in field_content:
            _replace_this_feature_with_value_recursively(
                parent=field_content, child_field=child_field_child_field
            )
    elif dataclasses.is_dataclass(field_content):
        _replace_feature_with_value_in_dataclass(field_content)


def _replace_feature_with_value_in_dataclass(dataclass_instance) -> None:
    for field in dataclasses.fields(dataclass_instance):
        _replace_this_feature_with_value_recursively(dataclass_instance, field)


@dataclasses.dataclass
class ProtoUnitCell:
    """Defines an archetype of UnitCell (parameterized by Features)."""
    proto_unit_cell: UnitCell

    def __post_init__(self):
        self.features = self.proto_unit_cell.get_cell_unique_features()
        self.period = self.proto_unit_cell.periodicity[0]

    def generate_initial_variables(self, n_cells: int) -> tf.Tensor:
        tensor_columns = []
        clip_value_min = []
        clip_value_max = []
        for feature in self.proto_unit_cell.get_cell_unique_features():
            vmin = [feature.vmin]
            vmax = [feature.vmax]
            clip_value_min.append(vmin)
            clip_value_max.append(vmax)
            tensor_columns.append(tf.random.uniform([n_cells], vmin, vmax))

        tensor = tf.stack(tensor_columns, axis=0)
        constraint_func = lambda x: tf.clip_by_value(x, clip_value_min, clip_value_max)
        return tf.Variable(tensor, constraint=constraint_func)

    def generate_cells_from_parameter_tensor(self, tensor: tf.Tensor) -> list[UnitCell]:
        if tensor.shape[0] != len(self.features):
            raise ValueError("Tensor must have shape (n_features, n_unit_cells).")

        unit_cell_array = []
        for i in range(tensor.shape[-1]):
            unit_cell = copy.deepcopy(self.proto_unit_cell)
            features = unit_cell.get_cell_unique_features()
            parameters = tensor[:, i]
            for feature, init_value in zip(features, parameters):
                feature.set_value(init_value)
            unit_cell.replace_features()
            unit_cell_array.append(unit_cell)
        return unit_cell_array


def _rasterize_layer(
    layer: Layer,
    periodicity: tuple[ParameterType, ParameterType],
    pixel_density: float,
    enforce_4fold_symmetry: bool = False,
    wavelength: float | None = None,
) -> raster.Canvas:
    if isinstance(layer.material, Material):
        if wavelength is None:
            raise ValueError("Wavelength required for Material index.")
        layer_value = layer.material.index_at(wavelength)
    else:
        layer_value = layer.material

    return raster.Canvas(
        x_width=periodicity[0],
        y_width=periodicity[1],
        spacing=pixel_density,
        background_value=layer_value,
        enforce_4fold_symmetry=enforce_4fold_symmetry,
    ).rasterize(layer.get_shapes(wavelength))


def get_avaliable_materials(custom_csv_dir: str | None = None) -> list[str]:
    """Returns a list of available material strings."""
    if custom_csv_dir is None:
        custom_csv_dir = os.path.join(_ROOT, "material_data")

    avail_materials_dir = glob.glob(os.path.join(custom_csv_dir, "*.csv"))
    avali_materials = [
        os.path.split(file_path)[-1].split(".")[0]
        for file_path in avail_materials_dir
    ]
    return avali_materials


@dataclasses.dataclass
class Material:
    """Defines a material class."""
    name: str
    custom_csv_dir: str | None = None

    def __post_init__(self):
        if self.custom_csv_dir is None:
            self.custom_csv_dir = os.path.join(_ROOT, "material_data")

        csv_dir = os.path.join(self.custom_csv_dir, self.name + ".csv")
        if not os.path.exists(csv_dir):
            avaliable_materials = get_avaliable_materials(self.custom_csv_dir)
            default_mat_str = ", ".join(avaliable_materials)
            raise ValueError(
                f"CSV file for {self.name} not found in {self.custom_csv_dir}.\n"
                f"Available materials: {default_mat_str}"
            )

        self.wl_n = []
        self.wl_k = []
        self.n = []
        self.k = []

        with open(csv_dir, "r") as file:
            reader = csv.reader(file)
            mode = None
            for row in reader:
                if len(row) == 0 or row[0].strip() == "":
                    continue
                if row[1] == "n":
                    mode = "n"
                    continue
                elif row[1] == "k":
                    mode = "k"
                    continue
                if mode == "n":
                    self.wl_n.append(float(row[0]) * 1e-6)
                    self.n.append(float(row[1]))
                elif mode == "k":
                    self.wl_k.append(float(row[0]) * 1e-6)
                    self.k.append(float(row[1]))

        self.min_wl_n = min(self.wl_n)
        self.max_wl_n = max(self.wl_n)
        if len(self.wl_k) > 0:
            self.min_wl_k = min(self.wl_k)
            self.max_wl_k = max(self.wl_k)

        self.n_interp = interpolate.interp1d(self.wl_n, self.n)
        if len(self.wl_k) > 0:
            self.k_interp = interpolate.interp1d(self.wl_k, self.k)

    def index_at(self, wavelength):
        if not (self.min_wl_n <= wavelength <= self.max_wl_n):
            raise ValueError(f"Wavelength {wavelength} is out of range.")

        n_value = self.n_interp(wavelength)
        if hasattr(self, "k_interp") and self.k_interp is not None:
            k_value = self.k_interp(wavelength)
        else:
            k_value = 0j

        return n_value + 1.0j * k_value


@dataclasses.dataclass
class SimConfig:
    """Defines a simulation configuration."""
    xy_harmonics: tuple[int, int]
    resolution: int
    minibatch_size: int = 100
    return_tensor: bool = False
    return_zeroth_order: bool | None = None
    use_transmission: bool | None = None
    include_z_comp: bool | None = None

    def __post_init__(self):
        if self.xy_harmonics[0] % 2 != 1 or self.xy_harmonics[0] < 1:
            raise ValueError("xy_harmonics[0] must be a positive odd int.")
        elif self.xy_harmonics[1] % 2 != 1 or self.xy_harmonics[1] < 1:
            raise ValueError("xy_harmonics[1] must be a positive odd int.")

        if self.return_tensor:
            if self.return_zeroth_order is None:
                self.return_zeroth_order = True
            if self.use_transmission is None:
                self.use_transmission = True
            if self.include_z_comp is None:
                self.include_z_comp = False
        else:
            if (
                (self.return_zeroth_order is not None)
                or (self.use_transmission is not None)
                or (self.include_z_comp is not None)
            ):
                warnings.warn(
                    "When return_tensor is False, return_zeroth_order, "
                    "use_transmission, include_z_comp are ignored."
                )


@dataclasses.dataclass
class SimInstance:
    """Defines a simulation instance."""
    unit_cell_array: list[UnitCell]
    incidence: Incidence
    sim_config: SimConfig

    def __post_init__(self):
        x_periodicitys = [uc.periodicity[0] for uc in self.unit_cell_array]
        y_periodicitys = [uc.periodicity[1] for uc in self.unit_cell_array]
        if len(set(x_periodicitys)) != 1:
            raise ValueError("All x periods must be the same.")
        if len(set(y_periodicitys)) != 1:
            raise ValueError("All y periods must be the same.")

        refl_indices = [uc.refl_index for uc in self.unit_cell_array]
        tran_indices = [uc.tran_index for uc in self.unit_cell_array]
        if len(set(refl_indices)) != 1:
            raise ValueError("All reflection indices must be the same.")
        if len(set(tran_indices)) != 1:
            raise ValueError("All transmission indices must be the same.")

    def get_variables(self) -> list[tf.Variable]:
        variables = []
        for unit_cell in self.unit_cell_array:
            variables.extend(unit_cell.get_variables())
        return variables


@dataclasses.dataclass
class SimResult:
    """The result of an RCWA simulation."""
    rx: tf.Tensor
    ry: tf.Tensor
    rz: tf.Tensor
    r_eff: tf.Tensor
    r_power: tf.Tensor
    tx: tf.Tensor
    ty: tf.Tensor
    tz: tf.Tensor
    t_eff: tf.Tensor
    t_power: tf.Tensor
    xy_harmonics: tuple[int, int]

    @staticmethod
    def _get_0th(fields, xy_harmonics):
        return fields[:, :, 0, np.prod(xy_harmonics) // 2, 0]

    def ref_field(self, config: SimConfig) -> tf.Tensor:
        if config.return_zeroth_order:
            rx = self._get_0th(self.rx, self.xy_harmonics)
            ry = self._get_0th(self.ry, self.xy_harmonics)
            rz = self._get_0th(self.rz, self.xy_harmonics)
        else:
            rx, ry, rz = self.rx, self.ry, self.rz

        if config.include_z_comp:
            return tf.stack([rx, ry, rz], axis=-1)
        else:
            return tf.stack([rx, ry], axis=-1)

    def trn_field(self, config: SimConfig) -> tf.Tensor:
        if config.return_zeroth_order:
            tx = self._get_0th(self.tx, self.xy_harmonics)
            ty = self._get_0th(self.ty, self.xy_harmonics)
            tz = self._get_0th(self.tz, self.xy_harmonics)
        else:
            tx, ty, tz = self.tx, self.ty, self.tz

        if config.include_z_comp:
            return tf.stack([tx, ty, tz], axis=-1)
        else:
            return tf.stack([tx, ty], axis=-1)

    def get_result_using_config(self, config: SimConfig) -> SimResult | tf.Tensor:
        if not config.return_tensor:
            return self
        if config.use_transmission:
            return self.trn_field(config)
        else:
            return self.ref_field(config)


def minibatch_sim_instance(
    sim_instance: SimInstance, minibatch_size: int
) -> list[SimInstance]:
    """Generates a list of minibatch simulation instances."""
    unit_cell_array_chunks = [
        sim_instance.unit_cell_array[i : i + minibatch_size]
        for i in range(0, len(sim_instance.unit_cell_array), minibatch_size)
    ]
    sim_instance_array = []
    for unit_cell_array in unit_cell_array_chunks:
        sim_instance_array.append(
            SimInstance(
                unit_cell_array=unit_cell_array,
                incidence=sim_instance.incidence,
                sim_config=sim_instance.sim_config,
            )
        )
    return sim_instance_array


def combine_sim_results(sim_results: list[SimResult] | list[tf.Tensor]) -> SimResult | tf.Tensor:
    """Combines a list of simulation results into one."""
    if isinstance(sim_results[0], tf.Tensor):
        return tf.concat(sim_results, axis=1)

    rx = tf.concat([r.rx for r in sim_results], axis=1)
    ry = tf.concat([r.ry for r in sim_results], axis=1)
    rz = tf.concat([r.rz for r in sim_results], axis=1)
    r_eff = tf.concat([r.r_eff for r in sim_results], axis=1)
    r_power = tf.concat([r.r_power for r in sim_results], axis=1)
    tx = tf.concat([r.tx for r in sim_results], axis=1)
    ty = tf.concat([r.ty for r in sim_results], axis=1)
    tz = tf.concat([r.tz for r in sim_results], axis=1)
    t_eff = tf.concat([r.t_eff for r in sim_results], axis=1)
    t_power = tf.concat([r.t_power for r in sim_results], axis=1)
    xy_harmonics = sim_results[0].xy_harmonics
    return SimResult(
        rx=rx, ry=ry, rz=rz, r_eff=r_eff, r_power=r_power,
        tx=tx, ty=ty, tz=tz, t_eff=t_eff, t_power=t_power,
        xy_harmonics=xy_harmonics,
    )


def simulate_parameterized_unit_cells(
    parameter_tensor: tf.Tensor,
    proto_cell: ProtoUnitCell,
    incidence: Incidence,
    sim_config: SimConfig,
) -> tf.Tensor:
    """Simulate RCWA for parameterized unit cells."""
    minibatch_size = sim_config.minibatch_size
    simulate_func = simulate_parameterized_unit_cells_one_batch

    if minibatch_size < parameter_tensor.shape[1]:
        parameters_chunks = [
            parameter_tensor[:, i : i + minibatch_size]
            for i in range(0, parameter_tensor.shape[1], minibatch_size)
        ]
        sim_results = []
        for parameters_chunk in tqdm.tqdm(parameters_chunks):
            sim_results.append(
                simulate_func(
                    parameter_tensor=parameters_chunk,
                    proto_cell=proto_cell,
                    incidence=incidence,
                    sim_config=sim_config,
                )
            )
            gc.collect()
        return combine_sim_results(sim_results)

    return simulate_func(
        parameter_tensor=parameter_tensor,
        proto_cell=proto_cell,
        incidence=incidence,
        sim_config=sim_config,
    )


def simulate_parameterized_unit_cells_one_batch(
    parameter_tensor: tf.Tensor,
    proto_cell: ProtoUnitCell,
    incidence: Incidence,
    sim_config: SimConfig,
) -> tf.Tensor:
    if not sim_config.return_tensor:
        raise ValueError("SimConfig.return_tensor=True is required.")

    if len(proto_cell.features) == 0:
        raise ValueError("The proto cell has no features (not parameterized).")

    children = proto_cell.generate_cells_from_parameter_tensor(parameter_tensor)
    sim_instance = SimInstance(
        unit_cell_array=children,
        incidence=incidence,
        sim_config=sim_config,
    )
    return simulate_one(sim_instance)


def simulate(sim_instance: SimInstance) -> SimResult:
    """Simulates the periodic unit cell using RCWA."""
    minibatch_size = sim_instance.sim_config.minibatch_size
    if minibatch_size < len(sim_instance.unit_cell_array):
        sim_instances = minibatch_sim_instance(
            sim_instance=sim_instance, minibatch_size=minibatch_size
        )
        sim_results = simulate_batch(sim_instances=sim_instances)
        return combine_sim_results(sim_results=sim_results)
    else:
        return simulate_one(sim_instance=sim_instance)


def simulate_batch(sim_instances: list[SimInstance]) -> list[SimResult]:
    """Simulates a batch of periodic unit cells using RCWA."""
    return [simulate_one(sim_instance) for sim_instance in sim_instances]


def simulate_one(sim_instance: SimInstance) -> SimResult:
    """Simulates a single unit cell using RCWA."""
    incidence_dict = utils.unravel_incidence(sim_instance.incidence)

    batched_layer_thicknesses = []
    for unit_cell in sim_instance.unit_cell_array:
        layer_thicknesses = unit_cell.get_thickness()
        layer_thicknesses = tf.cast(layer_thicknesses, tf.complex64)
        layer_thicknesses = layer_thicknesses[
            tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis
        ]
        batched_layer_thicknesses.append(layer_thicknesses)
    layer_thicknesses = tf.concat(batched_layer_thicknesses, axis=1)

    er1 = sim_instance.unit_cell_array[0].refl_index ** 2
    er2 = sim_instance.unit_cell_array[0].tran_index ** 2

    n_cells = len(sim_instance.unit_cell_array)
    batch_size = len(incidence_dict["wavelength"])
    ER_t_array = []
    for unit_cell in sim_instance.unit_cell_array:
        ER_t_wl = []
        for wavelength in incidence_dict["wavelength"]:
            this_epsilon = unit_cell.get_epsilon(
                sim_instance.sim_config.resolution,
                wavelength=wavelength.numpy(),
            )
            this_epsilon = this_epsilon[tf.newaxis, tf.newaxis, tf.newaxis, :, :, :]
            ER_t_wl.append(this_epsilon)
        ER_t = tf.concat(ER_t_wl, axis=0)
        ER_t_array.append(ER_t)
    ER_t = tf.concat(ER_t_array, axis=1)
    UR_t = tf.ones_like(ER_t)

    refl_n = sim_instance.unit_cell_array[0].refl_index

    output = rcwa_tf.simulate_rcwa(
        incidence_dict,
        PQ=sim_instance.sim_config.xy_harmonics,
        n_cells=n_cells,
        n_layers=len(sim_instance.unit_cell_array[0].layers),
        layer_thicknesses=layer_thicknesses,
        L_xy=sim_instance.unit_cell_array[0].periodicity,
        er1=er1,
        er2=er2,
        ER_t=ER_t,
        UR_t=UR_t,
        refl_n=refl_n,
    )

    result = SimResult(
        rx=tf.math.conj(output["rx"]),
        ry=tf.math.conj(output["ry"]),
        rz=tf.math.conj(output["rz"]),
        r_eff=tf.math.conj(output["R"]),
        r_power=tf.math.conj(output["REF"]),
        tx=tf.math.conj(output["tx"]),
        ty=tf.math.conj(output["ty"]),
        tz=tf.math.conj(output["tz"]),
        t_eff=tf.math.conj(output["T"]),
        t_power=tf.math.conj(output["TRN"]),
        xy_harmonics=sim_instance.sim_config.xy_harmonics,
    )

    return result.get_result_using_config(sim_instance.sim_config)