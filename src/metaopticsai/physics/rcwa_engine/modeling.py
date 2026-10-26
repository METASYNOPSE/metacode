"""
Modeling module for metabox.

This module provides classes and functions for creating and training metamodels.

Compatible with Python 3.11.9+
"""
from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import os
import pickle
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm.keras import TqdmCallback
import sys

from . import rcwa 
from . import utils


@dataclasses.dataclass
class SimulationLibrary:
    """Stores the simulation parameters and the simulation output.

    Attributes:
        protocell: the simulated protocell.
        incidence: the incidence of the light.
        sim_config: the simulation configuration.
        feature_values: the sampled feature values.
        simulation_output: the simulation output.
    """
    protocell: rcwa.ProtoUnitCell
    incidence: rcwa.Incidence
    sim_config: dict
    feature_values: np.ndarray
    simulation_output: tf.Tensor

    def get_training_x(self) -> np.ndarray:
        """Returns the training input."""
        if len(self.incidence.wavelength) != self.simulation_output.shape[0]:
            raise ValueError(
                "Number of wavelengths in simulation output doesn't match incidence."
            )

        feature_values = self.feature_values
        wavelength_values = self.incidence.wavelength
        output_values = self.simulation_output

        n_wavelengths, n_instances, _ = output_values.shape
        n_features = feature_values.shape[0]

        wavelengths = tf.cast(wavelength_values, tf.float32)[:, tf.newaxis]
        wavelengths = tf.tile(wavelengths, [1, n_instances])
        wavelengths = wavelengths[tf.newaxis, ...]
        expanded_features = feature_values[:, tf.newaxis, ...]
        expanded_features = tf.cast(expanded_features, tf.float32)
        expanded_features = tf.tile(expanded_features, [1, n_wavelengths, 1])
        expanded_features = tf.concat([wavelengths, expanded_features], axis=0)
        expanded_features = tf.reshape(expanded_features, [n_features + 1, -1])
        return expanded_features.numpy().T

    def get_training_y(self) -> np.ndarray:
        """Returns the training output."""
        output_values = self.simulation_output
        if hasattr(output_values, "numpy"):
            output_values = output_values.numpy()
            
        return np.asarray(output_values).reshape(-1, 2)

    def save(self, name: str, path: str, overwrite: bool = False):
        """Saves the simulation library."""
        save_simulation_library(self, name=name, path=path, overwrite=overwrite)


def sample_protocell(
    protocell: rcwa.ProtoUnitCell,
    incidence: rcwa.Incidence,
    sim_config: rcwa.SimConfig,
) -> SimulationLibrary:
    """Sample a protocell with a given incidence.

    Args:
        protocell: The protocell to sample.
        incidence: The incidence to sample the protocell with.
        sim_config: The simulation configuration.

    Returns:
        SimulationLibrary: The simulation library.
    """
    features = protocell.features
    sampling_values_per_feature = []
    for feature in features:
        if feature.sampling is None:
            raise ValueError(f"Feature {feature.name} has no sampling value.")
        uniform_sampling = np.linspace(feature.vmin, feature.vmax, feature.sampling)
        sampling_values_per_feature.append(uniform_sampling)
    
    feature_values = list(itertools.product(*sampling_values_per_feature))
    feature_values = np.array(feature_values).T

    output = rcwa.simulate_parameterized_unit_cells(
        feature_values,
        protocell,
        incidence,
        sim_config,
    ).numpy()

    return SimulationLibrary(
        protocell=protocell,
        incidence=incidence,
        sim_config=sim_config,
        feature_values=feature_values,
        simulation_output=output,
    )


@tf.keras.utils.register_keras_serializable(package="modeling")
class NormComplexLayer(tf.keras.layers.Layer):
    """A layer that converts two features into a normalized complex feature column."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        a, b = tf.split(inputs, 2, axis=-1)
        norm = tf.sqrt(a**2 + b**2)
        norm = tf.clip_by_value(norm, 1, np.inf)
        a = tf.cast(a / norm, tf.complex64)
        b = tf.cast(b / norm, tf.complex64)
        return a + b * 1j


@tf.keras.utils.register_keras_serializable(package="modeling")
class ComplexLayer(tf.keras.layers.Layer):
    """A layer that converts two features into a complex feature column."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        a, b = tf.split(inputs, 2, axis=-1)
        a = tf.cast(a, tf.complex64)
        b = tf.cast(b, tf.complex64)
        return a + b * 1j


@tf.keras.utils.register_keras_serializable(package="modeling")
def euclidian_distance(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Calculates the euclidian distance between two complex numbers."""
    return tf.abs(y_true - y_pred) ** 2


def create_fcc_model(
    normalizer: tf.keras.layers.Normalization,
    optimizer: tf.keras.optimizers.Optimizer,
    hidden_layer_units_list: list[int],
    activation_list: list[str],
    limit_output_to_unity: bool = False,
) -> tf.keras.Sequential:
    """Creates a simple fully connected network with a normalization layer."""
    layers = [normalizer]
    for n_units, activation in zip(hidden_layer_units_list, activation_list):
        layers.append(tf.keras.layers.Dense(n_units, activation=activation))
    layers.append(tf.keras.layers.Dense(4))
    if limit_output_to_unity:
        layers.append(NormComplexLayer())
    else:
        layers.append(ComplexLayer())

    model = tf.keras.Sequential(layers)
    optimizer.build(model.trainable_variables)
    model.compile(loss=euclidian_distance, optimizer=optimizer)
    return model


@dataclasses.dataclass
class Metamodel:
    """Stores a trained metamodel."""
    model: tf.keras.Sequential
    history: tf.keras.callbacks.History
    protocell: rcwa.ProtoUnitCell
    sim_config: dict

    def save(
        self,
        name: str,
        path: str = "./saved_metamodels",
        overwrite: bool = False,
    ) -> None:
        """Saves the metamodel to a file."""
        new_folder = os.path.join(path, name)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        model_path = os.path.join(new_folder, "tf_model")
        pkl_path = os.path.join(new_folder, name + ".pkl")
        json_path = os.path.join(new_folder, "info.json")
        self.model.save(model_path, overwrite=overwrite)

        if not overwrite:
            if os.path.exists(pkl_path):
                raise ValueError("File already exists.")

        with utils.suppress_stdout_stderr():
            new_self = copy.deepcopy(self)
            del new_self.model
            with open(pkl_path, "wb") as filehandler_pkl:
                pickle.dump(new_self, filehandler_pkl)

        json_dict = dataclasses.asdict(self.protocell)
        for item in json_dict:
            if np.iscomplex(json_dict[item]):
                json_dict[item] = str(json_dict[item])
        json_d = json.dumps(json_dict, indent=4)
        with open(json_path, "w") as outfile:
            outfile.write(json_d)
        print("Saved metamodel to " + new_folder)

    def plot_training_history(self) -> None:
        """Plots the training history."""
        import matplotlib.pyplot as plt

        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

    def set_feature_constraint(
        self,
        feature_str: str,
        vmin: float | None,
        vmax: float | None,
    ) -> None:
        """Sets the constraint of a feature."""
        for feature in self.protocell.features:
            if feature_str == feature.name:
                break

        current_vmin = feature.vmin
        current_vmax = feature.vmax

        if vmin is None:
            vmin = current_vmin
        if vmax is None:
            vmax = current_vmax
        if vmin < current_vmin:
            raise ValueError("Minimum constraint must be >= minimum feature value.")
        if vmax > current_vmax:
            raise ValueError("Maximum constraint must be <= maximum feature value.")
        setattr(feature, "vmin", vmin)
        setattr(feature, "vmax", vmax)


def load_metamodel(name: str, save_dir: str = "./saved_metamodels") -> Metamodel:
    """Loads a metamodel from a file."""
    new_folder = os.path.join(save_dir, name)
    model_path = os.path.join(new_folder, "tf_model")
    pkl_path = os.path.join(new_folder, name + ".pkl")
    if not os.path.exists(pkl_path):
        raise ValueError("File does not exist.")
    with open(pkl_path, "rb") as filehandler:
        with utils.suppress_stdout_stderr():
            my_model = pickle.load(filehandler)
            my_model.model = tf.keras.models.load_model(model_path)
    return my_model


def create_and_train_model(
    sim_lib: SimulationLibrary,
    n_epochs: int = 100,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
    hidden_layer_units_list: list[int] | None = None,
    activation_list: list[str] | None = None,
    limit_output_to_unity: bool = False,
    train_batch_size: int | None = None,
    validation_split: float = 0.05,
    verbose: int = 0,
) -> Metamodel:
    """Creates and fits a given model to the atom library."""
    if hidden_layer_units_list is None:
        hidden_layer_units_list = [64, 128, 256, 64]
    if activation_list is None:
        activation_list = ["relu", "relu", "relu", "relu"]

    if len(hidden_layer_units_list) != len(activation_list):
        raise ValueError(
            "Number of hidden layers must equal number of activation functions."
        )
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()

    train_input = sim_lib.get_training_x()
    train_output = sim_lib.get_training_y()

    normalizer = tf.keras.layers.Normalization(
        axis=-1, input_dim=train_input.shape[-1]
    )
    normalizer.adapt(train_input)

    model = create_fcc_model(
        normalizer=normalizer,
        optimizer=optimizer,
        hidden_layer_units_list=hidden_layer_units_list,
        activation_list=activation_list,
        limit_output_to_unity=limit_output_to_unity,
    )

    history = model.fit(
        train_input.astype(np.float32),
        train_output.astype(np.complex64),
        validation_split=validation_split,
        verbose=verbose,
        epochs=n_epochs,
        batch_size=train_batch_size,
        callbacks=[TqdmCallback(verbose=1)],
    )

    return Metamodel(
        model=model,
        history=history,
        protocell=sim_lib.protocell,
        sim_config=sim_lib.sim_config,
    )


def save_simulation_library(
    sim_lib: SimulationLibrary,
    name: str,
    path: str,
    overwrite: bool = False,
) -> None:
    """Saves the simulation library to a file."""
    if not os.path.isdir(path):
        raise ValueError("path must be a directory.")
    if not os.path.exists(path):
        os.makedirs(path)

    full_path = os.path.join(path, name + ".pkl")
    if not overwrite:
        if os.path.exists(full_path):
            raise ValueError("File already exists.")

    with open(full_path, "wb") as filehandler_pkl:
        pickle.dump(sim_lib, filehandler_pkl)
    
    json_dict = dataclasses.asdict(sim_lib)
    for item in json_dict:
        json_dict[item] = utils.recursively_convert_ndarray_in_dict_to_list(item)
        if np.any(np.iscomplex(json_dict[item])):
            json_dict[item] = str(json_dict[item])
    json_d = json.dumps(json_dict, indent=4)
    with open(full_path + ".json", "w") as outfile:
        outfile.write(json_d)

    print("Saved the atom library to " + full_path)


def load_simulation_library(name: str, path: str) -> SimulationLibrary:
    """Loads a SimulationLibrary from a file."""
    full_path = os.path.join(path, name + ".pkl")
    if not os.path.exists(full_path):
        raise ValueError("File does not exist.")
    with open(full_path, "rb") as filehandler:
        return pickle.load(filehandler)