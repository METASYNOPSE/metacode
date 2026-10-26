"""
Utility functions and classes for metabox.

Compatible with Python 3.11.9+
"""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from typing import Any

import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys 

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def recursively_convert_ndarray_in_dict_to_list(item: Any) -> Any:
    """Recursively converts ndarray item in dict to list"""
    if isinstance(item, np.ndarray):
        item = item.tolist()
    elif isinstance(item, dict):
        for key, value in item.items():
            item[key] = recursively_convert_ndarray_in_dict_to_list(value)
    return item


@dataclasses.dataclass
class Feature:
    """Defines a feature variable.

    Args:
        vmin: the minimum value of the feature.
        vmax: the maximum value of the feature.
        name: the name of the feature.
        sampling: the number of samples to take between vmin and vmax. If None,
            the sampling is undefined.
        initial_value: the initial value of the feature.
        value: the current value of the feature.
    """

    vmin: float
    vmax: float
    name: str
    initial_value: float | None = None
    sampling: int | None = None
    value: tf.Variable | None | float = None

    def __post_init__(self):
        if ":" in self.name:
            raise ValueError(
                "The name of the feature cannot contain the character ':'"
            )

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def initialize_value(self) -> None:
        """Initializes the variables of the feature."""
        if self.initial_value is not None:
            tensor = tf.math.real(self.initial_value)
            tensor = tf.cast([tensor], tf.float32)
        else:
            tensor = tf.random.uniform([1], self.vmin, self.vmax, tf.float32)
        self.value = tensor

    def set_variable(self) -> None:
        """Convert self.value to a variable"""
        constraint_func = lambda x: tf.clip_by_value(x, self.vmin, self.vmax)
        self.value = tf.Variable(
            self.value, constraint=constraint_func, name=self.name
        )

    def set_value(self, value: Any) -> None:
        """Set the value of the feature to value"""
        self.value = value


@dataclasses.dataclass
class Incidence:
    """Defines the physical properties of the incident light.

    Args:
        wavelength: the wavelengths of the light in meters.
        theta: tuple of the angles of incidence in degrees on the xz plane.
            Defaults to (0,).
        phi: tuple of the angles of incidence in degrees on the yz plane.
            Defaults to (0,).
        jones_vector: the Jones vector of the incident light.
            Defaults to (1, 0) which corresponds to a linearly polarized
            light with the electric field vector parallel to the x axis.
    """

    wavelength: tuple[float, ...]
    theta: tuple[float, ...] = (0,)
    phi: tuple[float, ...] = (0,)
    jones_vector: tuple[float, float] = (1, 0)


def unravel_wavelength_theta_phi(
    wavelength: list[float], theta: list[float], phi: list[float]
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Unravels the wavelength, theta, and phi lists into tensors.

    Args:
        wavelength: a list of wavelengths in meters.
        theta: a list of angles of incidence in degrees on the xz plane.
        phi: a list of angles of incidence in degrees on the yz plane.

    Returns:
        A tuple of tensors of shape (batch_size,).
    """
    wavelength_base = wavelength
    theta_base = theta
    phi_base = phi

    wavelength_out = tf.convert_to_tensor(
        np.repeat(wavelength_base, np.size(theta_base) * np.size(phi_base)),
        dtype=tf.float32,
    )

    theta_out = (
        np.pi
        / 180.0
        * tf.convert_to_tensor(
            np.tile(theta_base, np.size(wavelength_base) * np.size(phi_base)),
            dtype=tf.float32,
        )
    )

    phi_vals = np.repeat(phi_base, np.size(theta_base))
    phi_vals = np.tile(phi_vals, np.size(wavelength_base))
    phi_out = np.pi / 180.0 * tf.convert_to_tensor(phi_vals, dtype=tf.float32)

    return wavelength_out, theta_out, phi_out


def unravel_incidence(incidence: Incidence) -> dict[str, Any]:
    """Serializes an incidence data into lists."""
    (
        wavelength_batch,
        theta_batch,
        phi_batch,
    ) = unravel_wavelength_theta_phi(
        wavelength=list(incidence.wavelength),
        theta=list(incidence.theta),
        phi=list(incidence.phi),
    )
    x_pol, y_pol = incidence.jones_vector
    x_pol_batch = tf.cast(
        tf.repeat(x_pol, len(wavelength_batch)), tf.complex64
    )
    y_pol_batch = tf.cast(
        tf.repeat(y_pol, len(wavelength_batch)), tf.complex64
    )

    return {
        "wavelength": wavelength_batch,
        "theta": theta_batch,
        "phi": phi_batch,
        "ptm": x_pol_batch,
        "pte": y_pol_batch,
    }


# Type aliases using Python 3.11+ syntax
ParameterType = Feature | float | tf.Tensor
CoordType = tuple[ParameterType, ParameterType]

TF_FUNCTIONS = [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "sin",
    "sinh",
    "tan",
    "tanh",
    "exp",
    "sqrt",
    "square",
    "reduce_sum",
    "reduce_mean",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "matmul",
    "transpose",
    "reshape",
    "expand_dims",
    "squeeze",
    "stack",
    "concat",
]


def wavelength_to_rgb(wavelength: float) -> tuple[float, float, float]:
    """
    Convert a wavelength in the visible spectrum to RGB.
    
    Args:
        wavelength: wavelength in meters
        
    Returns:
        RGB tuple with values between 0 and 1
    """
    wavelength_nm = wavelength * 1e9
    gamma = 0.8

    if 380 <= wavelength_nm < 440:
        attenuation = 0.3 + 0.7 * (wavelength_nm - 380) / (440 - 380)
        R = ((-(wavelength_nm - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif 440 <= wavelength_nm < 490:
        R = 0.0
        G = ((wavelength_nm - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif 490 <= wavelength_nm < 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength_nm - 510) / (510 - 490)) ** gamma
    elif 510 <= wavelength_nm < 580:
        R = ((wavelength_nm - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif 580 <= wavelength_nm < 645:
        R = 1.0
        G = (-(wavelength_nm - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif 645 <= wavelength_nm <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength_nm) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    return (R, G, B)