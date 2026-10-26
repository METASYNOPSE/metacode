"""
Metrics module for metabox.

This module contains functions to calculate the metrics of the optical system.

Compatible with Python 3.11.9+
"""
from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from scipy import special
import sys 
from . import propagation


def get_ideal_mtf_volume(
    field_props: propagation.FieldProperties,
    focal_length: float,
) -> np.ndarray:
    """Calculate the volume under the ideal MTF surface.

    Args:
        field_props: the field properties.
        focal_length: the focal length of the lens in meters.

    Returns:
        np.ndarray: the volume under the ideal MTF surface.
    """
    lens_radius = (
        field_props.n_pixels * field_props.period / 2 / field_props.upsampling
    )
    mtf_norm_arr = []
    for wavelength in field_props.wavelength:
        sampling_radius = lens_radius
        x_max = (
            2
            * np.pi
            / wavelength
            * sampling_radius
            * np.sin(np.arctan(lens_radius / focal_length))
        )
        x = np.linspace(-x_max, x_max, field_props.n_pixels)
        y = np.linspace(-x_max, x_max, field_props.n_pixels)
        xv, yv = np.meshgrid(x, y)
        r = np.sqrt(xv**2 + yv**2)
        # Handle division by zero at r=0
        with np.errstate(divide='ignore', invalid='ignore'):
            intensity = 4 * (special.j1(r) / r) ** 2
            intensity = np.nan_to_num(intensity, nan=1.0)

        total_energy = np.sum(intensity)
        norm_intensity = intensity / total_energy

        mtf = np.abs(np.fft.fftshift(np.fft.fft2(norm_intensity)))
        mtf_max = np.amax(mtf)
        mtf_norm = mtf / mtf_max
        mtf_norm_arr.append(np.mean(mtf_norm))
    return np.array(mtf_norm_arr)


def get_max_intensity(field: propagation.Field2D) -> tf.Tensor:
    """Returns the maximum intensity of the field.

    Args:
        field: the field to calculate the maximum intensity of.

    Returns:
        tf.Tensor: The maximum intensity of the field.
    """
    tensor = field.tensor
    intensity = tf.math.abs(tensor) ** 2
    return tf.math.reduce_max(intensity, axis=[-1, -2])


def get_mtf_volume(
    field: propagation.Field2D, 
    use_log: bool = False
) -> tf.Tensor:
    """Returns the volume under the MTF surface.

    Args:
        field: the field to calculate the Strehl ratio of.
        use_log: use the log of the Strehl ratio for each sampling.

    Returns:
        tf.Tensor: the Strehl ratio of the field.
    """
    psf_tensor = field.get_intensity()
    norm_factor = tf.reduce_sum(psf_tensor, axis=[-2, -1], keepdims=True)
    norm_factor = tf.reshape(norm_factor, [len(field.wavelength), -1, 1, 1])
    norm_psf = psf_tensor / norm_factor
    norm_psf = tf.cast(norm_psf, dtype=tf.complex64)

    mtf = tf.abs(tf.signal.fftshift(tf.signal.fft2d(norm_psf)))
    mtf_max = tf.math.reduce_max(mtf, axis=[-1, -2])
    mtf_max = tf.reshape(mtf_max, [len(field.wavelength), -1, 1, 1])
    mtf_norm = mtf / mtf_max

    volume = tf.math.reduce_mean(mtf_norm, axis=[-1, -2])

    if use_log:
        volume = tf.math.log(volume)
        volume = tf.math.reduce_mean(volume)

    return volume


def get_center_intensity(
    field: propagation.Field2D, 
    use_log: bool = False
) -> tf.Tensor:
    """Returns the center intensity of the field.

    Args:
        field: the field to calculate the Strehl ratio of.
        use_log: use the log of the intensity for each sampling.

    Returns:
        tf.Tensor: the center intensity of the field.
    """
    psf_tensor = field.get_intensity()
    center_intensity = psf_tensor[:, :, field.n_pixels // 2, field.n_pixels // 2]
    return center_intensity