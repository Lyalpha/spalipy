# spalipy - Detection-based astrononmical image registration
# Copyright (C) 2018-2021  Joe Lyman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

import unittest

import numpy as np
from astropy.modeling import models
from astropy.table import Table

from spalipy import Spalipy

SHAPE = (380, 400)  # size of images used for testing
BUFFER = 50  # buffer for creating sources outside shape, to allow for transformations


def generate_image(translate=(0, 0), rotate=0.0, scale=1.0, num_dets=180, seed=0):
    """Create an array of a simulated astronomical image."""
    rng = np.random.default_rng(seed)
    # Define a fixed seed rng - source positions and relative fluxes must make sense
    rng_source = np.random.default_rng(42)

    # Generate a noise background from a combination of poisson+gaussian
    poisson_mean = 25
    poisson_noise = rng.poisson(lam=poisson_mean, size=SHAPE)

    gauss_sigma = 5
    gaussian_noise = rng.normal(loc=0, scale=gauss_sigma, size=SHAPE)

    # Define parameters for gaussian sources
    amplitude_low = poisson_mean + 100 * gauss_sigma
    amplitude_high = poisson_mean + 200 * gauss_sigma
    amplitude = rng_source.uniform(low=amplitude_low, high=amplitude_high, size=num_dets)

    x_mean = rng_source.uniform(-BUFFER, SHAPE[0] + BUFFER, size=num_dets)
    y_mean = rng_source.uniform(-BUFFER, SHAPE[1] + BUFFER, size=num_dets)

    stddev_low = 2
    stddev_scale = 0.1
    x_stddev = (rng_source.rayleigh(scale=stddev_scale, size=num_dets) + stddev_low) * scale
    y_stddev = (rng_source.rayleigh(scale=stddev_scale, size=num_dets) + stddev_low) * scale

    theta = np.radians(rng_source.uniform(low=0, high=360, size=num_dets))

    # Construct models gaussian sources, accounting for translation, rotation and scale
    if translate != (0, 0) or rotate != 0 or scale != 1:
        cosrot = np.cos(np.radians(rotate))
        sinrot = np.sin(np.radians(rotate))
        rottrans_mat = np.array(
            [[cosrot, -sinrot, translate[0]], [sinrot, cosrot, translate[1]], [0, 0, 1]]
        )
        scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        rottransscale_mat = scale_mat @ rottrans_mat

        x_mid = SHAPE[0] / 2  # subtract the midpoint of image so rotation
        y_mid = SHAPE[1] / 2  # and scaling is with respect to centre of image
        xy = np.column_stack((x_mean - x_mid, y_mean - y_mid, np.ones_like(x_mean)))
        new_xy = (rottransscale_mat @ xy.T).T
        x_mean = new_xy[:, 0] + x_mid
        y_mean = new_xy[:, 1] + y_mid

    det_models = models.Gaussian2D(
        amplitude=amplitude,
        x_mean=x_mean,
        y_mean=y_mean,
        x_stddev=x_stddev,
        y_stddev=y_stddev,
        theta=theta,
        n_models=num_dets,
    )

    # Generate and image of the sources
    xrange = np.arange(SHAPE[1])
    yrange = np.arange(SHAPE[0])
    x, y = np.meshgrid(xrange, yrange)
    x = np.repeat(x[:, :, np.newaxis], num_dets, axis=2)
    y = np.repeat(y[:, :, np.newaxis], num_dets, axis=2)
    det_signal = det_models(x, y, model_set_axis=2).sum(axis=2)

    det_table = Table(
        dict(x=x_mean, y=y_mean, flux=amplitude, fwhm=x_stddev * 2.355, flag=[0] * num_dets)
    )

    # Create and return the final noise + signal image and a table of detections
    return poisson_noise + gaussian_noise + det_signal, det_table


def generate_mask(bits=4, num_masked=500, seed=0):
    """Create a random mask that includes multiple bit values"""
    rng = np.random.default_rng(seed)
    x_mask = rng.integers(low=0, high=SHAPE[0], size=num_masked)
    y_mask = rng.integers(low=0, high=SHAPE[1], size=num_masked)

    mask = np.zeros(SHAPE, int)
    mask[x_mask, y_mask] = 1

    for i in range(1, bits):
        x_mask_bit = rng.choice(x_mask, size=num_masked // bits, replace=False)
        y_mask_bit = rng.choice(y_mask, size=num_masked // bits, replace=False)
        mask[x_mask_bit, y_mask_bit] |= 1 << i

    return mask


class TestSpalipy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_data, cls.template_dets = generate_image(seed=0)
        cls.source_data, cls.source_dets = generate_image(
            translate=(20, -20), rotate=60, scale=1.2, seed=1
        )
        cls.source_data_footprint, cls.source_dets_footprint = generate_image(
            translate=(3.4, -12.5), rotate=49.2, scale=1.0, seed=1
        )
        source_mask = generate_mask()
        source_mask[100:105, 330:335] = 16
        cls.source_mask = source_mask

        cls.expected_affine_transform_simple = np.array(
            [0.41713655, -0.72205574, -26.36528175, 281.10807233]
        )
        cls.expected_affine_transform_footprint = np.array(
            [0.65375097, -0.7569395, -0.35889457, 301.85752249]
        )
        cls.expected_affine_transform_quad_edge_buffer = np.array([])

        cls.expected_footprint_shape = (549, 547)

    def test_simple_align(self):
        """Test simple alignment produces expected affine transformation."""
        sp = Spalipy(
            self.source_data,
            template_data=self.template_data,
            min_n_match=10,
            sub_tile=1,
            spline_order=0,
        )
        sp.align()
        assert sp.aligned_data.shape == SHAPE
        assert np.allclose(sp.affine_transform.v, self.expected_affine_transform_simple)

    def test_spline_align(self):
        """Test that spline alignment is performed correctly"""
        sp = Spalipy(
            self.source_data,
            template_data=self.template_data,
            min_n_match=10,
            sub_tile=1,
            spline_order=1,
        )
        sp.align()
        assert sp.aligned_data.shape == SHAPE
        # Inclusion of a spline order should not affect the initial affine transform
        assert np.allclose(sp.affine_transform.v, self.expected_affine_transform_simple)
        # TODO more checking of spline alignment

    def test_mask_align(self):
        """Test inclusion of mask does not affect alignment and produces
        expected mask, distilled to a sum."""
        assert np.sum(self.source_mask) == 2646
        sp = Spalipy(
            self.source_data,
            source_mask=self.source_mask,
            template_data=self.template_data,
            min_n_match=10,
            sub_tile=1,
            spline_order=0,
        )
        sp.align()
        assert np.allclose(sp.affine_transform.v, self.expected_affine_transform_simple)
        assert np.sum(sp.aligned_mask) == 1839

    def test_multi_simple_align(self):
        """Test passing multiple arrays to align produces expected affine transformations"""
        sp = Spalipy(
            [self.source_data, self.source_data, self.source_data],
            source_mask=[self.source_mask, self.source_mask, self.source_mask],
            template_data=self.template_data,
            min_n_match=10,
            sub_tile=1,
            spline_order=0,
        )
        sp.align()
        assert isinstance(sp.aligned_data, list)
        for affine_transform in sp.affine_transform:
            assert np.allclose(affine_transform.v, self.expected_affine_transform_simple)

    def test_quad_edge_buffer(self):
        sp = Spalipy(
            self.source_data,
            template_data=self.template_data,
            min_n_match=10,
            sub_tile=1,
            spline_order=0,
            quad_edge_buffer=max(SHAPE),
        )
        with self.assertRaises(ValueError):
            sp.make_source_quadlist()
        with self.assertRaises(ValueError):
            sp.make_template_quadlist()
        sp = Spalipy(
            self.source_data,
            template_data=self.template_data,
            min_n_match=10,
            sub_tile=1,
            spline_order=0,
            quad_edge_buffer=100,
        )
        sp.align()
        assert len(sp.source_quadlist[0]) == 486
        assert len(sp.template_quadlist[0]) == 2
        assert np.allclose(sp.affine_transform.v, self.expected_affine_transform_simple)

    def test_preserve_footprints(self):
        sp = Spalipy(
            self.source_data_footprint,
            template_data=self.template_data,
            min_n_match=10,
            sub_tile=1,
            spline_order=0,
            preserve_footprints=True,
        )
        sp.align()
        assert np.allclose(sp.affine_transform.v, self.expected_affine_transform_footprint)
        assert sp.template_data.shape == self.expected_footprint_shape
        assert sp.aligned_data.shape == self.expected_footprint_shape
        # preserve_footprints should raise error when not providing template data
        with self.assertRaises(ValueError):
            sp = Spalipy(
                self.source_data_footprint,
                template_det=sp.template_det,
                min_n_match=10,
                sub_tile=1,
                spline_order=0,
                preserve_footprints=True,
            )
