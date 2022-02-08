"""
Test file for the COMP1730/COMP6730 major assignment,
Semester 1, 2021.

Please write all tests you develop in this file, and submit it
along with assignment_template.py

Please include the student IDs of all members of your group here
Student Ids:
u6073656
u6514828
u7279920
u6392815
"""

import sys
import pytest

# This will import all the functions from assignment.py
from assignment import *

# Source: https://en.wikipedia.org/wiki/Cotter_Dam
COTTER_DAM_SURFACE_AREA = 2850000  # square metres

# First 4 columns of first 5 rows of small dataset (for testing Q1)
SAMPLE_DATA_1 = np.array(
    [[693.366, 692.038, 690.964, 690.964],
     [693.406, 692.079, 691.025, 691.025],
     [693.383, 692.039, 691.018, 691.018],
     [693.457, 692.085, 691.058, 691.058],
     [693.457, 692.107, 691.091, 691.091]])

# First 4 columns of first 5 rows of small dataset
ELEVATION_DATA_SMALL_SAMPLE_1 = np.genfromtxt(
    'elevation_data_small_sample_1.csv', delimiter=',')

# Sample of water (dam) elevations from small dataset
ELEVATION_DATA_SMALL_SAMPLE_2 = np.genfromtxt(
    'elevation_data_small_sample_2.csv', delimiter=','
)

# Sample of land (non-dam) elevations from small dataset
ELEVATION_DATA_SMALL_SAMPLE_3 = np.genfromtxt(
    'elevation_data_small_sample_3.csv', delimiter=','
)

# First 4 columns of first 5 rows of large dataset
ELEVATION_DATA_LARGE_SAMPLE_1 = np.genfromtxt(
    'elevation_data_large_sample_1.csv', delimiter=',')

# Small dataset (Cotter Dam)
ELEVATION_DATA_SMALL = np.genfromtxt(
    'elevation_data_small.csv', delimiter=',')


# Question 1 tests
def test_read_dataset_elevation_data():
    """Tests read_dataset() on CSV files containing two small samples of
    the data from elevation_data_small.csv and elevation_data_large.csv.
    """
    cases = (
        ('elevation_data_small_sample_1.csv',
         np.array([[693.366, 692.038, 690.964, 690.964],
                   [693.406, 692.079, 691.025, 691.025],
                   [693.383, 692.039, 691.018, 691.018],
                   [693.457, 692.085, 691.058, 691.058],
                   [693.457, 692.107, 691.091, 691.091]])),
        ('elevation_data_large_sample_1.csv',
         np.array([[942.562, 944.54, 947.045, 947.045],
                   [942.457, 944.438, 946.994, 946.994],
                   [942.268, 944.241, 946.849, 946.849],
                   [942.031, 944.011, 946.687, 946.687],
                   [941.733, 943.676, 946.369, 946.369]])),
    )

    for filepath, expected in cases:
        actual = read_dataset(filepath)
        assert np.array_equal(
            actual[:5, :4], expected), \
            'expected array {}, but read {}.'.format(expected, actual)


# Question 2 tests
def test_minimum_elevation_simple():
    """Tests minimum_elevation() on small arrays of made-up values.
    """
    cases = (
        (np.array([0]), 0),
        (np.array([1, 2, 3]), 1),
        (np.array([10., -7., 0., -6.2, 1.5, 4.9]), -7),
    )

    for data, expected in cases:
        actual = minimum_elevation(data)
        assert actual == expected, \
            'expected minimum elevation of {} metres, but got {} metres.'.format(expected, actual)


def test_minimum_elevation_sample_data():
    """Tests minimum_elevation() on a small sample of data from
    elevation_data_small.csv.
    """
    cases = (
        (ELEVATION_DATA_SMALL_SAMPLE_1, min(
            ELEVATION_DATA_SMALL_SAMPLE_1.flatten().tolist())),

        (ELEVATION_DATA_SMALL_SAMPLE_1, 690.964),

        (ELEVATION_DATA_LARGE_SAMPLE_1, min(
            ELEVATION_DATA_LARGE_SAMPLE_1.flatten().tolist())),

        (ELEVATION_DATA_LARGE_SAMPLE_1, 941.733),
    )

    for data, expected in cases:
        actual = minimum_elevation(data)
        assert actual == expected, \
            'expected minimum elevation of {} metres, but got {} metres.'.\
            format(expected, actual)


def test_maximum_elevation_simple():
    """Tests maximum_elevation on small arrays of made-up values.
    """
    cases = (
        (np.array([0]), 0),

        (np.array([1, 2, 3]), 3),

        (np.array([10., -7., 0., -6.2, 1.5, 4.9]), 10.),
    )

    for data, expected in cases:
        actual = maximum_elevation(data)
        assert actual == expected, \
            'expected maximum elevation of {} metres, but got {} metres.'. \
            format(expected, actual)


def test_maximum_elevation_sample_data():
    """Tests maximum_elevation() on a small sample of data from
    elevation_data_small.csv."""

    cases = (
        (ELEVATION_DATA_SMALL_SAMPLE_1,
         max(ELEVATION_DATA_SMALL_SAMPLE_1.flatten().tolist())),

        (ELEVATION_DATA_SMALL_SAMPLE_1, 693.457),

        (ELEVATION_DATA_LARGE_SAMPLE_1,
         max(ELEVATION_DATA_LARGE_SAMPLE_1.flatten().tolist())),

        (ELEVATION_DATA_LARGE_SAMPLE_1, 947.045),
    )

    for data, expected in cases:
        actual = maximum_elevation(data)
        assert actual == expected, \
            'expected maximum elevation of {} metres, but got {} metres.'.\
            format(expected, actual)


def test_average_elevation_simple():
    """Tests average_elevation() on small arrays of made-up values.
    """
    cases = (
        (np.array([0]), 0),
        (np.array([1, 2, 3]), 2),
        (np.array([10., -7., 0., -6.2, 1.5, 4.9]), 8 / 15),
    )

    for data, expected in cases:
        actual = average_elevation(data)
        assert actual == expected, \
            'expected average elevation of {} metres, but got {} metres.'.\
            format(expected, actual)


def test_average_elevation_sample_data():
    """Tests average_elevation() on a small sample of data from
    elevation_data_small.csv.
    """
    flattened_list_1 = ELEVATION_DATA_SMALL_SAMPLE_1.flatten().tolist()
    flattened_list_2 = ELEVATION_DATA_LARGE_SAMPLE_1.flatten().tolist()

    cases = (
        (ELEVATION_DATA_SMALL_SAMPLE_1, 691.886),

        (ELEVATION_DATA_SMALL_SAMPLE_1, sum(flattened_list_1) /
         len(flattened_list_1)),

        (ELEVATION_DATA_LARGE_SAMPLE_1, 944.9923),

        (ELEVATION_DATA_LARGE_SAMPLE_1, sum(flattened_list_2) /
         len(flattened_list_2)),
    )

    for data, expected in cases:
        actual = average_elevation(data)
        assert abs(actual - expected) <= 0.001, \
            'expected average elevation of {} metres, but got {} metres.'.\
            format(expected, actual)


# Question 3 tests
def test_slope_simple():
    """Tests slope() on small arrays with made-up elevation values.
    """
    cases = (
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), 1, 1, 0),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 1, 1, 0.632),
        (np.array([[6, 3, 0], [6, 3, 0], [6, 3, 0]]), 1, 1, 0.6),
    )

    for data, x, y, expected in cases:
        actual = slope(data, x, y)
        assert abs(actual - expected) <= 0.001, \
            'expected total gradient (slope) of {} metres, but got {} metres.'\
            .format(expected, actual)


def test_slope_sample_data_water():
    """Tests slope() on a CSV file containing a sample of water
    elevation points from the small data set.
    """
    cases = (
        (ELEVATION_DATA_SMALL_SAMPLE_2, 1, 1, 0.009),

        (ELEVATION_DATA_SMALL_SAMPLE_2, 2, 3, 0.011),
    )

    for data, x, y, expected in cases:
        actual = slope(data, x, y)
        assert abs(actual - expected) <= 0.001, \
            'expected total gradient (slope) of {} metres, but got {} metres.'\
            .format(expected, actual)


def test_slope_sample_data_land():
    """Tests slope() on a CSV file containing a sample of land elevation
    points from the small data set.
    """
    cases = (
        (ELEVATION_DATA_SMALL_SAMPLE_3, 1, 1, 0.209),

        (ELEVATION_DATA_SMALL_SAMPLE_3, 2, 3, 0.405),
    )

    for data, x, y, expected in cases:
        actual = slope(data, x, y)
        assert abs(actual - expected) <= 0.001, \
            'expected total gradient (slope) of {} metres, but got {} metres.'\
            .format(expected, actual)


# Question 4 tests
def test_surface_area_cotter_dam():
    """Test to verify that surface_area(), when called on the small data
    set, returns a value roughly equal (within 5%) to the Cotter Dam's
    actual surface area, 285 Ha (2,850,000 m^2), as found on Wikipedia.
    """

    cases = (
        (ELEVATION_DATA_SMALL, SMALL_DATA_SET_COTTER_DAM_POINT[0], SMALL_DATA_SET_COTTER_DAM_POINT[1],
         COTTER_DAM_SURFACE_AREA),
    )

    for data, x, y, expected in cases:
        actual = surface_area(data, x, y)
        # Check that the difference between the estimated surface area
        # and the surface area listed on the Cotter Dam Wikipedia page
        # (285 Ha) is not greater than 5%
        assert abs(actual / expected - 1) * 100 <= 5, \
            'expected {}, but got {}.'.format(expected, actual)


# Question 5 tests
def test_expanded_surface_area_simple():
    """A trivial test to check that expanded_surface_area() can raise
    the water level of a small elevation grid of nine cells and
    calculate the new surface area, where the cell at coordinates (1, 1)
    of the small grid represents a dam and the remaining cells represent
    land. The grid resolution is assumed to be 5 metres.
    """

    cases = (
        # Test raising the water level from 550 metres to 700 metres, so
        # that the surrounding cells are flooded with water, thereby
        # raising their elevation to 700 metres
        (np.array(
            [[600, 600, 600],
             [600, 550, 600],
             [600, 600, 600]]),
         1, 1, 700, 9 * 25),
    )

    for data, x, y, water_level, expected in cases:
        actual = expanded_surface_area(data, water_level, x, y)
        assert abs(actual - expected) <= 0.001, \
            'expected an expanded surface area of {} square metres ' \
            'for a water level of {} metres at coordinates ({}, {}), ' \
            'but got {} square metres.'. \
            format(expected, water_level, x, y, actual)


# Question 6 tests
def test_impute_missing_values_simple():
    """This simple test should show that impute_missing_values() is
    able to detect and replace an extreme outlier value with an
    appropriate interpolation value (in this case, it should replace the
    outlier with the cell's original value, 600.0, since its surrounding
    cells all have a value of 600.0).
    """
    data_without_outlier = np.full((1000, 1000), 600.0)
    data_with_outlier = np.copy(data_without_outlier)
    data_with_outlier[0, 0] = OUTLIERS_VALUE

    cases = (
        (data_with_outlier,
         data_without_outlier),
    )

    for data, expected in cases:
        actual = impute_missing_values(data)
        assert np.array_equal(actual, expected), \
            'expected {}, but got {}.'.format(expected, actual)


if __name__ == '__main__':
    pytest.main(sys.argv)
