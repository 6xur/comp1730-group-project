"""
This is the assignment template for the COMP1730/COMP6730 major assignment
for Semester 1, 2021.

The assignment is due at 9:00am on Monday 24 May.

Please include the student IDs of all members of your group here
Student Ids:
u6073656
u6514828
u7279920
u6392815
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import griddata
from scipy import stats
import pandas as pd

# Constants
GRID_SIZE = 5  # metres
SLOPE_TOLERANCE = 0.065  # metres
ELEVATION_TOLERANCE = 2.0  # metres
OUTLIERS_VALUE = -3.403e38  # given in specification; mainly for debugging
COTTER_DAM_WALL_ELEVATION = 550.800  # metres above sea level
SMALL_DATA_SET_COTTER_DAM_POINT = (794, 234)  # (x * 5 metres, y * 5 metres) - lies on Cotter Dam
LARGE_DATA_SET_COTTER_DAM_POINT = (2878, 242)  # lies on Cotter Dam
LARGE_DATA_SET_BENDORA_DAM_POINT = (634, 3313)  # Bendora Dam
LARGE_DATA_SET_CORIN_DAM_POINT = (758, 5485)  # Corin Dam


# Question 1:
def read_dataset(file):
    """Takes the file path of a CSV file as input, reads the file, and
    returns the data as a numpy array.

    Parameters
    ----------
    file : str
        The filename of the data set.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the data set.

    """
    return np.genfromtxt(file, delimiter=',')


# Question 2:
def minimum_elevation(data_set):
    """Returns the minimum elevation of the region covered by the data
    set.

    Parameters
    ----------
    data_set : numpy.ndarray
        A numpy.ndarray object containing elevation data in metres.

    Returns
    -------
    float
        The minimum elevation in the data set in metres.
    """
    return data_set.min()


def maximum_elevation(data_set):
    """Returns the maximum elevation of the region covered by the data
    set.

    Parameters
    ----------
    data_set : numpy.ndarray
        A numpy.ndarray object containing elevation data in metres.

    Returns
    -------
    float
        The maximum elevation in the data set in metres.
    """
    return data_set.max()


def average_elevation(data_set):
    """Returns the average (mean) elevation of the region covered by the
    data set.

    Parameters
    ----------
    data_set : numpy.ndarray
        A numpy array containing elevation data in metres.

    Returns
    -------
    float
        The average (mean) elevation in the data set in metres.
    """
    return data_set.mean()


# Question 3
def slope(data_set, x_coordinate, y_coordinate):
    """Takes as inputs the elevation data set, an x coordinate and a
    y coordinate and returns the total gradient (slope) at the
    corresponding cell.

    Parameters
    ----------
    data_set : numpy.ndarray
        A numpy array containing elevation data in metres.
    x_coordinate : int
        The x (horizontal) coordinate of the point at which to calculate
        the slope.
    y_coordinate : int
        The y (vertical) coordinate of the point at which to calculate
        the slope.

    Returns
    -------
    float
        The total gradient (slope) at the given coordinates in metres.
    """
    cell_left, cell_right, cell_above, cell_below = \
        cell_neighbours(data_set, x_coordinate, y_coordinate)
    # cell_neighbours_gradient(data_set, x_coordinate, y_coordinate)  # alternative function

    dx = x_gradient(cell_left, cell_right)
    dy = y_gradient(cell_below, cell_above)

    return math.sqrt(dx ** 2 + dy ** 2)


def cell_neighbours(data_set, x_coordinate, y_coordinate):
    """Returns the given cell's neighbouring cells. If the given cell
    lies at one or more edges of the data set, then the given cell is
    returned for those edges.

    Parameters
    ----------
    data_set : numpy.ndarray
        A numpy array containing elevation data in metres.
    x_coordinate : int
        The x (horizontal) coordinate of the given cell.
    y_coordinate : int
        The y (vertical) coordinate of the given cell.

    Returns
    -------
    tuple
        A tuple containing the given cell's four neighbours, in the
        order: (cell_left, cell_right, cell_above, cell_below).
    """
    cell_centre = data_set[y_coordinate, x_coordinate]  # given cell

    if x_coordinate == 0:  # left-most column
        cell_left = cell_centre
    else:
        cell_left = data_set[y_coordinate, x_coordinate - 1]

    if x_coordinate == data_set.shape[1] - 1:  # right-most column
        cell_right = cell_centre
    else:
        cell_right = data_set[y_coordinate, x_coordinate + 1]

    if y_coordinate == 0:  # top-most row
        cell_above = cell_centre
    else:
        cell_above = data_set[y_coordinate - 1, x_coordinate]

    if y_coordinate == data_set.shape[0] - 1:  # bottom-most row
        cell_below = cell_centre
    else:
        cell_below = data_set[y_coordinate + 1, x_coordinate]

    return cell_left, cell_right, cell_above, cell_below


def cell_neighbours_gradient(data_set, x_coordinate, y_coordinate):
    """Alternative to `cell_neighbours()` which returns the given cell's
    neighbouring cells. If the given cell lies at one or more edges of
    the data set, then the gradient leading up to the given edge is
    extended to give an estimate for the edge cell.

    Parameters
    ----------
    data_set : numpy.ndarray
        A numpy array containing elevation data in metres.
    x_coordinate : int
        The x (horizontal) coordinate of the given cell.
    y_coordinate : int
        The y (vertical) coordinate of the given cell.

    Returns
    -------
    tuple
        A tuple containing the given cell's four neighbours, in the
        order: (cell_left, cell_right, cell_above, cell_below).
    """
    cell_centre = data_set[y_coordinate, x_coordinate]  # given cell

    # If on the Western most point minus off the x gradient function
    # leading up to the edge
    if x_coordinate == 0:  # left-most column
        cell_left = cell_centre - x_gradient(cell_centre, data_set[
            y_coordinate, x_coordinate + 2]) * GRID_SIZE
    else:
        cell_left = data_set[y_coordinate, x_coordinate - 1]

    # If on the Eastern most point add on the x gradient function
    # leading up to the edge
    if x_coordinate == data_set.shape[1] - 1:  # right-most column
        cell_right = cell_centre + \
                     x_gradient(data_set[y_coordinate, x_coordinate - 2],
                                cell_centre) * GRID_SIZE
    else:
        cell_right = data_set[y_coordinate, x_coordinate + 1]

    # If on the Northern most point add on the y gradient function
    # leading up to the edge
    if y_coordinate == 0:  # top-most row
        cell_above = cell_centre + \
                     y_gradient(data_set[y_coordinate + 2, x_coordinate],
                                cell_centre) * GRID_SIZE
    else:
        cell_above = data_set[y_coordinate - 1, x_coordinate]

    # If on the Southern most point minus off the y gradient function
    # leading up the edge
    if y_coordinate == data_set.shape[0] - 1:  # bottom-most row
        cell_below = cell_centre - \
                     y_gradient(cell_centre, data_set[
                         y_coordinate - 2, x_coordinate]) * GRID_SIZE
    else:
        cell_below = data_set[y_coordinate + 1, x_coordinate]

    return cell_left, cell_right, cell_above, cell_below


def x_gradient(cell_left, cell_right):
    """The horizontal slope of the cell between the two given
    neighbouring cells cell_left and cell_right.
    Gives gradient from west to east (left to right)
    Negative of this function gives gradient from east to west (right to left)

    Parameters
    ----------
    cell_left : float
        The elevation in the cell on the left of the cell for which the
        gradient is being calculated.
    cell_right : float
        The elevation in the cell on the right of the cell for which the
        gradient is being calculated.

    Returns
    -------
    float
        The x gradient (horizontal slope) in metres.
    """
    return (cell_right - cell_left) / (2 * GRID_SIZE)


def y_gradient(cell_below, cell_above):
    """The vertical slope of the cell between the two given neighbouring
    cells, cell_below and cell_above.
    Gives gradient from south to north (below to above)
    Negative of this function gives gradient from north to south (above to below)

    Parameters
    ----------
    cell_below: float
        The elevation, e_x,y+1, of the cell below the cell for which the
        y gradient is being calculated.
    cell_above: float
        The elevation, e_x,y-1, of the cell above the cell for which the
        y gradient is being calculated

    Returns
    -------
    float
        The y gradient (vertical slope) in metres.
    """
    return (cell_above - cell_below) / (2 * GRID_SIZE)


def slopes_np(data_set):
    """Fast version of slopes() which uses indices arrays to take
    advantage of NumPy's fast built-in functions. Returns the total
    gradient (slope) at every location in the given data set of
    elevations.

    Parameters
    ----------
    data_set : np.ndarray
        Array of elevation grid data in metres.

    Returns
    -------
    np.ndarray
        The total gradient (slope) at every cell in the given array,
        according to the formula:
            slope_x_y = sqrt(x_gradient^2 + y_gradient^2), where
            x_gradient = (elevation_x+1,y - elevation_x-1,y)/10, and
            y_gradient = (elevation_x,y+1 - elevation_x,y-1)/10.
    """
    # x and y coordinates at every cell
    x, y = np.meshgrid(np.arange(data_set.shape[1]), np.arange(data_set.shape[0]))

    # Right cell neighbour coordinates
    x_plus_1 = np.insert(arr=x[:, 1:], obj=-1, values=x[:, -1], axis=1)

    # Right cell neighbour elevations
    e_x_plus_1_y = data_set[y, x_plus_1]

    # Left cell neighbour coordinates
    x_minus_1 = np.insert(arr=x[:, :-1], obj=0, values=x[:, 0], axis=1)

    # Left cell neighbour elevations
    e_x_minus_1_y = data_set[y, x_minus_1]

    # Bottom cell neighbour coordinates
    y_plus_1 = np.insert(arr=y[1:, :], obj=-1, values=y[-1, :], axis=0)

    # Bottom cell neighbour elevations
    e_x_y_plus_1 = data_set[y_plus_1, x]

    # Top cell neighbour coordinates
    y_minus_1 = np.insert(arr=y[:-1, :], obj=0, values=y[0, :], axis=0)

    # Top cell neighbour elevations
    e_x_y_minus_1 = data_set[y_minus_1, x]

    # x gradients at every cell
    dx = np.divide(np.subtract(e_x_plus_1_y, e_x_minus_1_y), 10.)

    # y gradients at every cell
    dy = np.divide(np.subtract(e_x_y_plus_1, e_x_y_minus_1), 10.)

    return np.sqrt(np.add(np.square(dx), np.square(dy)))  # slope_x_y equation


# Question 4
def surface_area(data_set, x_coordinate, y_coordinate):
    """Return the surface area of the dam at the given coordinates.

    Parameters
    ----------
    data_set : numpy.ndarray
        A numpy array containing the elevation data in metres.
    x_coordinate : int
        The x (horizontal) coordinate of the point at which to calculate
        the dam surface area.
    y_coordinate : int
        The y (vertical) coordinate of the point at which to calculate
        the dam surface area.

    Returns
    -------
    float
        The surface of the dam at the given coordinates in square
        metres.

    """
    assert slope(data_set, x_coordinate, y_coordinate) < SLOPE_TOLERANCE, \
        f'chosen point has slope of {slope(data_set, x_coordinate, y_coordinate)}' \
        f'metres and is therefore likely not on a dam.'

    dam_elevation = data_set[y_coordinate, x_coordinate]

    dam_mask = get_dam_mask(data_set, dam_elevation)  # True => dam, False => land

    return dam_mask_to_surface_area(dam_mask)


def get_dam_mask(data_set, elevation):
    """Returns a boolean array showing points on an elevation grid which
    lie on the dam surface given one elevation point of the dam water
    level.

    Parameters
    ----------
    data_set : numpy.ndarray
        A numpy array containing the elevation points in metres.
    elevation : float
        An elevation point which lies on the dam in metres.

    Returns
    -------
    numpy.ndarray
        A numpy array containing boolean values, where True represents
        points on the elevation grid which lie on the dam, and False
        represents points which lie outside the dam.
    """
    elevation_lower_bound = elevation - ELEVATION_TOLERANCE
    elevation_upper_bound = elevation + ELEVATION_TOLERANCE

    dam_mask = np.ones(shape=data_set.shape, dtype=np.bool_)  # all true

    dam_slopes = slopes_np(data_set)  # fast NumPy version (compared to `slopes()`)

    # N.B. Although this makes the overall dam mask more accurate, it
    # also, unfortunately, destroys the perimeter of the dam. This is
    # because the neighbouring cells of points on the perimeter of the
    # dam include cells which lie outside the dam, resulting in a
    # dramatically increased slope at these locations. We ran out of
    # time to find a workaround to this problem.
    dam_mask = np.where(dam_slopes > SLOPE_TOLERANCE, np.False_, dam_mask)

    dam_mask = np.where(np.logical_or(data_set > elevation_upper_bound,
                                      data_set < elevation_lower_bound),
                        np.False_, dam_mask)

    # The following function has minimal effect and is commented out to
    # improve performance (see documentation for
    # `remove_isolated_true_values()`)
    # dam_mask = remove_isolated_true_values(dam_mask)

    return dam_mask


def remove_isolated_true_values(dam_mask):
    """Removes isolated True values from the given boolean array, i.e.,
    removes True values surrounded by False values:

      False False False
      False True  False
      False False False

    Note that this function performs very slowly on the large dataset,
    and does not appear to have much effect on the dam mask result (on
    the small data set, there were only around 60 isolated True values
    scattered outside the dam area). We did not have enough time to
    optimise it in the same way we optimised `slopes_np()`, which is why
    `remove_isolated_true_values()` is commented out in `get_dam_mask()`.

    Parameters
    ----------
    dam_mask : np.ndarray
        Array of boolean values, where True represents locations lying
        on a dam, and False represents locations lying on land.

    Returns
    -------
    np.ndarray
        The given array with isolated True values replaced by False.
    """
    # Ignore edges, look at centre of array only
    for y in range(1, dam_mask.shape[0] - 1):
        for x in range(1, dam_mask.shape[1] - 1):
            neighbours = cell_neighbours(dam_mask, x, y)
            if dam_mask[y, x] and not any(neighbours):
                dam_mask[y, x] = np.False_

    return dam_mask


def slopes(data_set):
    """Alternative fo slopes_np() which returns a 2d array containing
    the slope at every coordinate of the given elevation grid. Note
    that, since this function uses native Python functions on a very
    large array, it is quite slow - hence, `slopes_np()`, which fully
    utilises NumPy functions, is preferred.

    Parameters
    ----------
    data_set : numpy.ndarray
        An array containing elevation data in metres.

    Returns
    -------
    numpy.ndarray
        An array of the same dimensions as the given array but with the
        value of the slope at every coordinate.
    """
    dam_slopes = np.empty(data_set.shape)
    for y in range(data_set.shape[0]):
        for x in range(data_set.shape[1]):
            dam_slopes[y, x] = slope(data_set, x, y)

    return dam_slopes


def plot_dam(dam, title='Dam'):
    """Plot a heatmap of the given dam elevation grid. Alternatively, if
    the given array is a boolean mask containing only dam (True) and
    land (False) values, plot a two-colour representation of the area.

    Parameters
    ----------
    dam : numpy.ndarray
        A numpy array containing elevation values or boolean values
        representing water (True) and land (False).
    title : str, optional
        The dam plot title (default: 'Dam').

    Returns
    -------
    None

    Notes
    -----
    Source: https://www.geeksforgeeks.org/how-to-draw-2d-heatmap-using-matplotlib-in-python/
    """
    plt.imshow(X=dam, cmap='winter_r')  # blue = True (dam), green = False
    plt.title(title)
    plt.xticks([]), plt.yticks([])  # hide x and y ticks
    plt.show()


def dam_mask_to_surface_area(dam_mask):
    """Takes a 2d array of boolean values where True represents an
    elevation estimated to be on a dam (water) and False represents
    non-dam areas such as terrain, concrete structures, the surrounding
    land, etc. Values in the grid are known to be 5 metres apart.

    Parameters
    ----------
    dam_mask : numpy.ndarray
        A 2d numpy array containing boolean values where True represents
        a point on a dam and False represents a point on the surrounding
        land. Points in the array are known to be 5 metres (GRID_SIZE)
        apart.
    Returns
    -------
    float
        The surface area of the dam in square-metres.
    """
    num_dam_points = np.sum(dam_mask)  # counts True values

    return np.multiply(num_dam_points, GRID_SIZE ** 2)


# Question 5:
def expanded_surface_area(data_set, water_level, x_coordinate, y_coordinate):
    """Calculate the expanded surface area of the dam at the given
    coordinates based on a hypothetical new, higher water level.

    Parameters
    ----------
    data_set : numpy.ndarray
        A numpy array containing elevation points of one or more dams
        and the surrounding area in metres.
    water_level : float
        The new (expanded) water level in metres.
    x_coordinate : int
        The x (horizontal) coordinate of an elevation point lying on a
        dam.
    y_coordinate : int
        The y (vertical) coordinate of an elevation point lying on a
        dam.

    Returns
    -------
    float
        The expanded surface area of the dam at the given coordinates
        for the given water level.
    """

    # Check if slope approximately zero and therefore probably a dam
    dam_slope = slope(data_set, x_coordinate, y_coordinate)
    assert dam_slope < SLOPE_TOLERANCE, f'chosen point with slope of' \
                                        + '{dam_slope} metres is likely not on a dam.'

    # Check expanded water level greater than elevation at given coordinates
    elevation = data_set[y_coordinate, x_coordinate]
    assert water_level >= elevation, f'the expanded water level, {water_level}' \
                                     + ' metres, must be greater than or equal to the elevation of' \
                                     + f'the point on the dam, {elevation:.2f} metres.'

    original_dam_mask = get_dam_mask(data_set, elevation)

    expanded_area_mask = get_expanded_area_mask(data_set, water_level,
                                                elevation)

    # Combining both areas
    expanded_dam_mask = original_dam_mask | expanded_area_mask

    return dam_mask_to_surface_area(expanded_dam_mask)


def get_expanded_area_mask(data_set, water_level, elevation):
    """Return the expanded dam area mask, i.e., a boolean array of dam
    (True) and land (False) values, given the elevation of the original
    dam and an expanded water level.

    Note that the original dam is not included in the mask, only the
    area surrounding the dam if the water level were to increase. Can
    also be used to produce a rough approximation of the surrounding
    catchment area, if given the right water level value.

    Parameters
    ----------
    data_set : np.ndarray
        Array of elevation values in metres.
    water_level : float
        The expanded water level in metres.
    elevation : float
        The elevation of the dam for which to produce the expanded area mask.

    Returns
    -------
    np.ndarray
        An array of boolean values, where True represents the expanded
        dam area which surrounds the original dam, and False represents
        all other areas (including the original dam).
    """
    return np.logical_and(data_set < water_level, data_set >
                          elevation + ELEVATION_TOLERANCE)


# Question 6:
def impute_missing_values(data_set):
    """Takes as input the large elevation grid (with missing values) and
    returns an elevation grid with the missing values imputed or
    approximated.

    Parameters
    ----------
    data_set : numpy.ndarray
        A numpy array of an elevation grid in metres.

    Returns
    -------
    numpy.ndarray
        An elevation grid with the missing values imputed.

    """
    return interpolate_missing(replace_outliers(data_set))


def replace_outliers(data_set, threshold=3):
    """Returns the given array with outliers above a certain z-score
    (default is 3) replaced with np.nan.

    Parameters
    ----------
    data_set : numpy.ndarray
        A numpy array of elevation data containing outliers.
    threshold : int, optional
        The z-score above which to remove (replace with np.nan) values.

    Returns
    -------
    numpy.ndarray
        The given numpy array with any outliers replaced with np.nan
        values.

    Notes
    -----
    Based on https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba

    """
    z_scores = np.abs(stats.zscore(data_set))
    return np.where(z_scores > threshold, np.nan, data_set)


def replace_outliers_iqr(data_set):
    """Alternative to replace_outliers() which uses inter-quartile range
    (IQR) instead of z-scores to remove outliers.

    Takes an input, the large elevation grid (with outliers), and
    identifies outliers through IQR replaces the outliers with np.nan.

    Parameters
    ----------
    data_set : numpy.ndarray
        An elevation grid with values in metres.

    Returns
    -------
    numpy.ndarray
        The given elevation grid with outliers replaced with np.nan.
    """
    # Converts NumPy array to Pandas DataFrame
    df = pd.DataFrame(data_set)

    for x in df:
        # Inter-quartile range formula
        q3, q1 = np.percentile(df.loc[:, x], [75, 25])
        iqr = q3 - q1

        upper_bound = q3 + (1.5 * iqr)
        lower_bound = q1 - (1.5 * iqr)

        # Values outside of lower and upper bound changed to np.nan
        df.loc[df[x] < lower_bound, x] = np.nan
        df.loc[df[x] > upper_bound, x] = np.nan

    return df.to_numpy()  # return as NumPy array


def interpolate_missing(data_set):
    """Takes a dataset containing elevation data and numpy.nan values
    (representing missing points) and returns the data set with the
    numpy.nan values interpolated using SciPy's interpolation.griddata()
    with the nearest neighbours method.

    Parameters
    ----------
    data_set : numpy.ndarray
        A numpy array containing elevation grid data and numpy.nan
        values to be interpolated.

    Returns
    -------
    numpy.ndarray
        The given array of elevation data with the numpy.nan values
        interpolated using the nearest neighbours method.

    Notes
    -----
    Based on
    https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python.
    """
    x = np.arange(0, data_set.shape[1])
    y = np.arange(0, data_set.shape[0])
    xx, yy = np.meshgrid(x, y)  # all x, y indices
    data_set = np.ma.masked_invalid(data_set)  # numpy masked array
    x1 = xx[~data_set.mask]  # valid x indices
    y1 = yy[~data_set.mask]  # valid y indices
    flattened = data_set[~data_set.mask]
    gd1 = griddata((x1, y1), flattened, (xx, yy), method='nearest')

    return gd1


def full_catchment_area(data_set, x_coordinate, y_coordinate):
    """Returns a boolean array showing points on an elevation grid which
    lie in the catchment area of the dam given an x and y coordinate point
    which lies on the dam itself.

    Parameters
    ----------
    data_set : numpy.ndarray
        An elevation grid with values in metres.
    x_coordinate : int
        The x (horizontal) coordinate of an elevation point lying on a
        dam.
    y_coordinate : int
        The y (vertical) coordinate of an elevation point lying on a
        dam.

    Returns
    -------
    numpy.ndarray
        A numpy array containing boolean values, where True represents
        points on the elevation grid which lie in the catchment area
        of the dam, and False represents points which lie outside the
        catchment area.

    """
    # utilises the surface area mask
    mask = get_dam_mask(data_set, data_set[y_coordinate, x_coordinate])

    # makes a list of coordinates of all True values in mask
    list_of_coordinates = []
    for y in range(mask.shape[0] - 1):
        for x in range(mask.shape[1] - 1):
            if mask[y, x] == True:
                list_of_coordinates.append((y, x))

    def catchment_area(list):
        # runs through the above list of coordinates of True values and checks
        # if the elevations leading off in each cardinal direction are higher.
        # if they are higher then they are considered part of the catchment area
        # and are changed to True.
        a = 0
        while a < len(list):
            y_coordinate = list[a][0]
            x_coordinate = list[a][1]
            for i in range(1, data_set.shape[1] - x_coordinate - 1):
                if data_set[y_coordinate, x_coordinate + i] >= ((data_set[y_coordinate, x_coordinate + i - 1]) - 0.02):
                    mask[y_coordinate, x_coordinate + i] = True
                else:
                    break

            for j in range(1, x_coordinate):
                if data_set[y_coordinate, x_coordinate - j] >= ((data_set[y_coordinate, x_coordinate - j + 1]) - 0.02):
                    mask[y_coordinate, x_coordinate - j] = True
                else:
                    break

            for k in range(1, data_set.shape[0] - y_coordinate - 1):
                if data_set[y_coordinate + k, x_coordinate] >= ((data_set[y_coordinate + k - 1, x_coordinate]) - 0.02):
                    mask[y_coordinate + k, x_coordinate] = True
                else:
                    break

            for l in range(1, y_coordinate):
                if data_set[y_coordinate - l, x_coordinate] >= ((data_set[y_coordinate - l + 1, x_coordinate]) - 0.02):
                    mask[y_coordinate - l, x_coordinate] = True
                else:
                    break
            a = a + 1

    catchment_area(list_of_coordinates)

    # repeats the above to incorporate the new True values
    second_list_of_coordinates = []
    for y in range(mask.shape[0] - 1):
        for x in range(mask.shape[1] - 1):
            if mask[y, x] == True:
                second_list_of_coordinates.append((y, x))

    catchment_area(second_list_of_coordinates)

    # repeats the above to incorporate the new True values
    third_list_of_coordinates = []
    for y in range(mask.shape[0] - 1):
        for x in range(mask.shape[1] - 1):
            if mask[y, x] == True:
                third_list_of_coordinates.append((y, x))

    catchment_area(third_list_of_coordinates)

    # to increase run time it would be better to only run the function again
    # with the new True rather than all of the coordinates.
    return mask


# Code in the following if statement will only be executed when this file is run - not when it is imported.
# If you want to use any of your functions (such as to answer questions) please write the code to
# do so inside this if statement. We'll cover it in more detail in an upcoming lecture.
if __name__ == "__main__":
    data_set_small = read_dataset('elevation_data_small.csv')  # Cotter Dam
    minimum = minimum_elevation(data_set_small)
    maximum = maximum_elevation(data_set_small)
    average = average_elevation(data_set_small)
    data_set_small_slopes = slopes_np(data_set_small)
    steepest = data_set_small_slopes.max()
    max_surface_area = expanded_surface_area(data_set_small,
                                             COTTER_DAM_WALL_ELEVATION,
                                             SMALL_DATA_SET_COTTER_DAM_POINT[0],
                                             SMALL_DATA_SET_COTTER_DAM_POINT[1])

    print('Written Report Question 1 answers')
    print('  Part 1')
    print(f'    minimum elevation = {minimum:.2f} m')
    print(f'    maximum elevation = {maximum:.2f} m')
    print(f'    average elevation = {average:.2f} m')
    print()
    print('  Part 2')
    print(f'    slope of steepest point = {steepest:.2f} m')
    print()
    print('  Part 3')
    print(f'    maximum surface area = {max_surface_area:.2f} m^2')
    print()

    print('Loading Question 6 plots and other plots...')
    # Uncomment each block of code to produce plots

    # Surface area of Cotter Dam
    # dam_point1 = (SMALL_DATA_SET_COTTER_DAM_POINT[0], SMALL_DATA_SET_COTTER_DAM_POINT[1])
    # elevation1 = data_set_small[dam_point1[1], dam_point1[0]]
    # dam_mask1 = get_dam_mask(data_set_small, elevation1)
    # plot_dam(dam_mask1, title='Surface area of Cotter Dam')

    # Expanded surface area of Cotter
    # water_level1 = COTTER_DAM_WALL_ELEVATION
    # expanded_dam_mask1 = get_expanded_area_mask(data_set_small, water_level1, elevation1)
    # plot_dam(dam_mask1 | expanded_dam_mask1, title='Maximum surface area of Cotter Dam')

    # Catchment area of Cotter Dam
    # catchment_area1 = full_catchment_area(data_set_small, dam_point1[0], dam_point1[1])
    # plot_dam(catchment_area1, title='Catchment area of Cotter Dam')

    # If wanting to produce below plots uncomment line 872 + any required block of code
    # Using the large data set
    # data_set_large = read_dataset('elevation_data_large.csv')

    # Surface area of Cotter Dam
    # dam_point2 = (LARGE_DATA_SET_COTTER_DAM_POINT[0], LARGE_DATA_SET_COTTER_DAM_POINT[1])
    # elevation2 = data_set_large[dam_point2[1], dam_point2[0]]
    # dam_mask2 = get_dam_mask(data_set_large, elevation2)
    # plot_dam(dam_mask2, title='Surface area of Cotter Dam')

    # Surface area of Bendora Dam
    # dam_point3 = (LARGE_DATA_SET_BENDORA_DAM_POINT[0], LARGE_DATA_SET_BENDORA_DAM_POINT[1])
    # elevation3 = data_set_large[dam_point3[1], dam_point3[0]]
    # dam_mask3 = get_dam_mask(data_set_large, elevation3)
    # plot_dam(dam_mask3, title='Surface area of Bendora Dam')

    # Surface area of Corin Dam
    # dam_point4 = (LARGE_DATA_SET_CORIN_DAM_POINT[0], LARGE_DATA_SET_CORIN_DAM_POINT[1])
    # elevation4 = data_set_large[dam_point4[1], dam_point4[0]]
    # dam_mask4 = get_dam_mask(data_set_large, elevation4)
    # plot_dam(dam_mask4, title='Surface area of Corin Dam')

    # Surface area of all three dams
    # plot_dam(dam_mask2 | dam_mask3 | dam_mask4, title = 'Surface area of Cotter, Bendora and Corin Dams')

    # Catchment area of Bendora Dam
    # plot_dam(full_catchment_area(data_set_large, dam_point3[0], dam_point3[1]),title='Catchment area of Bendora Dam')

    # Catchment area of Corin Dam
    # plot_dam(full_catchment_area(data_set_large, dam_point4[0], dam_point4[1]),title='Catchment area of Corin Dam')

