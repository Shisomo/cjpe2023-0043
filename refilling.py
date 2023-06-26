import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from scipy.misc import derivative
from scipy.optimize import curve_fit


# cumulative formula
def integrate_data(a):
    return (a.sum() * 2 - a[0] - a[-1]) / 2


#  Re method, the prediction refilling method based on “Time Segment”
def refilling(ydata, default_point=30, kneed_process=True) -> np.array and list:
    # ydata is sap flow data
    # default_point is the default point to differ refilling and transpiration
    # kneed_process means automatic kneed point switch
    # Create the x-axis of the data
    xdata = np.arange(len(ydata))

    if kneed_process == True:
        # The mothed to distinguishing transpiration and refilling
        # Adjust the data format for inflection point detection
        x_k, y_k = xdata.tolist(), ydata.tolist()
        # Initialize the model for detection configuration
        model = KneeLocator(x_k, y_k, curve='convex', direction='decreasing', online=True,
                            interp_method="polynomial")
        # get kneedle point
        output_knees = [model.knee, model.knee_y]
        # Test results
        if output_knees[0] is None:
            output_knees[0] = default_point
        if isinstance(output_knees[0], int):
            output_knees[0] = int(output_knees[0])
    else:
        output_knees = [default_point, ydata[default_point]]

    # prediction refilling based on Midnight
    ydata_fit = ydata.copy()
    ydata_fit[default_point:] = 0
    # Set return format
    return ydata_fit, [sum(ydata_fit) / sum(ydata)]


#  Line method, the prediction refilling method based on linear decay model
def extended_line_refilling(ydata, default_point=10, kneed_process=True) -> np.array and list:
    # Create the x-axis of the data
    xdata = np.arange(len(ydata))

    # The mothed to distinguishing transpiration and refilling
    if kneed_process == True:
        # Adjust the data format for inflection point detection
        x_k, y_k = xdata.tolist(), ydata.tolist()
        # Initialize the model for detection configuration
        model = KneeLocator(x_k, y_k, curve='convex', direction='decreasing', online=False, interp_method="polynomial")
        # get kneedle point
        output_knees = [model.knee, model.knee_y]
        # Test results
        if output_knees[0] is None:
            output_knees[0] = default_point
        if isinstance(output_knees[0], int):
            output_knees[0] = int(output_knees[0])
    else:
        output_knees = [default_point, ydata[default_point]]

    def fn_exp(x, b):
        return ydata[0] * np.exp(-b * x)

    popt, _ = curve_fit(fn_exp, xdata[:output_knees[0]], ydata[:output_knees[0]], bounds=(0, [0.5]))
    ydata_fit = ydata[0] * np.exp(-((popt[0] * xdata).astype(np.float64)))

    # prediction refilling based on Liner decay model
    # Create Liner model
    def fn_line(x):
        return ydata[0] * np.exp(-(popt[0] * x))

    # The slope of the refilling curve
    liner_slope = derivative(fn_line, output_knees[0], dx=1e-6)
    ydata_fit[:int(output_knees[0])] = ydata[:int(output_knees[0])]
    ydata_fit[int(output_knees[0]):] = liner_slope * xdata[int(output_knees[0]):] + ydata[
        int(output_knees[0])] - liner_slope * output_knees[0]
    # Eliminate refilling curve < 0
    for i in range(len(ydata_fit)):
        if ydata_fit[i] < 0:
            ydata_fit[i] = 0

    # Set return format
    return ydata_fit, [sum(ydata_fit) / sum(ydata)]


#  Exp method, the prediction refilling method based on Exponential decay model.
def extended_exp_refilling(ydata, default_point=10, kneed_process=True) -> np.array and list:
    # Create the x-axis of the data
    xdata = np.arange(len(ydata))

    # The mothed to distinguishing transpiration and refilling
    if kneed_process == True:
        # Adjust the data format for inflection point detection
        x_k, y_k = xdata.tolist(), ydata.tolist()
        # Initialize the model for detection configuration
        model = KneeLocator(x_k, y_k, curve='convex', direction='decreasing', online=False, interp_method="polynomial")
        # get kneedle point
        output_knees = [model.knee, model.knee_y]
        # Test results
        if output_knees[0] is None:
            output_knees[0] = default_point
        if isinstance(output_knees[0], int):
            output_knees[0] = int(output_knees[0])
    else:
        output_knees = [default_point, ydata[default_point]]

    # prediction refilling based on Exponential decay model
    # Create Exponential model
    def fn_exp(x, b):
        return ydata[0] * np.exp(-b * x)

    # Start
    popt, _ = curve_fit(fn_exp, xdata[:output_knees[0]], ydata[:output_knees[0]], bounds=(0, [0.5]))
    # "ydata_fit" is refilling curve
    ydata_fit = ydata[0] * np.exp(-((popt[0] * xdata).astype(np.float64)))

    # Gradient descent adjusts the slope of the curve
    num_step = 1
    adjustment_step_len = 0.0002
    while (ydata[5:15] - ydata_fit[5:15]).min() < 0:
        if num_step > 500:
            break
        ydata_fit = ydata[0] * np.exp(-(((popt[0] + adjustment_step_len * num_step) * xdata).astype(np.float64)))
        num_step += 1
    # Set return format
    return ydata_fit, [sum(ydata_fit) / sum(ydata)]


# Et method, the prediction method based on extend transpiration inversion
def extended_transpiration(ydata, default_point=10, kneed_process=True, min_strategy=True):
    # Create the x-axis of the data
    xdata = np.arange(len(ydata))

    if kneed_process == True:
        # The mothed to distinguishing transpiration and refilling
        # Adjust the data format for inflection point detection
        x_k, y_k = xdata.tolist(), ydata.tolist()
        # Initialize the model for detection configuration
        model = KneeLocator(x_k, y_k, curve='convex', direction='decreasing', online=False,
                            interp_method="polynomial")
        # get kneedle point
        output_knees = [model.knee, model.knee_y]
        # Test results
        if output_knees[0] is None:
            output_knees[0] = default_point
        if isinstance(output_knees[0], int):
            output_knees[0] = int(output_knees[0])
    else:
        output_knees = [default_point, ydata[default_point]]

    # prediction refilling based on Midnight

    ydata_fit = ydata.copy()
    ydata_fit[default_point:] = 0

    if min_strategy == True:
        ydata_fit = ydata_fit + (ydata[output_knees[0]:] - ydata_fit[output_knees[0]:]).min()
    else:
        def fn_pushback(x, a):
            return a

        popt, _ = curve_fit(fn_pushback, xdata[output_knees[0]:], ydata[output_knees[0]:])
        ydata_fit = popt[0] + 0 * ydata
    # Set return format
    return ydata_fit, [sum(ydata_fit) / sum(ydata)]


def plot_refilling(xdata, ydata, msg, channel, path):
    # time_index_x 横轴 X
    # y = np.vstack(y).transpose()
    # y_2 = np.vstack(y_2).transpose()
    plt.figure(figsize=(8, 8), dpi=40)
    plt.figure(1)
    plt.ylim(0, ydata.max() * 1.2)
    plt.text(xdata.max() * 0.5, ydata.max() * 0.9, msg + '_' + channel)
    plt.plot(xdata, ydata, 'o-')
    # plt.plot(x, y_2)
    address = path + msg + '_' + channel + '.png'
    plt.savefig(address)
    plt.clf()
