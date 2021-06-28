##################################################################
# Curve - Waveform and Signal manipulation library
# Author: Juan Pablo Caram
# Date: 2018-2019
#
##################################################################


import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from matplotlib.pyplot import plot, semilogx, semilogy, loglog, annotate, gca
from collections import UserList
import pickle
from itertools import groupby


class CurveError(Exception):
    pass


class ComplexCurveError(Exception):
    pass


class Curve(object):

    version = 1.1

    def __init__(self, data, y=None, dtype=None):
        """
        Initilizes the Curve object with data.
        
        Usage:
        
            mycurve = Curve(x, y)
            
        or
        
            mycurve = Curve([x, y])
        
        :param data: A 2xn (or nx2) list or Numpy array, or a 1-d vector
                     for the x-axis values (In this case, must also provide
                     the y parameter).
        :param y: 1-d vector of y-axis values correspondind to the x-axis
                  values provided in the data parameter. If this is not
                  provided it assumes that data contains both x and y-axis data
                  in a 2-times-n data structure.
        :param dtype: Type in which the data is stored.
        """

        if y is None:
            
            if isinstance(data, list):
                if dtype is not None:
                    # Explicitly convert types.
                    self.x = np.array(data[0], dtype=dtype[0])
                    self.y = np.array(data[1], dtype=dtype[1])
                else:
                    # Apply types if list or tuple.
                    # X
                    if isinstance(data[0], (list, tuple)):
                        self.x = np.array(data[0], dtype=np.float64)  # Re-type.
                    else:
                        self.x = np.array(data[0])  # Assume its type is understood by numpy.

                    # Y
                    if isinstance(data[1], (list, tuple)):
                        self.y = np.array(data[1], dtype=np.float64)
                    else:
                        self.y = np.array(data[1])

            elif data.shape[0] == 2:  # Is Numpy array
                # self.data = data
                if dtype is not None:
                    self.x = np.array(data[0], dtype=dtype[0])
                    self.y = np.array(data[1], dtype=dtype[1])
                else:
                    self.x = np.array(data[0], dtype=np.float64)
                    self.y = np.array(data[1], dtype=np.float64)

            elif data.shape[1] == 2:  # Is Numpy array
                # self.data = data.transpose()
                d = data.transpose()
                if dtype is not None:
                    self.x = np.array(d[0], dtype=dtype[0])
                    self.y = np.array(d[1], dtype=dtype[1])
                else:
                    self.x = np.array(d[0], dtype=np.float64)
                    self.y = np.array(d[1], dtype=np.float64)
    
            else:
                raise ValueError("Expected a 2xn or nx2 matrix but shape is " +
                                 str(data.shape))
        
        else:  # y has been provided
            # self.data = np.array([data, y], dtype=dtype)
            if dtype is not None:
                self.x = np.array(data, dtype=dtype[0])
                self.y = np.array(y, dtype=dtype[1])
            else:
                if isinstance(data, (list, tuple)):
                    self.x = np.array(data, dtype=np.float64)
                else:
                    self.x = np.array(data)

                if isinstance(y, (list, tuple)):
                    self.y = np.array(y, dtype=np.float64)
                else:
                    self.y = np.array(y)
    
    @staticmethod
    def from_pickle(fname):
        """
        Create a Curve object by reading data from the file fname.
        The data must be casted into a nx2 or 2xn Numpy array.
        
        :param fname: Path to a Python pickle file.
        :returns: Curve object.
        """
        
        with open(fname, 'rb') as f:
            sig = np.array(pickle.load(f))
            
        return Curve(sig)

    def __getstate__(self):
        """
        State of the object for serialization.

        :return: (self.x, self.y)
        """
        return self.x, self.y

    def __setstate__(self, state):
        """
        Set the state of the object from (x, y).

        :param state: (x, y)
        :return: None
        """
        self.x, self.y = state

    def __eq__(self, other):

        return isinstance(other, Curve) and (
            all(other.x == self.x) and all(other.y == self.y)
        )

    # @property
    # def x(self):
    #     return self.data[0]

    # @x.setter
    # def x(self, value):
    #     self.data[0] = value

    # @property
    # def y(self):
    #     return self.data[1]

    # @y.setter
    # def y(self, value):
    #     self.data[1] = value

    @property
    def duration(self):
        """
        Difference between the first and last values of the x-axis.
        """
        return self.x[-1] - self.x[0]
    
    def check(self, fix=0):
        """
        Check for proper data (Monotonicity of the x-axis values).
        Fix it if necessary by sorting (x, y) pairs by x.

        :param fix: Fix non-monotonicity in the x-axis by sorting (x, y) pairs by x.
                    Default is False.
        """
        
        # Check for monotonic time.
        tdiff = np.diff(self.x)
        tgood = tdiff > 0
        
        while not np.all(tgood):
            if fix:
                # TODO: This always drops the first point.
                # data = [
                #     self.x[1:][tgood],
                #     self.y[1:][tgood]
                # ]
                self.x = self.x[1:][tgood]
                self.y = self.y[1:][tgood]
                # self.data = np.array(data, dtype=float)
                tdiff = np.diff(self.x)
                tgood = tdiff > 0
            else:
                raise CurveError("X axis is non-monotonic.")

    def fix_zero_dx(self):
        """
        Removes data points where the step size in the x-axis is 0
        (i.e. consecutive points with the same x-axis value).

        :return: Curve instance without repeating x-axis values.
        """
        tdiff = np.diff(self.x)
        idx = np.argwhere(tdiff == 0)

        x = np.delete(self.x, idx + 1)
        y = np.delete(self.y, idx + 1)

        return Curve(x, y)

    def eye(self, start, period, uis=1.0):
        """
        Generates an eye diagram slicing the waveform
        in period-long segments. Returns a list of these segments.

        :param start: Start time (x-axis value) at which to start the eye diagram.
        :param period: Period of the signal, interval at which to slice the curve,
                       width of the reulting eye digram.
        :param uis: Multiplier for period. Default is 1.0.
        :returns: List of Curves
        """
        segments = []
        i = 0
        while start + (i+uis)*period <= self.x[-1]:
            seg = self.interval(xmin=start + i*period, xmax=start + (i+uis)*period)
            seg.x -= start + i*period
            segments.append(seg)
            i += 1
        return segments
    
    def at(self, value, interpolate=True):
        """
        Returns y(x=value), interpolating values (trapezoid), otherwise
        returns y for the closest existing x < value. If values is
        a list or Numpy array, returns a Numpy array of corresponding values of y.

        :param value: X-axis value(s) for which the Y-axis value(s) is(are) returned.
                      May be a list/Numpy array of values or a scalar (float/int).
        :param interpolate: Wheather to interpolate between closest X values,
                            or to approximate to the nearest available.
                            Default is True.
        :return: Corresponding Y-axis value(s).
        """

        # If value is a list, unpack and re-run per item.
        # NOTE: This will interpolate regardless of the 'interpolate' parameter.
        if isinstance(value, list) or isinstance(value, np.ndarray):
            # return np.array([self.at(x, interpolate=interpolate) for x in value])
            return np.interp(value, self.x, self.y)

        # The x axis is ordered so we can do a quick search.
        index_above = np.searchsorted(self.x, value)

        # Outside the limits? Just return first/last value.
        if index_above == 0:
            return self.y[0]

        if index_above == len(self.x):
            return self.y[-1]

        # Interpolate
        if interpolate:
            x1 = self.x[index_above]
            x0 = self.x[index_above - 1]
            y1 = self.y[index_above]
            y0 = self.y[index_above - 1]
            dx = x1 - x0
            dy = y1 - y0
            return y0 + (value - x0) * dy / dx
        else:
            return self.y[index_above - 1]

    def resample(self, max_grow=10):
        """
        Resamples the curve at uniform intervals. It chooses the minumum existing
        spacing between x-axis values as the sampling period unless the resulting
        number of samples is greater than max_grow times the current number of samples.
        In such case, the period is set to be such that the resulting number of
        samples is exactly max_grow times the current number of samples.

        :param max_grow: Maximum allowed increase in number of data points.
        :returns: Resulting resampled curve.
        """

        # Minimum spacing between x values.
        mind = np.min(np.diff(self.x))

        # New number of samples
        n = int(self.duration / mind + 1)

        if n > max_grow * self.x.size:
            n = max_grow * self.x.size

        # x and y
        x = np.linspace(self.x[0], self.x[-1], num=n)
        y = self.at(x)

        return Curve([x, y])

    def envelope(self):
        """
        Calculates the envelope of the curve. Uses the hilbert function from
        scipy.signal.

        :returns: Envelope curve.
        """

        if self.y.dtype == np.dtype(np.complex128):
            raise ComplexCurveError('Complex-valued Curve does not support envelope()')

        resampled = self.resample()
        analytic = hilbert(resampled.y)

        return Curve([resampled.x, np.abs(analytic)])

    def envelope2(self, tc=None, numpoints=101):
        """
        Calculates the envelope of the curve. Slices the curve into uniform intervals
        and computes the maximum of the interval to determine the envelope.

        :param tc: Interval width for computing the maximum as the envelope.
                   If not provided, tc = duration / (numpoints - 1).
        :param numpoints: Used if tc is not provided to calaculate tc.
        :returns: Envelope curve.
        """

        if self.y.dtype == np.dtype(np.complex128):
            raise ComplexCurveError('Complex-valued Curve does not support envelope2()')

        if tc is not None:
            numpoints = int(self.duration / tc + 1)
        else:
            tc = self.duration / (numpoints - 1)

        x = np.linspace(self.x[0], self.x[-1], num=numpoints)
        y = np.zeros(numpoints)

        for i in range(numpoints - 1):
            seg = self.interval(xmin=self.x[0] + i * tc, xmax=self.x[0] + i * tc + tc)
            y[i] = seg.y.max()

        y[-1] = y[-2]

        return Curve([x, y])

    def diff(self):
        """
        Computes the difference (derivative) of this curve.
        
        :returns: A Curve containing the derivative of this curve.
        """

        dy = np.diff(self.y)
        dx = np.diff(self.x)

        # Note the arbitrary use of x[1:] for the new Curve.
        return Curve([self.x[1:], dy / dx])

    def interval(self, xmin=None, xmax=None, include_edges=True, interpolate=True):
        """
        Extracts a segment of the curve for xmin < x < xmax. If xmin or xmax
        are not specified, they are considered the min(x) or max(x)
        respectively.

        :param xmin: Minimum x-axis value of the interval.
        :param xmax: Maximum x-axis value of the interval.
        :param include_edges: If xmin or xmax exceed the limits of the curve,
                              whether to include these limits or truncate
                              at the actual limits of the curve.
                              Default is True.
        :param interpolate: If include_edge is True, whether to interpolate to
                            compute the extremes. Default is True.
        :returns: Curve for the specified interval.
        """

        index_above_min = None
        index_above_max = None
        ymin = None
        ymax = None

        # data = np.copy(self.data)
        x = np.copy(self.x)
        y = np.copy(self.y)

        if xmin is not None:
            index_above_min = np.searchsorted(self.x, xmin)
            # Update...
            # data = self.data[:, index_above_min:]
            x = self.x[index_above_min:]
            y = self.y[index_above_min:]

            if include_edges:

                # xmin could end up being equal or larger than
                # due to numerical errors.
                # if xmin < data[0][0]:
                if xmin < x[0]:
                    ymin = self.at(xmin, interpolate=interpolate)
                    # Update...
                    # data = np.hstack([[[xmin], [ymin]], data])
                    x = np.hstack([[xmin], x])
                    y = np.hstack([[ymin], y])

        if xmax is not None:
            # index_above_max = np.searchsorted(data[0], xmax)
            index_above_max = np.searchsorted(x, xmax)
            # print "index_above_max =", index_above_max
            # Update...
            # data = data[:, :index_above_max]
            x = x[:index_above_max]
            y = y[:index_above_max]

            if include_edges:

                # if xmax > data[0][-1]:
                if xmax > x[-1]:
                    ymax = self.at(xmax, interpolate=interpolate)
                    # Update...
                    # data = np.hstack([data, [[xmax], [ymax]]])
                    x = np.hstack([x, [xmax]])
                    y = np.hstack([y, [ymax]])

        # return Curve(data)
        return Curve(x, y)

    def integrate(self):
        """
        Generates a new Curve with the integral (trapezoidal)
        of this Curve.

        :return: A Curve containing the integral of this curve.
        """

        intgr = self.copy()
        intgr.y[0] = 0

        dx = np.diff(self.x)
        dy = 0.5 * (self.y[:-1] + self.y[1:])
        da = dx * dy

        intgr.y[1:] = np.cumsum(da)

        return intgr

    def average(self):
        """
        Computes a curve whose value at any given x, are
        the average of the y-axis values for all previous
        values of x.
        
        :return: Curve with the average of this Curve throughout x.
        """

        # data = np.zeros(self.data.shape)

        relativex = self.x - self.x[0]
        # deltax = relativex[1:] - relativex[:-1]
        contrib = 1 / relativex[1:]
        values = self.integrate().y[1:] * contrib

        # data[0, :] = self.x[:]
        x = self.x[:]  # Makes a copy
        # data[1][0] = self.y[0]
        y = np.zeros(self.y.shape, dtype=self.y.dtype)
        y[0] = self.y[0]
        # data[1][1:] = values
        y[1:] = values
        # return Curve(data)
        return Curve(x, y)

    def cross(self, edge=None):
        """
        Computes the times (or x-axis values) at which this curve
        crosses 0 (in y-axis values).
        
        To compute the crossing of a different threshold,
        shift the curve first:
            
            mycurve = Curve(x, y)
            cross_times = (mycurve - threshold).cross()
            
        :param edge: Whether to get just rising or falling edges.
                     Possible values are 'rise', 'fall' or None (default).
                     If None, it computes both rising and falling edges.
        :return: 1-d Numpy array of values corresponding to when/where
                 along the x-axis, the y-axis values cross 0.
        """

        if self.y.dtype == np.dtype(np.complex128):
            raise ComplexCurveError('Complex-valued Curve does not support cross()')

        results = []
        idx = 1

        def calc_cross(i):
            t2 = self.x[i]
            v2 = self.y[i]
            t1 = self.x[i - 1]
            v1 = self.y[i - 1]
            m = (v2 - v1) / (t2 - t1)
            return (m * t1 - v1) / m  # tcross
        
        while True:
            try:
                # np.argwhere returns list of lists.
                idx += np.argwhere(self.y[idx:] > 0)[0][0]
                if edge is None or edge is 'rise':
                    # Interpolate.
                    # TODO: idx - 1 exists?
                    # t2 = self.x[idx]
                    # v2 = self.y[idx]
                    # t1 = self.x[idx - 1]
                    # v1 = self.y[idx - 1]
                    # m = (v2 - v1) / (t2 - t1)
                    # tcross = (m * t1 - v1) / m
                    # NOTE: This condition can only be False
                    #       the first time. Optimize this so it is not
                    #       evaluated every time.
                    if self.y[idx - 1] <= 0:
                        tcross = calc_cross(idx)
                        results.append(tcross)

                idx += np.argwhere(self.y[idx:] < 0)[0][0]
                if edge is None or edge is 'fall':
                    # t2 = self.x[idx]
                    # v2 = self.y[idx]
                    # t1 = self.x[idx - 1]
                    # v1 = self.y[idx - 1]
                    # m = (v2 - v1) / (t2 - t1)
                    # tcross = (m * t1 - v1) / m
                    # NOTE: This condition can only be False
                    #       the first time. Optimize this so it is not
                    #       evaluated every time.
                    if self.y[idx - 1] >= 0:
                        tcross = calc_cross(idx)
                        results.append(tcross)
            except IndexError:
                break

        return np.array(results)

    def cross2(self, edge=None):
        """
        NOTE: cross2() will eventually replace cross()

        Computes the times (or x-axis values) at which this curve
        crosses 0 (in y-axis values).

        To compute the crossing of a different threshold,
        shift the curve first:

            mycurve = Curve(x, y)
            cross_times = (mycurve - threshold).cross()

        :param edge: Whether to get just rising or falling edges.
                     Possible values are 'rise', 'fall' or None (default).
                     If None, it computes both rising and falling edges.
        :return: 1-d Numpy array of values corresponding to when/where
                 along the x-axis, the y-axis values cross 0.
        """

        if self.y.dtype == np.dtype(np.complex128):
            raise ComplexCurveError('Complex-valued Curve does not support cross2()')

        def calc_cross(i):
            """
            Interpolate between points i and i+1.
            """
            t2 = self.x[i + 1]
            v2 = self.y[i + 1]
            t1 = self.x[i]
            v1 = self.y[i]
            m = (v2 - v1) / (t2 - t1)
            tcross = (m * t1 - v1) / m
            return tcross

        ypos = self.y >= 0
        yposprime = np.diff(ypos.astype(int))
        # yposprime = 1 for positions right before a rising edge
        # crossing, -1 before a falling edge crossing and 0
        # everywhere else.

        if edge is None:
            idx = np.argwhere(yposprime != 0)
        elif edge == 'rise':
            idx = np.argwhere(yposprime == 1)
        elif edge == 'fall':
            idx = np.argwhere(yposprime == -1)
        else:
            raise ValueError('Unknown edge: {}'.format(edge))

        xc = np.apply_along_axis(calc_cross, 0, idx)
        return xc[:, 0]

    def period(self, threshold=None, verbose=False):
        """
        Computes a curve with the period of this curve. The period
        is defined as the time between rising edges crossing the
        specified threshold. If not provided, it is set to the
        curve's average. Values are defined at the time of the
        threshold crossing and are with respect to the previous
        threshold crossing.

        :param verbose: If true, prints debug information.
        :param threshold: Value of the curve at which it is considered
            to have completed/started a period.
        :return: Period Curve.
        """

        if threshold is None:
            threshold = self.average().y[-1]
            if verbose:
                print("Threshold set to", threshold)
        
        crossingt = (self - threshold).cross2(edge='rise')

        p = crossingt[1:] - crossingt[:-1]
        return Curve([crossingt[1:], p])

    def duty(self, threshold=None, verbose=False):
        """
        Calculates the duty cycle of the signal/curve.

        :param threshold: Y-axis value threshold for logic high-low.
                          If not provided, the average Y is computed and the value
                          at the end of the signal is interpreted as the amplitude.
                          The threshold is then set to half of the amplitude.
                          Default is None.
        :param verbose: Print out additional information.
        :return: Curve containing the ducty cycle as a function of X.
        """

        if threshold is None:
            threshold = self.average().y[-1]
            if verbose:
                print("Threshold set to", threshold)

        crossingtr = (self - threshold).cross(edge='rise')
        crossingtf = (self - threshold).cross(edge='fall')
        p = self.period(threshold=threshold, verbose=verbose)

        pts = min([len(crossingtr), len(crossingtf), len(p.y)])
        ctr = crossingtr[-pts:-1]
        ctf = crossingtf[-pts:-1]
        per = p.y[-pts:-1]

        # print pts, len(ctr), len(ctf), len(per)

        duty = ctr - ctf

        if ctr[-1] > ctf[-1]:
            return (per - duty) / per
        else:
            return -duty / per

    def frequency(self, threshold=None, verbose=False):
        """
        Computes the frequency of the signal/curve. This is computed
        as 1/period. See Curve.period().

        :param threshold: Threshold used for computing the period of each cycle.
        :param verbose: Print out additional information.
        :return: Curve containing the frequency of this curve as a function of X.
        """

        p = self.period(threshold=threshold, verbose=verbose)
        f = p.copy()

        f.y = 1.0 / p.y
        return f

    def edgetime(self, edge='rise', lowfrac=0.1, highfrac=0.9):
        """
        Computes the rise/fall times of the signal/curve. The rise or fall times
        are defined as the time (X-axis interval) it takes for the signal value
        (Y-axis value) to transition between lowfrac and highfrac fractions of the
        amplitude. The amplitude is determined as the difference between the absolute
        minumum and maximum Y-axis values across all X-axis values.

        :param edge: Which edge, 'rise' or 'fall'
        :param lowfrac: Lower limit of the transition range as a fraction of the
                        amplitude.
        :param highfrac: Upper limit of the transition range as a fraction of the
                         amplitude.
        :return: Curve containing the rise or fall times of this curve as a function of X.
                 The X-axis values are specified for the end of the transition.

        """
        if self.y.dtype == np.dtype(np.complex128):
            raise ComplexCurveError('Complex-valued Curve does not support edgetime()')

        # Swing
        high = self.y.max()
        low = self.y.min()
        swing = high - low

        # Thresholds
        thrh = high - swing * (1 - highfrac)
        thrl = low + swing * lowfrac

        # Crossing times
        crossl = (self - thrl).cross(edge=edge)
        crossh = (self - thrh).cross(edge=edge)

        # Ensure the beginning of the edge is first
        if edge == 'rise':
            cross1 = crossl
            cross2 = crossh
        else:
            cross1 = crossh
            cross2 = crossl

        while cross1[0] > cross2[0]:
            cross2 = cross2[1:]

        # Ensure the beginning is less than 1 cycle early
        while cross1[1] < cross2[0]:
            cross1 = cross1[1:]

        # Truncate lengths to the minimum
        n = min([len(cross1), len(cross2)])
        cross1 = cross1[0:n]
        cross2 = cross2[0:n]

        edget = cross2 - cross1

        return Curve([cross2, edget])

    @staticmethod
    def get_contiguous(x, nbins):
        """
        The list x contains indexes of a list which is nbins elements long.
        This function locates all contiguous indexes and returns a list
        of contiguous groups. Each group contains [first index, last index,
        number of contiguous indexes]. The origin list is assumed circular,
        therefore, if one group end at nbins-1 and another starts at 0,
        they are merged into a single group.

        Example:

            >>> Curve.get_contiguous([0, 1, 5, 6, 9], 10)
            [[5, 6, 2], [9, 1, 3]]

        :param x: List of indexes of an origin list.
        :param nbins: Number of elements of the origin list.
        :returns: List, each element is a 3-element list. Empty if
                  x is empty.
        """

        n = len(x)
        if n == 0:
            return []

        groups = [[x[0], x[0], 1]]
        for i in range(n - 1):
            if x[i] + 1 == x[(i + 1) % n]:
                groups[-1][1] = x[i + 1]
                groups[-1][2] += 1
            else:
                groups.append([x[(i + 1) % n], x[(i + 1) % n], 1])

        if groups[-1][1] == nbins - 1 and groups[0][0] == 0:
            groups[0][0] = groups[-1][0]
            groups[0][2] += groups[-1][2]
            groups.pop()
        return groups

    @staticmethod
    def get_largest_group(x, nbins):
        """
        Returns the largest group from the list generated by get_contiguous().

        :param x: List of indexes of an origin list.
        :param nbins: Number of elements of the origin list.
        :returns: 3-element list, [first, last, length].
        """

        groups = Curve.get_contiguous(x, nbins)
        if len(groups) == 0:
            return None
        largest = groups[0]
        if len(groups) == 1:
            return largest
        for i in range(1, len(groups)):
            if groups[i][2] > largest[2]:
                largest = groups[i]
        return largest

    @staticmethod
    def prbs_check(data, taps=[7, 6]):
        """
        Returns a list where each element is positive if the
        correspondig bit was erroneous based on the
        given PRBS sequence.

        Example:

            >>> tstart = 1.006e-8
            >>> period = 1/25e9
            >>> print "period = {:.1f} ps".format(period*1e12)
            >>> tsample = np.arange(tstart, txdiffc.x[-1], period)
            >>> samples = [1 if x>0 else 0 for x in txdiffc.at(tsample)]
            >>> print samples[:10]
            >>> print sum(Curve.prbs_check(samples))

        :param data: List of bit values, bolean or 0/1.
        :param taps: Specifies the LFSR taps in decreasing order (Only 2 taps implemented).
        :returns: List of correct/incorrect values (0 is correct).
        """
        output = [
            (data[i] ^ (data[i + (taps[0] - taps[1])] ^ data[i + taps[0]])) or
            not any(data[i:i+7])
            for i in range(0, len(data) - taps[0])
        ]
        return output

    def eye_box(self, start, f0, threshold=0.0, phase_step=0.05):
        """
        Computes the coordinates of the eye opening in a digital signal.

        Locates the eye opening by slicing the bit period into slices
        of width phase_step (fraction of bit period), detecting which
        slices don't have threshold crossings, finding the largest
        number of contiguous threshold-crossing-free slices, and
        sampling in the middle. Verified with PRBS check.

        Each cycle is normalized to 1.0, therefore the left and
        right coordinates are between 0 and 1.0, and the width of
        the eye in seconds is (right - left) / f0. Since the eye
        is periodic, it is possible to have right < left. In such case,
        the width is (right + 1 - left) / f0.

        :param start: X-axis value at which to start measuring.
        :param f0: Symbol rate.
        :param threshold: Y-axis value above which it is interpreted as a
                          logic 1, otherwise as logic 0.
        :param phase_step: Width of the bins in fraction of one period (1/f0)
                           to look for absence of threshold crossings.
        :returns: [success, left, right, bottom, top, nsamples]
        """

        if self.y.dtype == np.dtype(np.complex128):
            raise ComplexCurveError('Complex-valued Curve does not support eye_box()')

        c1 = self.interval(xmin=start)

        # Reset X axis
        c1.x -= c1.x[0]

        # Crossing times
        crossings = (c1 - threshold).cross2()
        # Normalize to the period, modulo 1.
        data = (crossings * f0) % 1

        # Crossing density relative to uniform distribution
        nbins = int(1 / phase_step)
        crosshist = np.histogram(data, bins=np.arange(0, nbins + 1, 1) / nbins)

        # Bins with no threshold crossings
        nocrossidxs = np.argwhere(crosshist[0] == 0)[:, 0]

        # print("nbins =", nbins)
        # print(crosshist)
        # print(nocrossidxs)
        # print(get_largest_group(nocrossidxs, nbins))

        # Groups of contiguous bins with no crossings ordered by size,
        # largest first.
        groups = self.get_contiguous(nocrossidxs, nbins)
        if groups is None or len(groups) == 0:
            return False, None, None, None, None, int(c1.duration * f0)
        sgroups = sorted(groups, key=lambda x: x[2], reverse=True)
        
        for lg in sgroups:
            # If lg[0] == 0, we want to look at max lg[nbins-1]
            # If nbins = 10, (lg[0] + nbins - 1) % (nbins) == 9 % 10 == 9
            # lg[0]
            #  0              9 % 10  == 9
            #  1              10 % 10 == 0
            #  2              11 % 10 == 1
            left = data[data < ((lg[0] + nbins - 1) % nbins + 1) / nbins].max()

            # The 'right' side might wrap around, thus the modulo operation.
            # If nbins == 10, lg[1] == 9, then (lg[1] + 1) % nbins == 0.
            right = data[data > ((lg[1] + 1) % nbins) / nbins].min()

            # print(left, right)
            if left > right:
                right += 1

            offset = (left + right) / 2  # Horizontal center
            # offset = lg[0] / nbins
            # print("offset = {:.3f}".format(offset))
            nsamples = int(c1.duration * f0)
            ts = offset / f0 + np.arange(1, nsamples, 1) / f0
            vs = c1.at(ts)

            # PRBS Check
            bits = vs > threshold
            # print("".join(['1' if b else '0' for b in bits]))
            errors = Curve.prbs_check(bits)
            
            if sum(errors) == 0:
                bottom = vs[vs < threshold].max()
                top = vs[vs > threshold].min()
                return True, left, right % 1, bottom, top, nsamples

        return False, None, None, None, None, int(c1.duration * f0)

    def eye_width_height(self, start, f0, threshold=0.0, phase_step=0.05):
        """
        Calculates the width and the height of the data's eye diagram.

        This is a simple wrapper of Curve.eye_box().

        :param start: X-axis value at which to start the computation.
        :param f0: Symbol rate.
        :param threshold: Y-axis value above which it is interpreted as a
                          logic 1, otherwise as logic 0.
        :param phase_step: See Curve.eye_box()
        :return: (width, height, nsamples) in the original units of the curve.
                 If it fails, (None, None, nsamples)
        """

        if self.y.dtype == np.dtype(np.complex128):
            raise ComplexCurveError('Complex-valued Curve does not support eye_width_height()')

        success, left, right, bottom, top, nbits = self.eye_box(
            start, f0, threshold=threshold, phase_step=phase_step
        )

        if success:

            width = right - left if right > left else right + 1 - left
            width *= 1 / f0
            height = top - bottom

            return width, height, nbits

        else:

            return None, None, nbits

    def plot(self, ax=None, xscale=1.0, yscale=1.0, **kwargs):
        """
        Plots the curve using matplotlib.pyplot.plot().

        :param ax: If provided, plots on the given axes. Otherwise,
                   uses the current axes.
        :param xscale: Multiplier for x-axis values.
        :param yscale: Multiplier for y-axis values.
        :param kwargs: Additional keyword arguments passed to
                       matplotlib.pyplot.plot().
        :return: See Matplotlib.pyplot.plot and Matplotlib.axes.Axes.plot
        """
        if ax is not None:
            return ax.plot(self.x * xscale, self.y * yscale, **kwargs)
        else:
            return plot(self.x * xscale, self.y * yscale, **kwargs)

    def semilogx(self, ax=None, xscale=1.0, yscale=1.0, **kwargs):
        """
        Plots the curve using matplotlib.pyplot.semilogx().

        :param ax: If provided, plots on the given axes. Otherwise,
                   uses the current axes.
        :param xscale: Multiplier for x-axis values.
        :param yscale: Multiplier for y-axis values.
        :param kwargs: Additional keyword arguments passed to
                       matplotlib.pyplot.plot().
        :return: See Matplotlib.pyplot.plot and Matplotlib.axes.Axes.plot
        """
        if ax is not None:
            return ax.semilogx(self.x * xscale, self.y * yscale, **kwargs)
        else:
            return semilogx(self.x * xscale, self.y * yscale, **kwargs)

    def semilogy(self, ax=None, xscale=1.0, yscale=1.0, **kwargs):
        """
        Plots the curve using matplotlib.pyplot.semilogy().

        :param ax: If provided, plots on the given axes. Otherwise,
                   uses the current axes.
        :param xscale: Multiplier for x-axis values.
        :param yscale: Multiplier for y-axis values.
        :param kwargs: Additional keyword arguments passed to
                       matplotlib.pyplot.plot().
        :return: See Matplotlib.pyplot.plot and Matplotlib.axes.Axes.plot
        """
        if ax is not None:
            return ax.semilogy(self.x * xscale, self.y * yscale, **kwargs)
        else:
            return semilogy(self.x * xscale, self.y * yscale, **kwargs)

    def loglog(self, ax=None, xscale=1.0, yscale=1.0, **kwargs):
        """
        Plots the curve using matplotlib.pyplot.loglog().

        :param ax: If provided, plots on the given axes. Otherwise,
                   uses the current axes.
        :param xscale: Multiplier for x-axis values.
        :param yscale: Multiplier for y-axis values.
        :param kwargs: Additional keyword arguments passed to
                       matplotlib.pyplot.plot().
        :return: See Matplotlib.pyplot.plot and Matplotlib.axes.Axes.plot
        """
        if ax is not None:
            return ax.loglog(self.x * xscale, self.y * yscale, **kwargs)
        else:
            return loglog(self.x * xscale, self.y * yscale, **kwargs)

    def max(self):
        """
        (x, y) pair where max(y) occurs.

        :return: (x, y) pair where max(y) occurs.
        """
        idxmax = np.argmax(self.y)
        return self.x[idxmax], self.y[idxmax]

    def min(self):
        """
        (x, y) pair where min(y) occurs.

        :return: (x, y) pair where min(y) occurs.
        """
        idxmin = np.argmin(self.y)
        return self.x[idxmin], self.y[idxmin]

    def copy(self):
        """
        Create a copy of this object. The copy will have no references to this object.

        :return: Copy of this Curve.
        """

        return Curve(np.copy(self.x), np.copy(self.y), dtype=(self.x.dtype, self.y.dtype))

    def __add__(self, other):
        """
        Computes self + other.

        :param other: Scalar or Curve.
        :return: Resulting Curve.
        """

        # Sum to a scalar
        if isinstance(other, (int, float, complex, np.float32, np.float64)):
            # c = Curve(np.copy(self.data))
            c = self.copy()
            c.y += other
            return c

        # Sum to a curve
        # NOTE: The result uses the x-axis points of the curve
        # that has the most points. This is rather simplistic. We may want
        # to consider things such as the range of x values, and the
        # density of x values.
        if isinstance(other, Curve):
            if len(self.x) >= len(other.x):
                # Resample other.
                samples = other.at(self.x)
                c = self.copy()
            else:
                # Resample this on.
                samples = self.at(other.x)
                c = other.copy()
            c.y += samples
            return c

        # raise TypeError("Operator '+' not supported for type 'Curve'"
        #                 " and '{}'".format(type(other)))
        return NotImplemented

    def __mul__(self, other):
        """
        Computes self * other

        :param other: Scalar or Curve.
        :return: Resulting Curve.
        """

        # Multiply by scalar
        if isinstance(other, (int, float, complex)):
            c = self.copy()
            c.y *= other
            return c

        # Multiply by a curve
        # NOTE: The re-sampling approach is the same as in __add__.
        if isinstance(other, Curve):
            if len(self.x) >= len(other.x):
                # Resample other.
                samples = other.at(self.x)
                c = self.copy()
            else:
                # Resample this on.
                samples = self.at(other.x)
                c = other.copy()
            c.y *= samples
            return c

        return NotImplemented

    def __div__(self, other):
        """
        Computes self / other

        :param other: Scalar or Curve.
        :return: Resulting Curve.
        """

        # Divide by a scaler
        if isinstance(other, (int, float, complex)):
            # c = Curve(np.copy(self.data))
            c = self.copy()
            c.y /= other
            return c

        # Divide by a curve
        # NOTE: The re-sampling approach is the same as in __add__.
        if isinstance(other, Curve):
            if len(self.x) >= len(other.x):
                # Resample other.
                samples = other.at(self.x)
                c = self.copy()
            else:
                # Resample this on.
                samples = self.at(other.x)
                c = other.copy()
            c.y /= samples
            return c

        return NotImplemented

    def __pow__(self, other):
        """
        This curve to the power of other.

        :param other: Scalar
        :return: Resulting curve.
        """

        c = self.copy()

        # Power by a scalar
        if isinstance(other, (int, float, complex)):
            c.y = c.y**other
            return c

        return NotImplemented

    def __rpow__(self, other):
        """
        Other to the power of this curve.

        :param other:
        :return:
        """

        c = self.copy()

        # Power by a scalar
        if isinstance(other, (int, float, complex)):
            c.y = other ** c.y
            return c

    def __neg__(self):

        c = self.copy()
        c.y *= -1

        return c

    def __rdiv__(self, other):
        """
        Compute other / self

        :param other: Scalar
        :return: Resulting Curve.
        """

        c = self.copy()

        if isinstance(other, (int, float, complex)):
            c.y = other / c.y
            return c

        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):

        # other - curve
        # -(curve - other)
        return -self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __repr__(self):
        out = ""
        if len(self.x) > 7:
            out += "x:"
            for i in range(5):
                out += "  {:6.2e}".format(self.x[i])
            out += "  ...  {:6.2e}\n".format(self.x[-1])
            out += "y:"
            for i in range(5):
                out += "  {:6.2e}".format(self.y[i])
            out += "  ...  {:6.2e}".format(self.y[-1])

        else:
            out += "x:"
            for i in range(len(self.x)):
                out += "  {:6.2e}".format(self.x[i])
            out += "\ny:"
            for i in range(len(self.x)):
                out += "  {:6.2e}".format(self.y[i])

        return out

    def __getattr__(self, name):
        """
        Provides the class-defined attributes and adds attributes that
        match to some Numpy function calls.

        :param name:
        :return:
        """

        # Built-ins
        if name in self.__dict__:
            return self.__dict__[name]

        # Numpy methods
        if name in ["abs", "real", "imag", "sin", "cos", "angle", "cumprod",
                    "cumsum", "arccos", "arccosh", "arcsin", "arcsinh",
                    "arctan", "arctanh", "tan", "tanh", "log", "log2", "log10",
                    "exp"]:
            def npfunc():
                return self.numpy_on_yaxis(name)
            return npfunc

    def numpy_on_yaxis(self, npfunc):
        """
        Applies a Numpy function to the y-axis data.

        :param npfunc: Name of the Numpy function.
        :return: A new Curve with the same x-axis as this object and the
            and the result from the Numpy call as the y-axis.
        """
        c = self.copy()
        c.y = getattr(np, npfunc)(c.y)
        return c

    def filter(self, fc, order=8, kind='low', fs=None):
        """
        Applies a digital filter to the curve. The filter is either
        a high-pass or a low-pass filter as specified by the kind
        parameter. The cut-off frequency is fc in Hz.
        The curve is first resampled with a uniform sample spacing,
        with a period of 1/fs.

        At this time, the filter is a Butterworth filter and it is
        applied in both directions in order to get 0 phase shift.

        :param fc: 3-dB frequency in Hertz.
        :param order: Order of the filter.
        :param kind: 'low' or 'high' pass.
        :param fs: (Re-) sampling frequency. If not supplied,
            it is computed as 20 * fc.
        :return: Filtered Curve.
        """

        fs = fs or 20 * fc
        fnyq = fs / 2

        # Resampling
        x = np.arange(self.x[0], self.x[-1], step=1 / fs)
        yresamp = self.at(x)

        # Filter and filtering
        b, a = butter(order, fc / fnyq, btype=kind, output='ba')
        yfilt = filtfilt(b, a, yresamp)

        return Curve(x, yfilt)


class EyeDiagram(UserList):

    def __init__(self, curve, start, period):
        """
        Creates an eye diagram from a Curve object by splitting it
        into a list of Curves, each with a duration of period.

        :param curve:
        :type curve: Curve
        :param start:
        :param period:
        """
        
        super().__init__()
        
        self.data = curve.eye(start=start, period=period)

    def plot(self, **kwargs):
        """
        Plots the eye diagram.
        
        :param kwargs: See Curve.
        """
        for seg in self.data:
            seg.plot(**kwargs)


class Marker:
    """
    Single-point marker on a Curve.
    """

    def __init__(self, x=None, y=None, coords_offset=(0, 0), plotopts=None):
        self.x = x
        self.y = y

        self.coords_offset = coords_offset

        self.plotopts = {
            'marker': 'x',
            'markercolor': 'black',
            'coordfmt': "x={:.2g}\ny={:.2g}"
        }
        self.plotopts.update(plotopts or {})

    def plot(self, point=True, coords=True, ax=None):
        """
        Display the marker on a MatPlotLib Axes.
        """

        ax = ax or gca()

        if point:
            ax.plot(self.x, self.y,
                    self.plotopts['marker'],
                    color=self.plotopts['markercolor'])

        if coords:
            ax.annotate(self.plotopts['coordfmt'].format(self.x, self.y),
                        xy=(self.x, self.y),
                        xytext=(self.x + self.coords_offset[0],
                                self.y + self.coords_offset[1]))


class DeltaMaker:
    """
    Relative position between two Markers.
    """

    def __init__(self, m1, m2, plotopts=None):
        """

        :param m1: First Marker
        :param m2: Second Marker
        """

        self.m1 = m1
        self.m2 = m2

        self.dx = m2.x - m1.x
        self.dy = m2.y - m1.y

        self.plotopts = {

        }
        self.plotopts.update(plotopts or {})

    def plot(self):
        """
        Display the DeltaMarker on a MatPlotLib Axes.
        """

        # Horizontal arrow
        annotate("", xy=(self.m1.x, self.m1.y),
                 xytext=(self.m2.x, self.m1.y),
                 arrowprops=dict(arrowstyle="<->",
                                 # shrink=0,
                                 color='gray',
                                 linestyle="--"
                                 ))

        # Vertical Arrow
        annotate("", xy=(self.m2.x, self.m1.y),
                 xytext=(self.m2.x, self.m2.y),
                 arrowprops=dict(arrowstyle="<->",
                                 # shrink=0,
                                 color='gray',
                                 linestyle="--"
                                 ))

        # Calculate margins
        ax = gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xd = xlim[1] - xlim[0]
        yd = ylim[1] - ylim[0]
        xgap = xd * 0.03
        ygap = yd * 0.03

        # Horizontal delta
        annotate("dx={:.2g}".format(self.dx),
                 xy=(self.m1.x + self.dx / 2, self.m1.y + ygap),
                 horizontalalignment='center')

        # Vertical delta
        annotate("dy={:.2g}".format(self.dy),
                 xy=(self.m2.x + ygap, self.m2.y - self.dy / 2))


class VerticalSlice:
    """
    Slice of a list of curves at a given value of their X-axis.
    """

    def __init__(self, curves, x, plotopts=None):
        """
        Set of markers with a common x value. The markers are
        generated at the y value of each curve for the given x.

        The `markers` property contains the list of markers
        curresponding to each curve in `curves`.

        :param curves: List/tuple of Curves.
        :param x: X value.
        """

        self.x = x
        self.curves = curves
        self.y = [c.at(x) for c in curves]
        self.markers = [Marker(x, y) for y in self.y]

        self.plotopts = {
            'slice': True,  # Draw a slice across the axes
            'slicelinefmt': '--',
            'slicelinecolor': 'gray',
            'slicelinewidth': 1.2,
            'yvalfmt': "y={:.2g}",
            'dyvalfmt': "dy={:.2g}"
        }
        self.plotopts.update(plotopts or {})

    def plot(self, ax=None):
        """
        Plots the VerticalSlice on MatPlotLib Axes.
        """

        # Make ax a list.
        if ax is None:
            ax = [gca()]
        elif isinstance(ax, (list, tuple)):
            pass
        else:
            ax = [ax]

        # Wring number of axes.
        if len(ax) > 1 and len(ax) != len(self.curves):
            raise RuntimeError('Number of axes must be 1 or must match the number of curves.')

        # Expand the ax list to the number of curves.
        if len(ax) == 1:
            ax *= len(self.curves)

        # Group by axes
        for axi, mrks in groupby(zip(ax, self.markers, self.y), lambda x: x[0]):
            # print(f'axi: {axi}, mrks: {mrks}')

            xlim = axi.get_xlim()
            ylim = axi.get_ylim()
            xd = xlim[1] - xlim[0]
            yd = ylim[1] - ylim[0]
            xgap = xd * 0.03
            ygap = yd * 0.03

            # Slice the whole axes
            if self.plotopts['slice']:
                axi.plot([self.x, self.x], [ylim[0], ylim[1]],
                         self.plotopts['slicelinefmt'],
                         color=self.plotopts['slicelinecolor'],
                         linewidth=self.plotopts['slicelinewidth'])

            mrks = sorted(mrks, key=lambda x: x[2])

            for _, m, y in mrks:
                # print(f"m: {m}")
                m.plot(coords=False, ax=axi)  # Plot the point marker
                axi.annotate(self.plotopts['yvalfmt'].format(m.y),  # Annotate the y value.
                             xy=(self.x + 2 * ygap, m.y))

            for i, (_, m, y) in enumerate(mrks[:-1]):
                # Arrow between markers.
                axi.annotate("", xy=(self.x, mrks[i][2]),
                             xytext=(self.x, mrks[i + 1][2]),
                             arrowprops=dict(arrowstyle="<->",
                                             # shrink=0,
                                             color='gray',
                                             linestyle="-"
                                             )
                             )
                # Distance between markers.
                axi.annotate(self.plotopts['dyvalfmt'].format(abs(mrks[i][2] - mrks[i + 1][2])),
                             xy=(self.x + 2 * ygap, mrks[i][2] / 2 + mrks[i + 1][2] / 2))
