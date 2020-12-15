import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from tools import lifter
import tools


def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.
    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal
    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)


def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.
    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs
    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspec_ = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspec_, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------


def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.
    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    x = samples.shape[0]  #
    #print(x)
    y = x // winlen * 2 - 1
    segment = np.zeros((y, winlen))
    for i in range(y):
        segment[i, :] = samples[i * winshift:i * winshift + winlen]
    return segment


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.
    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)
    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    b = np.array([1, -p])
    a = 1
    N = input.shape[0]
    M = input.shape[1]
    x = np.zeros((N, M))
    for i in range(N):
        x[i, :] = signal.lfilter(b, a, input[i, :])
    return x


def windowing(input):
    """
    Applies hamming window to the input frames.
    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    N = input.shape[0]
    M = input.shape[1]
    x = np.zeros((N, M))
    window = signal.hamming(M, sym=False)
    for i in range(N):
        x[i, :] = input[i, :] * window

    return x


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT
    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    N = len(input)
    M = nfft
    x = np.zeros((N, M))
    x = fftpack.fft(input, M)
    x = abs(x)
    x = x * x

    return x


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum
    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    N = len(input)  # 92frames
    Mel = tools.trfbank(samplingrate, len(input[0]))  # 40filters*512
    M = Mel.shape[0]
    # plot the filters in linear frequency scale x=k*512/fs(20000)
    #plt.figure()
    x = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            x[i, j] = np.log(np.sum(input[i, :] * Mel[j, :]))
    return x


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform
    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    N = len(input)
    M = nceps
    x = np.zeros((N, M))

    y = fftpack.dct(input, type=2)  # 91*40
    # y = fftpack.dct(input, type=2, n=nceps) #y=91*13  n= length of transform
    x = y[:, :13]
    # lx = lifter(x)

    return x  # , lx


def dtw(x, y, dist):
    """Dynamic Time Warping.
    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))
    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD
    Note that you only need to define the first output for this exercise.
    """
    N = len(x)
    M = len(y)
    LD = np.zeros((N,M))
    AD = np.zeros((N,M))
    for a in range(N):
        for b in range(M):
            LD[a,:b] = dist(x[a],y[b])
    AD[0,:0] = LD[0,:0]
    for a in range(1,N):
        AD[a,:0] = LD[a,:0] + AD[a-1,:0]
    for b in range(1,M):
        AD[0,:b] = LD[0,:b] + AD[0,:b-1]
    for a in range(1,N):
        for b in range(1,M):
            AD[a,:b] = LD[a,:b] + min(AD[a-1,:b],AD[a-1,:b-1],AD[a,:b-1])

    b = AD[N-1,M-1] / (M+N)
    return b


def dist(x,y):
    if x==y:
        return 1
    else:
        return 0