import numpy as np
from numba import jit, prange


# pure python

# defining pure standard deviation functions


def didatic_stdev(img):
    stdev = np.sqrt(np.sum((img-np.mean(img))**2))
    return(stdev)


def better_stdev(img):
    n2 = img.size
    totalsum = img.sum()
    sqvalssum = (img**2).sum()
    stdev = np.sqrt(sqvalssum/n2 - (totalsum/n2)**2)
    return(stdev)


# function to create general 2D sliding window functions


def sw_func(img, wr, func):
    ws = 1+2*wr
    funcimg = np.empty(img.shape)  # change to empty
    img_pad = np.pad(img, wr)
    for y in range(0, funcimg.shape[0]):
        for x in range(0, funcimg.shape[1]):
            subimg = img_pad[y:y+ws,
                             x:x+ws]
            funcimg[y, x] = func(subimg)
    return(funcimg)


# creating 2D sliding window functions


def numpy_swsd(img, wr): return sw_func(img, wr=1, func=np.std)


def didatic_swsd(img, wr): return sw_func(img, wr=1, func=didatic_stdev)


def better_swsd(img, wr): return sw_func(img, wr=1, func=better_stdev)


def optimized_swsd(img, wr=1):
    stdevimg = np.empty(img.shape)  # change to empty
    img = np.pad(img, wr)

    ws = 1+2*wr
    n2 = ws**2

    for y in range(0, stdevimg.shape[0]):
        x = 0
        # first iteration
        subimg = img[y:y+ws,
                     x:x+ws]

        oldvalssum = subimg[:, 0].sum()         # used in next iteration
        totalsum = oldvalssum + subimg[:, 1:].sum()

        oldsqvalssum = (subimg[:, 0]**2).sum()  # used in next iteration
        sqvalssum = oldsqvalssum + (subimg[:, 1:]**2).sum()

        #stdev = np.sqrt( (n2*sqvalssum-totalsum**2) / n2**2)
        stdevimg[y, x] = np.sqrt(sqvalssum/n2-(totalsum/n2)**2)

        # next iterations
        for x in range(1, stdevimg.shape[1]):
            subimg = img[y:y+ws,
                         x:x+ws]
            newvalssum = subimg[:, ws-1].sum()
            totalsum += newvalssum - oldvalssum
            newsqvalssum = (subimg[:, ws-1]**2).sum()
            sqvalssum += newsqvalssum - oldsqvalssum
            #stdev = np.sqrt( (n2*sqvalssum-totalsum**2) / n2**2)
            stdevimg[y, x] = np.sqrt(sqvalssum/n2-(totalsum/n2)**2)

            oldvalssum = subimg[:, 0].sum()        # used in next iteration
            oldsqvalssum = (subimg[:, 0]**2).sum()  # used in next iteration

    return(stdevimg)


# numba versions

# defining pure standard deviation functions

@jit(parallel=True, nopython=True, boundscheck=False,)
def numba_didatic_stdev(img):
    stdev = np.sqrt(np.sum((img-np.mean(img))**2))
    return(stdev)


@jit(parallel=True, nopython=True, boundscheck=False,)
def numba_better_stdev(img):
    n2 = img.size
    totalsum = img.sum()
    sqvalssum = (img**2).sum()
    stdev = np.sqrt(sqvalssum/n2 - (totalsum/n2)**2)
    return(stdev)


# creating 2D sliding window functions

@jit(parallel=True, nopython=True, boundscheck=False,)
def pad(img, wr, val):
    padded_img = np.empty((img.shape[0]+2*wr, img.shape[1]+2*wr))
    padded_img[:wr, :] = val
    padded_img[-wr:, :] = val
    padded_img[wr:-wr, :wr] = val
    padded_img[wr:-wr, -wr:] = val
    padded_img[wr:-wr, wr:-wr] = img
    return(padded_img)


@jit(parallel=True, nopython=True, boundscheck=False,)
def numba_numpy_swsd(img, wr):
    ws = 1+2*wr
    funcimg = np.zeros(img.shape)  # change to empty
    img_pad = pad(img, wr, 0)
    for y in prange(0, funcimg.shape[0]):
        for x in prange(0, funcimg.shape[1]):
            subimg = img_pad[y:y+ws,
                             x:x+ws]
            funcimg[y, x] = np.std(subimg)
    return(funcimg)


@jit(parallel=True, nopython=True, boundscheck=False,)
def numba_didatic_swsd(img, wr):
    ws = 1+2*wr
    funcimg = np.zeros(img.shape)  # change to empty
    img_pad = pad(img, wr, 0)
    for y in prange(0, funcimg.shape[0]):
        for x in prange(0, funcimg.shape[1]):
            subimg = img_pad[y:y+ws,
                             x:x+ws]
            funcimg[y, x] = numba_didatic_stdev(subimg)
    return(funcimg)


@jit(parallel=True, nopython=True, boundscheck=False,)
def numba_better_swsd(img, wr):
    ws = 1+2*wr
    funcimg = np.zeros(img.shape)  # change to empty
    img_pad = pad(img, wr, 0)
    for y in range(0, funcimg.shape[0]):
        for x in range(0, funcimg.shape[1]):
            subimg = img_pad[y:y+ws,
                             x:x+ws]
            funcimg[y, x] = numba_better_stdev(subimg)
    return(funcimg)


@jit(parallel=True, nopython=True, boundscheck=False,)
def numba_optimized_swsd(img, wr=1):
    stdevimg = np.empty(img.shape)  # change to empty
    img_pad = pad(img, wr, 0)

    ws = 1+2*wr
    n2 = ws**2

    for y in range(0, stdevimg.shape[0]):
        x = 0
        # first iteration
        subimg = img_pad[y:y+ws,
                         x:x+ws]

        oldvalssum = subimg[:, 0].sum()        # used in next iteration
        totalsum = subimg[:, 1:].sum() + oldvalssum

        oldsqvalssum = (subimg[:, 0]**2).sum()  # used in next iteration
        sqvalssum = (subimg[:, 1:]**2).sum() + oldsqvalssum

        stdevimg[y, x] = np.sqrt((n2*sqvalssum-totalsum**2) / n2**2)

        # next iterations
        for x in range(1, stdevimg.shape[1]):
            subimg = img_pad[y:y+ws,
                             x:x+ws]
            newvalssum = subimg[:, ws-1].sum()
            totalsum += newvalssum - oldvalssum
            newsqvalssum = (subimg[:, ws-1]**2).sum()
            sqvalssum += newsqvalssum - oldsqvalssum
            stdevimg[y, x] = np.sqrt((n2*sqvalssum-totalsum**2) / n2**2)

            oldvalssum = subimg[:, 0].sum()        # used in next iteration
            oldsqvalssum = (subimg[:, 0]**2).sum()  # used in next iteration
    return(stdevimg)


@jit(parallel=True, nopython=True, boundscheck=False,)
def numba_optimized_parallel_swsd(img, wr=1):
    stdevimg = np.empty(img.shape)  # change to empty
    img_pad = pad(img, wr, 0)

    ws = 1+2*wr
    n2 = ws**2

    for y in prange(0, stdevimg.shape[0]):
        x = 0
        # first iteration
        subimg = img_pad[y:y+ws,
                         x:x+ws]

        oldvalssum = subimg[:, 0].sum()        # used in next iteration
        totalsum = subimg[:, 1:].sum() + oldvalssum

        oldsqvalssum = (subimg[:, 0]**2).sum()  # used in next iteration
        sqvalssum = (subimg[:, 1:]**2).sum() + oldsqvalssum

        stdevimg[y, x] = np.sqrt((n2*sqvalssum-totalsum**2) / n2**2)

        # next iterations
        for x in range(1, stdevimg.shape[1]):
            subimg = img_pad[y:y+ws,
                             x:x+ws]
            newvalssum = subimg[:, ws-1].sum()
            totalsum += newvalssum - oldvalssum
            newsqvalssum = (subimg[:, ws-1]**2).sum()
            sqvalssum += newsqvalssum - oldsqvalssum
            stdevimg[y, x] = np.sqrt((n2*sqvalssum-totalsum**2) / n2**2)

            oldvalssum = subimg[:, 0].sum()        # used in next iteration
            oldsqvalssum = (subimg[:, 0]**2).sum()  # used in next iteration
    return(stdevimg)


if __name__ == '__main__':
    # gettings useeful functionalities
    from utils import numba_get_stats as get_stats
    from sys import argv
    #import matplotlib.pyplot as plt
    #import seaborn as sns
    # do not mess here
    img_comp = np.array([[1]])

    def lone_test(functions, N=100, wr=2, times=3):
        # creating test arrays
        img = np.arange(N**2).reshape((N, N))

        # defining function parameters
        ws = 2*wr+1

        # velocity test
        for i, function in enumerate(functions):
            name = function.__name__
            function_stats = get_stats(function, times, return_val=True)
            print(f'Statistics for function: {name}(N={N}, ws={ws})')
            avg, std, minv, maxv, resp = function_stats(img, wr)
            rel_std = 100*std/np.abs(avg)  # %
            amp = maxv - minv
            print(
                f'  avg±std:\t{avg:.4g}±{std:.4g} s\t(rel_std: {rel_std:.2f}%)')
            print(f'  amp: {amp:4g} = [{minv:.4g}, {maxv:.4g}] s')
            print(f'  function run {times} times.', end='\n\n')

    def runtime_results(functions, Ns=[5], wss=[3, 5, 7], times=3):
        def print_stats(*stats):
            if len(stats) == 0:
                print('name\tN\tws\tavg\tstd\trel_std\tminv\tmaxv\tamp')
            else:
                print('\t'.join([str(i) for i in stats]))
        print_stats()
        for function in functions:
            name = function.__name__
            function_stats = get_stats(function, times,)
            for N in Ns:
                img = np.random.rand(N, N)
                for ws in wss:
                    wr = ws//2+1
                    avg, std, minv, maxv = function_stats(img, wr)
                    rel_std = 100*std/np.abs(avg)  # %
                    amp = maxv - minv
                    print_stats(name, N, ws, avg, std,
                                rel_std, minv, maxv, amp)

    functions = [
        numpy_swsd,
        didatic_swsd,
        better_swsd,
        optimized_swsd,
        numba_numpy_swsd,
        numba_didatic_swsd,
        numba_better_swsd,
        numba_optimized_swsd,
        numba_optimized_parallel_swsd,
    ]

    #lone_test(functions, N=100, wr=2, times=3)
    runtime_results(functions,
                    # Ns=[300],
                    # wss=[3, 5, 7, 9],
                    Ns=[100, 200, 300, 400,]    # 800, 1600],
                    wss=[2*i+1 for i in range(30)],
                    times=3)
