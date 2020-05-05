import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def sw_func(img, wr, func,
            padmode='constant'):
    ws = 1+2*wr
    funcimg = np.zeros(img.shape)  # change to empty
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

        oldvalssum = subimg[:, 0].sum()        # used in next iteration
        totalsum = subimg[:, 1:].sum() + oldvalssum

        oldsqvalssum = (subimg[:, 0]**2).sum()  # used in next iteration
        sqvalssum = (subimg[:, 1:]**2).sum() + oldsqvalssum

        #stdev = np.sqrt( (n2*sqvalssum-totalsum**2) / n2**2)
        stdevimg[y, x] = np.sqrt((n2*sqvalssum-totalsum**2) / n2**2)

        # next iterations
        for x in range(1, stdevimg.shape[1]):
            subimg = img[y:y+ws,
                         x:x+ws]
            newvalssum = subimg[:, ws-1].sum()
            totalsum += newvalssum - oldvalssum
            newsqvalssum = (subimg[:, ws-1]**2).sum()
            sqvalssum += newsqvalssum - oldsqvalssum
            #stdev = np.sqrt( (n2*sqvalssum-totalsum**2) / n2**2)
            stdevimg[y, x] = np.sqrt((n2*sqvalssum-totalsum**2) / n2**2)

            oldvalssum = subimg[:, 0].sum()        # used in next iteration
            oldsqvalssum = (subimg[:, 0]**2).sum()  # used in next iteration

    return(stdevimg)


if __name__ == '__main__':
    # gettings useeful functionalities
    from utils import get_stats

    # creating test arrays
    N = 100
    img = np.arange(N**2).reshape((N, N))
    # do not mess here
    img_comp = np.array([[0.]])

    # number of samples for statistics
    times = 2
    # functions to be used
    functions = [
        numpy_swsd,
        didatic_swsd,
        better_swsd,
        optimized_swsd
    ]

    # defining function parameters
    wr = 2
    ws = 2*wr+1

    ## visual acuracy test
    #fig, axes = plt.subplots(3, 1)
    #sns_heatmap_annot = False

    # velocity test
    for i, function in enumerate(functions):
        name = function.__name__
        function_stats = get_stats(function, times, return_val=True)
        print(f'Statistics for function: {name}(N={N}, ws={ws})')
        avg, std, minv, maxv, resp = function_stats(img, wr)
        rel_std = 100*std/np.abs(avg)  # %
        amp = maxv - minv
        print(f'    avg±std:	{avg:.4g}±{std:.4g} s	(rel_std: {rel_std:.2f}%)')
        print(f'    amp: {amp:4g} = [{minv:.4g}, {maxv:.4g}] s')
        print(f'    function run {times} times.')
        print()

    ## acuracy test
    # sns.heatmap(resp[-1])
    # plt.show()
    # sns.heatmap(stdevimg, ax=axes[0], annot=sns_heatmap_annot)
    # plt.show()
