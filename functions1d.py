import numpy as np


def didatic_mwsd(arr, n=3):
    _arr_ = np.pad(arr, n//2)
    mwsd = np.empty_like(arr)
    for i in range(len(mwsd)):
        subarr = _arr_[1+i-n//2:1+i-n//2+n]
        mwsd[i] = np.std(subarr)
    return(mwsd)


def optimized_mwsd(arr, n=3):
    _arr_ = np.pad(arr, n//2)
    mwsd = np.empty_like(arr)

    # first window
    window = _arr_[0:n]
    totalsum = np.sum(window)                               # n-1
    sqvalssum = np.sum(window**2)                           # n + ne-1
    mwsd[0] = np.sqrt(sqvalssum/n-(totalsum/n)**2)         # 5

    # next windows
    for i in range(1, arr.shape[0]):                        # *(N-1)
        oldval = window[0]
        oldsqval = oldval**2                                # 1
        window = _arr_[i:i+n]
        newval = window[n-1]
        totalsum += newval - oldval                         # 2
        newsqval = newval**2                                # 1
        sqvalssum += newsqval - oldsqval                    # 2
        mwsd[i] = np.sqrt(sqvalssum/n-(totalsum/n)**2)     # 5
    return(mwsd)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = 3
    arr = np.random.rand(10)

    fig, ax = plt.subplots(2, 1)
    sns.heatmap([optimized_mwsd(arr, 3)], ax=ax[0], annot=True)
    sns.heatmap([didatic_mwsd(arr,   3)], ax=ax[1], annot=True)

    plt.show()
