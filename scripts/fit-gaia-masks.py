"""
Fit a polynomial to log10(radius) vs mag.  The radius is from the masks for
actual saturated stars used in DES.  The mag is gaia g
"""
import numpy as np
import fitsio
import hickory
import esutil as eu
from esutil.numpy_util import between


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stars-file', help='input stars file path', required=True,
    )
    parser.add_argument('--pdf', help='output pdf file path', required=True)
    return parser.parse_args()


def main():
    args = get_args()
    order = 2

    t = fitsio.read(args.stars_file, lower=True)
    radius_factor = 2.0
    t['radius'] /= radius_factor
    logr = np.log10(t['radius'])

    ylim = [np.log10(1/radius_factor), np.log10(400/radius_factor)]

    tab = hickory.Table(
        figsize=(12, 10),
        nrows=2,
        ncols=2,
    )

    xlabel = 'GAIA G [mag]'
    magmin = 5
    magmax = [16, 17, 17]
    plot_magmax = 19
    plys = {}
    for iband, band in enumerate(['r', 'i', 'z']):
        print(band)

        plt = tab.axes[iband]
        plt.set(
            xlabel=xlabel,
            ylabel='log10 Mask radius %s-band [arcsec]' % band,
            xlim=[magmin, plot_magmax],
            ylim=ylim,
        )
        w, = np.where(
            (t['band'] == band) &
            between(t['mag'], magmin, magmax[iband]) &
            between(logr, ylim[0], ylim[1])
        )
        print('found:', w.size)

        plt.hexbin(
            t['mag'][w], logr[w],
            bins='log',
            cmap='Greens',
        )

        bs = eu.stat.Binner(t['mag'][w], logr[w])
        bs.dohist(nbin=30, calc_stats=True)

        coeffs = np.polyfit(bs['xmean'], bs['ymean'], order)
        ply = np.poly1d(coeffs)
        print(repr(coeffs))
        plt.errorbar(
            bs['xmean'], bs['ymean'], yerr=bs['yerr'], color='black', zorder=1,
        )

        xvals = np.linspace(magmin, plot_magmax)
        plt.curve(xvals, ply(xvals), color='grey', zorder=2)

        plys[band] = ply

    tab[1, 1].set(xlabel=xlabel, ylabel='poly ratio')

    rp = 10.0**(plys['r'](xvals))
    ip = 10.0**(plys['i'](xvals))
    zp = 10.0**(plys['z'](xvals))

    tab[1, 1].curve(xvals, ip/rp, label='i poly/r poly')
    tab[1, 1].curve(xvals, zp/rp, label='z poly/r poly')
    tab[1, 1].legend()

    print('writing:', args.pdf)
    tab.savefig(args.pdf)


if __name__ == '__main__':
    main()
