"""
heavily simplified version of the original simple tqdm from

https://github.com/noamraph/tqdm
"""
__all__ = ['PBar', 'prange']

import sys
import time


def PBar(iterable, desc='', total=None, leave=True, file=sys.stdout,
         mininterval=0.5, miniters=1, n_bars=20, pad=None, flush=False):
    """
    Get an iterable object, and return an iterator which acts exactly like the
    iterable, but prints a progress meter and updates it every time a value is
    requested.

    parameters
    ----------
    desc: string, optional
        An optional short string, describing the progress, that is added
        in the beginning of the line.
    total: int, optional
        Optional number of expected iterations. If not given,
        len(iterable) is used if it is defined.
    file: file-like object, optional
        A file-like object to output the progress message to. Default
        stderr
    leave: bool, optional
        If True, leave the remaining text from the progress.  If False,
        delete it.
    mininterval: float, optional
        default 0.5
    miniters: int, optional
        default 1

        If less than mininterval seconds or miniters iterations have passed
        since the last progress meter update, it is not updated again.
    n_bars: int, optional
        The width of the bar.
    pad: int, optional
        The number of characters to pad integers to. Default of None does
        no padding.
    flush: bool, optional
        If True, each progress update is flushed with a newline instead of
        overwriting the old bar. Default is False.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    prefix = desc+': ' if desc else ''

    sp = StatusPrinter(file, flush=flush)
    sp.print_status(prefix + format_meter(0, total, 0, n_bars=n_bars, pad=pad))

    start_t = last_print_t = time.time()
    last_print_n = 0
    n = 0
    for obj in iterable:
        yield obj
        # Now the object was created and processed, so we can print the meter.
        n += 1
        if n - last_print_n >= miniters:
            # We check the counter first, to reduce the overhead of time.time()
            cur_t = time.time()
            if cur_t - last_print_t >= mininterval:
                pstat = format_meter(
                    n,
                    total,
                    cur_t-start_t,
                    n_bars=n_bars,
                    pad=pad,
                )
                sp.print_status(prefix + pstat)

                last_print_n = n
                last_print_t = cur_t

    if not leave and not flush:
        sp.print_status('')
        sys.stdout.write('\r')
    else:
        if last_print_n < n:
            cur_t = time.time()

            pstat = format_meter(
                n,
                total,
                cur_t-start_t,
                n_bars=n_bars,
                pad=pad,
            )
            sp.print_status(prefix + pstat)
        if not flush:
            file.write('\n')


def prange(*args, **kwargs):
    """
    A shortcut for writing PBar(range(...))

    e.g.

    import time
    from pbar import prange
    for i in prange(20):
        print(i)
        time.sleep(0.1)
    """
    return PBar(range(*args), **kwargs)


def format_interval(t):
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return '%d:%02d:%02d' % (h, m, s)
    else:
        return '%02d:%02d' % (m, s)


def format_meter(n, total, elapsed, n_bars=20, pad=None):
    # n - number of finished iterations
    # total - total number of iterations, or None
    # elapsed - number of seconds passed since start
    if n > total:
        total = None

    elapsed_str = format_interval(elapsed)

    if total:
        frac = float(n) / total

        bar_length = int(frac*n_bars)
        bar = '#'*bar_length + '-'*(n_bars-bar_length)

        percentage = '%3d%%' % (frac * 100)

        left_str = format_interval(elapsed / n * (total-n)) if n else '?'

        if pad is None:
            return '|%s| %d/%d %s [elapsed: %s left: %s]' % (
                bar, n, total, percentage, elapsed_str, left_str)
        else:
            pstr = '%d' % pad
            barstr = '|%s| %' + pstr + 'd/%' + pstr + 'd %s [elapsed: %s left: %s]'
            barstr = barstr % (
                bar, n, total, percentage, elapsed_str, left_str)
            return barstr
    else:
        if pad is None:
            return '%d [elapsed: %s]' % (n, elapsed_str)
        else:
            pstr = '%d' % pad
            barstr = '%' + pstr + 'd [elapsed: %s]'
            barstr = barstr % (n, elapsed_str)
            return barstr


class StatusPrinter(object):
    def __init__(self, file, flush=True):
        self.file = file
        self.last_printed_len = 0
        self.flush = flush

    def print_status(self, s):
        if self.flush:
            self.file.write(s + "\n")
        else:
            self.file.write('\r'+s+' '*max(self.last_printed_len-len(s), 0))
        self.file.flush()
        self.last_printed_len = len(s)
