import numpy as np
import matplotlib.pyplot as plt


def bland_altman_plot(m1, m2,
                      *,
                      sd_limit=1.96,
                      ax=None,
                      scatter_kws=None,
                      mean_line_kws=None,
                      limit_lines_kws=None,
                      xlabel=None,
                      ylabel=None,
                      ):
    """Bland-Altman Plot.

    A Bland-Altman plot is a graphical method to analyze the differences
    between two methods of measurement. The mean of the measures is plotted
    against their difference.

    Args:
        m1, m2: pandas Series or array-like
        sd_limit : float, default 1.96
            The limit of agreements expressed in terms of the standard deviation of
            the differences. If `md` is the mean of the differences, and `sd` is
            the standard deviation of those differences, then the limits of
            agreement that will be plotted will be
                           md - sd_limit * sd, md + sd_limit * sd
            The default of 1.96 will produce 95% confidence intervals for the means
            of the differences.
            If sd_limit = 0, no limits will be plotted, and the ylimit of the plot
            defaults to 3 standard deviatons on either side of the mean.
        ax: matplotlib.axis, optional
            matplotlib axis object to plot on.
        scatter_kws: dict
            Options to to style the scatter plot. Accepts any keywords for the
            matplotlib Axes.scatter plotting method
        mean_line_kws: dict
            Options to to style the scatter plot. Accepts any keywords for the
            matplotlib Axes.axhline plotting method
        limit_lines_kws: dict
            Options to to style the scatter plot. Accepts any keywords for the
            matplotlib Axes.axhline plotting method

    Returns:
        ax: matplotlib Axis object
    """

    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    if ax is None:
        ax = plt.gca()

    scatter_kws = scatter_kws or {}
    if 's' not in scatter_kws:
        scatter_kws['s'] = 20
    mean_line_kws = mean_line_kws or {}
    limit_lines_kws = limit_lines_kws or {}
    for kws in [mean_line_kws, limit_lines_kws]:
        if 'color' not in kws:
            kws['color'] = 'gray'
        if 'linewidth' not in kws:
            kws['linewidth'] = 1
    if 'linestyle' not in mean_line_kws:
        kws['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kws:
        kws['linestyle'] = ':'

    ax.scatter(means, diffs, **scatter_kws)
    ax.axhline(mean_diff, **mean_line_kws)  # draw mean line

    # Annotate mean line with mean difference
    ax.annotate('mean diff:\n{}'.format(np.round(mean_diff, 2)),
                xy=(0.99, 0.55),
                horizontalalignment='right',
                xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (2 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kws)
        ax.annotate('-{}*SD:\n{}'.format(sd_limit, np.round(lower, 2)),
                    xy=(0.99, 0.30),
                    horizontalalignment='right',
                    xycoords='axes fraction')
        ax.annotate('+{}*SD:\n{}'.format(sd_limit, np.round(upper, 2)),
                    xy=(0.99, 0.80),
                    horizontalalignment='right',
                    xycoords='axes fraction')

    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

    ax.set_ylabel(ylabel or 'Difference')
    ax.set_xlabel(xlabel or 'Means')
    plt.tight_layout()
    return ax
