import numpy as np
from functools import lru_cache
import piff


@lru_cache(maxsize=128)
def get_piff_psf(psf_path):
    """load a piff.PSF object from the specified file"""
    return piff.read(psf_path)


@lru_cache(maxsize=1024)
def get_piff_psf_info(*, expnum, piff_campaign):
    """Query the piff PSF QA flags from the database

    Parameters
    ----------
    expnum : int
        The exposure number to query.
    piff_campaign : str
        The Piff campaign in DESDM.

    Returns
    -------
    piff_info : dict of dicts
        A dictionary keyed on filename with the table values for cutting and
        other stats.
    """

    query = """\
select
  distinct
  qa.expnum,
  qa.ccdnum,
  t.tag,
  qa.filename,
  qa.flag,
  qa.fwhm_cen,
  qa.star_t_std,
  qa.star_t_mean,
  qa.nstar,
  qa.exp_star_t_mean,
  qa.exp_star_t_std
from
  PIFF_HSM_MODEL_QA qa,
  proctag t,
  miscfile m
where
  qa.expnum = {expnum}
  and t.tag = '{piff_campaign}'
  and t.pfw_attempt_id = m.pfw_attempt_id
  and m.filetype = 'piff_model'
  and m.filename = qa.filename
""".format(expnum=expnum, piff_campaign=piff_campaign)

    import easyaccess as ea
    conn = ea.connect(section='desoper')
    curs = conn.cursor()
    curs.execute(query)

    c = curs.fetchall()
    if len(c) == 0:
        raise RuntimeError("Could not fetch Piff PSF QA info!")

    piff_info = {}
    for (
        expnum, ccdnum, tag, filename, desdm_flags,
        fwhm_cen, star_t_std, star_t_mean,
        nstar, exp_star_t_mean, exp_star_t_std,
    ) in c:
        if filename not in piff_info:
            piff_info[filename] = {}
        piff_info[filename] = dict(
            desdm_flags=desdm_flags,
            expnum=expnum,
            ccdnum=ccdnum,
            fwhm_cen=fwhm_cen,
            star_t_std=star_t_std,
            star_t_mean=star_t_mean,
            nstar=nstar,
            exp_star_t_mean=exp_star_t_mean,
            exp_star_t_std=exp_star_t_std,
        )

    return piff_info


def compute_piff_flags(
    *, piff_info, max_fwhm_cen, min_nstar, max_exp_T_mean_fac, max_ccd_T_std_fac,
):
    """Compute flags from Piff stats.

    Parameters
    ----------
    piff_info : dict
        A dictionary with the Piff info used to compute the flags.
    max_fwhm_cen : float
        The maximum FWHM cen for a given solution. A good default is 3.6 arcseconds.
    min_nstar : int
        Minimum number of stars that a PSF model must use. A good default is 25.
    max_exp_T_mean_fac : float
        The maximum allowed difference between a CCDs mean T and the mean over the
        focal plane in terms of the variation over the focal plane

            (star_t_mean - exp_star_t_mean) < max_exp_T_mean_fac * exp_star_t_std

        A good default is 4.
    max_ccd_T_std_fac : float
        The maximum allowed value of the star T std. across a CCD

            star_t_std < max_ccd_T_std_fac * star_t_mean

        A good default is 0.3.

    Returns
    -------
    flags : int
        The flags integer. This is a bit flag field with the following defs

            2**0 - DESDM flags != 0
            2**1 - fwhm_cen >= max_fwhm_cen
            2**2 - star_t_std >= max_ccd_T_std_fac * star_t_mean
            2**3 - nstar < min_nstar
            2**4 - |star_t_mean - exp_star_t_mean| >= max_exp_T_mean_fac * exp_star_t_std
    """  # noqa
    flags = 0

    if piff_info["desdm_flags"] != 0:
        flags |= 2**0

    if piff_info["fwhm_cen"] >= max_fwhm_cen:
        flags |= 2**1

    if piff_info["star_t_std"] >= max_ccd_T_std_fac * piff_info["star_t_mean"]:
        flags |= 2**2

    if piff_info["nstar"] < min_nstar:
        flags |= 2**3

    if (
        np.abs(piff_info["star_t_mean"] - piff_info["exp_star_t_mean"])
        >= max_exp_T_mean_fac * piff_info["exp_star_t_std"]
    ):
        flags |= 2**4

    return flags
