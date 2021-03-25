#
# Step 1: Subselect of saturation star data needed to feed matching routine (from DESOPER)
#

create table some_satstars as 
select c.pfw_attempt_id,c.expnum,c.ccdnum,c.band,s.ra,s.dec,s.radius 
from proctag t, catalog c, satstar s 
where t.tag='Y6A1_COADD_INPUT' 
    and t.pfw_attempt_id=c.pfw_attempt_id 
    and c.filetype='cat_satstar' 
    and c.band in ('r','i','z') 
    and c.filename=s.filename;

# Step 2: Then move table to DESSCI so that can access GAIA_DR2 at the same time

# Step 3: Perform match against GAIA_DR2 
#   Basically this takes an tile near the center of the footprint, 
#   extends its bounds by a factor of 20, so O(100 sq deg)
#   obtains entries from GAIA and the SOME_SATSTARS table above
#   matches them (with 1.0 matching criteria)
#   writes a FITS tabel (allsat.fits) that can then be used to understand the relation between mask sizes and stellar magnitude

comp_GAIA_to_SATSTAR.py --tile DES0238-3206 -o allsat.fits -v 2 --method fractional --extend 20.

