#! /usr/bin/env python3
"""
Query GAIA_DR2 to obtain stars on (or near) a coadd tile
"""


######################################################################################
def get_ALL_SATSTARS_data(dbh,dbSchema,Timing=False,verbose=0):
    """ Pull GAIA data (ra,dec,mag) for stars in a region
    """

    t0=time.time()
    query="select s.expnum,s.ccdnum,s.band,s.ra,s.dec,s.radius from gruendl.some_satstars s"
#
    if (verbose > 0):
        if (verbose == 1):
            QueryLines=query.split('\n')
            QueryOneLine='sql = '
            for line in QueryLines:
                QueryOneLine=QueryOneLine+" "+line.strip()
            print("{:s}".format(QueryOneLine))
        if (verbose > 1):
            print("{:s}".format(query))
#
#   Establish a DB cursor
#
    curDB = dbh.cursor()
#    curDB.execute(query)
#    desc = [d[0].lower() for d in curDB.description]

    prefetch=100000
    curDB.arraysize=int(prefetch)
    curDB.execute(query)
#    header=[d[0].lower() for d in curDB.description]
    header=[d[0].upper() for d in curDB.description]
    cat_data=pd.DataFrame(curDB.fetchall())

    CatDict={}
    if (cat_data.empty):
        print("# No values returned from query of {tval:s} ".format(tval="GAIA_DR2"))
        for val in header:
            CatDict[val]=np.array([])
    else:
        cat_data.columns=header
        for val in header:
            CatDict[val]=np.array(cat_data[val])
    curDB.close()

    if (verbose>0):
        print("# Number of Some_SATSTARS objects found is {nval:d} ".format(nval=CatDict[header[0]].size))
    if (Timing):
        t1=time.time()
        print(" Query execution time: {:.2f}".format(t1-t0))

    return CatDict,header




######################################################################################

if __name__ == "__main__":

    import argparse
    import os
    import despydb.desdbi
    import time
#    import yaml
    import mepipelineappintg.cat_query as cq
    import pandas as pd
    import fitsio
    import numpy as np

    from astropy.coordinates import SkyCoord
    from astropy import units as u



    parser = argparse.ArgumentParser(description='Query code to obtain image inputs for COADD/multiepoch pipelines.')
    parser.add_argument('--tilename', action='store', type=str, default=None,
                        help='Tilename of interest')
    parser.add_argument('--extend', action='store', type=float, default=0.0,
                        help='Amount to extend tile boundary (default=0.0).  Units depend on --method. Negative values will shrink but are not strictly controlled.')
    parser.add_argument('--method', action='store', type=str, default='fixed',
                        help='Method to used with --extend. Either "fractional" (expand by a factor) or "fixed" (default) number of arcminutes')
    parser.add_argument('-o', '--output', action='store', type=str, required=True,
                        help='Output FITS table to be written')
    parser.add_argument('-s', '--section', action='store', type=str, default='db-dessci',
                        help='section of .desservices file with connection info')
    parser.add_argument('-S', '--Schema', action='store', type=str, default='des_admin',
                        help='DB schema (do not include \'.\').')
    parser.add_argument('-v', '--verbose', action='store', type=int, default=0,
                        help='Verbosity (defualt:0; currently values up to 2)')
    args = parser.parse_args()
    if args.verbose:
        print("Args: ", args)

    #
    #   Handle simple args (verbose, Schema, PFW_ATTEMPT_ID)
    #
    verbose = args.verbose

    if args.Schema is None:
        dbSchema = ""
    else:
        dbSchema = f"{args.Schema}."

    Timing=True

    ########################################################
    #
    #   Setup a DB connection
    #
    try:
        desdmfile = os.environ["des_services"]
    except KeyError:
        desdmfile = None
    dbh = despydb.desdbi.DesDbi(desdmfile, args.section, retry=True)
    #    cur = dbh.cursor()


    if (args.tilename == "ALL"):
#
#       Note this method currently sucks down too much memory to execute on a machine w 256 GB or memory so not recommended
#
        print("Warning: the current implementation using all sucks down too much memory (you have been forwarned!)")
        GCat,GHead=cq.get_ALL_GAIA_data(dbh,dbSchema,Timing=True,verbose=args.verbose)
        SCat,SHead=get_ALL_SATSTARS_data(dbh,dbSchema,Timing=True,verbose=args.verbose)
    else:
        tileDict=cq.query_Tile_edges(args.tilename,dbh,dbSchema,verbose=args.verbose)

        tileDict=cq.expand_range(tileDict,extend=args.extend,method=args.method,verbose=args.verbose)
#
#       Since I apparently am  not consistent crowbar the structure for bounds of a tile to that used in 
#       bounding an RA/DEC search.
#
        radec_box={}
        for tile in tileDict:
            if (tileDict[tile]['crossra0']=="Y"):
                radec_box['crossra0']=True
            else:
                radec_box['crossra0']=False
            radec_box['ra1']=tileDict[tile]['racmin']
            radec_box['ra2']=tileDict[tile]['racmax']
            radec_box['dec1']=tileDict[tile]['deccmin']
            radec_box['dec2']=tileDict[tile]['deccmax']

        GCat,GHead=cq.get_cat_radec_range(radec_box,dbh,dbSchema,Timing=True,verbose=args.verbose)

        CList=['expnum','ccdnum','band','ra','dec','radius']
        SCat,SHead=cq.get_cat_radec_range(radec_box,dbh,dbSchema='gruendl.',table='some_satstars',cols=CList,Timing=True,verbose=args.verbose)

#
#   Sanity checks that I got the formats right.
#
#    for col in SHead:
#        print(col,SCat[col].dtype)
#    for col in GHead:
#        print(col,GCat[col].dtype)


#
#   Match catalogs 
#
    c2=SkyCoord(ra=GCat['RA']*u.degree,dec=GCat['DEC']*u.degree)
    c1=SkyCoord(ra=SCat['RA']*u.degree,dec=SCat['DEC']*u.degree)

    SCat['Mag']=np.full_like(SCat['RA'],-1.0)

#   Sets matching radius criterion to 1 arcsec.
    MatchRad=1.0
    idx2, d2d, d3d = c1.match_to_catalog_sky(c2)
    idx1=np.arange(SCat['RA'].size)
    wsm=np.where(d2d.arcsecond<MatchRad)
#    print(GCat['RA'].size)
#    print(SCat['RA'].size)
#    print(idx1.size)
#    print(idx1[wsm].size)

    ret_cat={}
    for col in SCat:
        ret_cat[col]=SCat[col][idx1[wsm]]
    ret_cat['Mag']=GCat['PHOT_G_MEAN_MAG'][idx2[wsm]]

    print("Checking that matching was sane")
    for i in [10,50,1000,2000,5000,10000,20000]:
        if (i < idx1[wsm].size):
            print("-------------------------------------------")
            print(i,SCat['RA'][idx1[wsm][i]],SCat['DEC'][idx1[wsm][i]])
            print(i,GCat['RA'][idx2[wsm][i]],GCat['DEC'][idx2[wsm][i]],GCat['PHOT_G_MEAN_MAG'][idx2[wsm][i]])
    print("-------------------------------------------")
    print("Faintest magnitude among matches: ",np.amax(ret_cat['Mag']))

    fitsio.write(args.output, ret_cat, extname='GAIA_SATSTAR', clobber=True)

    exit(0)
