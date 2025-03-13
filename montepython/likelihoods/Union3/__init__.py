"""
.. module:: Union3
    :synopsis: Union3 likelihood based on arXiv:2311.12098 
               with analytical marginalization over the SN absolute magnitude
    :data: taken from https://github.com/CobayaSampler/sn_data/tree/master/Union3

.. moduleauthor:: Anton Chudaykin  <Anton.Chudaykin@unige.ch> 
    extension of the lkl written by
    Vivian Poulin <vivian.poulin@umontpellier.fr>, with help from Dillon Brout and Dan Scolnic

Based on the previous Pantheon lkl from Rodrigo von Marttens and Antonella Cid, which was based on JLA likelihood writted by Benjamin Audren

.. code::

    C00 = mag_covmat_file

.. note::

    Since there are a lot of file manipulation involved, the "pandas" library
    has to be installed -- it is an 8-fold improvement in speed over numpy, and
    a 2-fold improvement over a fast Python implementation. The "numexpr"
    library is also needed for doing the fast array manipulations, done with
    blas daxpy function in the original c++ code. Both can be installed with
    pip (Python package manager) easily.

"""
import numpy as np
import scipy.linalg as la
import montepython.io_mp as io_mp
try:
    import numexpr as ne
except ImportError:
    raise io_mp.MissingLibraryError(
        "This likelihood has intensive array manipulations. You "
        "have to install the numexpr Python package. Please type:\n"
        "(sudo) pip install numexpr --user")
from montepython.likelihood_class import Likelihood_sn
from time import time

class Union3(Likelihood_sn):

    def __init__(self, path, data, command_line):

	#Read the data and covariance matrix
        try:
            Likelihood_sn.__init__(self, path, data, command_line)
        except IOError:
            raise io_mp.LikelihoodError(
                "The Union3 data files were not found. Please check if "
                "the following files are in the data/Union3 directory: "
                "\n-> full_long.dataset"
                "\n-> lcparam_full.txt"
                "\n-> mag_covmat.txt")

        # are there conflicting experiments?
        conflicting_experiments = [
            'Pantheon', 'Pantheon_Plus_SH0ES',
            'hst', 'sh0es']
        for experiment in conflicting_experiments:
            if experiment in data.experiments:
                raise io_mp.LikelihoodError(
                    'Union3 reports conflicting SN or H0 measurments from: %s' %(experiment))

        # Load matrices from text files, whose names were read in the
        # configuration file
        self.C00 = self.read_matrix(self.mag_covmat_file)
        # Reading light-curve parameters from self.data_file 
        self.light_curve_params = self.read_light_curve_parameters()
        #print("self.light_curve_params=",self.light_curve_params)

        # Reordering by J. Renk. The following steps can be computed in the
        # initialisation step as they do not depend on the point in parameter-space
        #   -> likelihood evaluation is 30% faster

        # Compute the covariance matrix
        # The module numexpr is used for doing quickly the long multiplication
        # of arrays (factor of 3 improvements over numpy). It is used as a
        # replacement of blas routines cblas_dcopy and cblas_daxpy
        # For numexpr to work, we need (seems like a bug, but anyway) to create
        # local variables holding the arrays. This cost no time (it is a simple
        # pointer assignment)
        C00 = self.C00
        covm = ne.evaluate("C00")

	#The data are ordered by increasing z, which simplifies things.
        self.true_size = len(self.light_curve_params.zcmb)

        # Update the diagonal terms of the covariance matrix with the
        # statistical error
	## VP: statistical errors are already included! we ignore this step, leave it for comparison with former lkl.
#        covm += np.diag(sn.m_b_corr_err**2)

        # Whiten the residuals, in two steps.
        # Step 1) Compute the Cholesky decomposition of the covariance matrix, in
        # place. This is a time expensive (0.015 seconds) part, which is why it is
        # now done in init. Note that this is different to JLA, where it needed to
        # be done inside the loglkl function.
        self.cov = la.cholesky(covm, lower=True, overwrite_a=True)
        # Step 2) depends on point in parameter space -> done in loglkl calculation
        #self.cov2 = newcovm

        print("Union3 is initialized successfully")


    def loglkl(self, cosmo, data):
        """
        Compute negative log-likelihood (eq.15 Betoule et al. 2014)

        """
        # Recover the distance moduli from CLASS (a size N vector of double
        # containing the predicted distance modulus for each SN in the JLA
        # sample, given the redshift of the supernova.)


        redshifts = self.light_curve_params.zcmb
        size = redshifts.size

        moduli = np.empty((self.true_size, ))
        Mb_obs = np.empty((self.true_size, ))
        good_z = 0

        for index, row in self.light_curve_params.iterrows():
            z_cmb = row['zcmb']
            z_hel = z_cmb
            Mb_corr = row['mb']
            #print("Mb_corr=",Mb_corr)
            #if z_cmb > self.z_min:
            moduli[good_z] = 5 * np.log10((1+z_cmb)*(1+z_hel)*cosmo.angular_distance(z_cmb)) + 0*25
            Mb_obs[good_z] = Mb_corr
            good_z+=1

        #test
        #np.savetxt("moduli.txt", moduli, fmt="%.10e") 

        if self.use_abs_mag:
            # Sample the SN absolute magnitude
            # Convenience variables: store the nuisance parameters in short named
            # variables
            M = (data.mcmc_parameters['M']['current'] *
             data.mcmc_parameters['M']['scale'])

            # This operation loops over all supernovae!
            # Compute the moduli
            residuals = Mb_obs - moduli - M - 25.

            # Step 2) (Step 1 is done in the init) Solve the triangular system, also time expensive (0.02 seconds)
            residuals = la.solve_triangular(self.cov, residuals, lower=True, check_finite=False)

            # Finally, compute the chi2 as the sum of the squared residuals
            chi2 = (residuals**2).sum()

        else:
            # Analytical marginalizetion over the SN absolute magnitude
            d = Mb_obs - moduli
            d_w = la.solve_triangular(self.cov, d, lower=True, check_finite=False)
            ones = np.ones_like(d)
            ones_w = la.solve_triangular(self.cov, ones, lower=True, check_finite=False)
            chi2 = np.dot(d_w, d_w) - (np.dot(d_w, ones_w)**2) / np.dot(ones_w, ones_w)


        return -0.5 * chi2
