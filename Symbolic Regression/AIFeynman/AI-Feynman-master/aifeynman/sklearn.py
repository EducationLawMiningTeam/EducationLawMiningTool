# written by William La Cava
import sys
import shutil
import numpy as np
import pandas as pd
import hashlib
import os
import time
import re
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sympy.parsing.sympy_parser import parse_expr
from sympy import Symbol, lambdify, N, preorder_traversal


from .dimensionalAnalysis import dimensionalAnalysis
from .get_pareto import Point, ParetoSet
# from .S_add_bf_on_numbers_on_pareto import add_bf_on_numbers_on_pareto
from .S_add_snap_expr_on_pareto import add_snap_expr_on_pareto
from .S_add_sym_on_pareto import add_sym_on_pareto
from .S_change_output import *
from .S_combine_pareto import combine_pareto
from .S_final_gd import final_gd
from .S_get_symbolic_expr_error import get_symbolic_expr_error
from .S_NN_train import NN_train
from .S_NN_eval import NN_eval
from .S_run_bf_polyfit import run_bf_polyfit
from .S_separability import *
from .S_symmetry import *
from .S_NN_get_gradients import evaluate_derivatives
from .S_brute_force_comp import brute_force_comp
from .S_brute_force_gen_sym import brute_force_gen_sym
from .S_compositionality import *
from .S_gen_sym import *
from .S_gradient_decomposition import identify_decompositions

class __LINE__(object):
    import sys

    def __repr__(self):
        try:
            raise Exception
        except:
            return str(sys.exc_info()[2].tb_frame.f_back.f_lineno)

__LINE__ = __LINE__()


PA = ParetoSet()

class AIFeynmanRegressor(RegressorMixin, BaseEstimator):
    """A sklearn API for AIFeynman
    Parameters
    ----------

    BF_try_time - time limit for each brute force call (set by default to 60 
                    seconds)
    BF_ops_file_type - file containing the symbols to be used in the brute 
        force code (set by default to "14ops.txt")
    polyfit_deg - maximum degree of the polynomial tried by the polynomial fit 
        routine (set be default to 4)
    NN_epochs - number of epochs for the training (set by default to 4000)
    vars_name - name of the variables appearing in the equation (including the 
        name ofthe output variable). This should be passed as a list of 
        strings, with the name of the variables appearing in the same 
        order as they are in the file containing the data
    test_percentage - percentage of the input data to be kept aside and 
        as the test
    """

    def __init__(self, 
                 BF_try_time=60, 
                 BF_ops_file_type="14ops.txt",
                 polyfit_deg=4,
                 NN_epochs=4000,
                 test_percentage=20,
                 random_state=None,
                 max_time=3600,
                 tmp_dir=None
                 ):

        self.BF_try_time=BF_try_time 
        self.BF_ops_file_type = BF_ops_file_type
        self.polyfit_deg=polyfit_deg
        self.NN_epochs=NN_epochs
        self.test_percentage=test_percentage
        self.random_state=random_state
        self.max_time=max_time
        self.tmp_dir=tmp_dir

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        self.start_time_ = time.time()
        X, y = check_X_y(X, y, accept_sparse=True)
        np.random.seed(self.random_state)
        data=np.array(X)
        data=np.hstack((data,y.reshape(-1,1)))
        print('data shape:',data.shape)
        if X.flags['C_CONTIGUOUS']:
            filehash = hashlib.md5(X).hexdigest()
        else:
            filehash = hashlib.md5(X.copy(order='C')).hexdigest()
        self.filename = 'tmp_data_' + filehash

        self.PA_ = ParetoSet()
        if self.tmp_dir != None:
            PATHDIR = self.tmp_dir
        else:
            PATHDIR = os.path.dirname(os.path.realpath(__file__))+'/'
        self.pathdir = (PATHDIR + filehash + '_'
                        + str(np.random.randint(2**15-1)) + '/')
        # update global self.pathdir
        if os.path.exists(self.pathdir):
            print('WARNING! {} already exists. Training may overwrite '
                  'files.'.format(self.pathdir))
        else:
            os.makedirs(self.pathdir)
        print('self.pathdir:', type(self.pathdir), self.pathdir)
        print('self.filename:', type(self.filename), self.filename)
        
        # if isinstance(X, pd.DataFrame):
        #     self.vars_name_ = list(X.columns) + ['target']
        # else:
        #     self.vars_name_ = []
        try:
            self._train(data, self.pathdir)
        except Exception as e:
            # make sure files get deleted
            if os.path.exists(self.pathdir):
                print('Except raised during training. Removing',self.pathdir)
                shutil.rmtree(self.pathdir)
            raise e

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        output = self._get_symbolic_expr_output(X, self.best_model_)
        if isinstance(output, float):
            output = np.ones(len(X))*output
        output = np.nan_to_num(output)
        return output

    def complexity(self):
        model_sym = parse_expr(self.best_model_)
        c=0
        for arg in preorder_traversal(model_sym):
            c += 1
        return c

    def _train(self, input_data, pathdir):
        """adapted from run_aifeynman"""
        # If the variable names are passed, do the dimensional analysis first
        filename_orig = self.filename
        try:
            if vars_name!=[]:
                dimensionalAnalysis(pathdir,self.filename,vars_name)
                DR_file = self.filename + "_dim_red_variables.txt"
                self.filename = self.filename + "_dim_red"
            else:
                DR_file = ""
        except Exception as e:
            print(__file__,__LINE__,e)
            DR_file = ""

        # save data to txt. full form is used in add__snap-expr_on_pareto
        # and final_gd
        np.savetxt(pathdir+self.filename, input_data)
        # Split the data into train and test set
        sep_idx = np.random.permutation(len(input_data))

        train_data = input_data[
                sep_idx[0:(100-self.test_percentage)*len(input_data)//100]]
        test_data = input_data[
                sep_idx[self.test_percentage*len(input_data)//100:len(input_data)]]

        np.savetxt(pathdir+self.filename+"_train",train_data)
        if test_data.size != 0:
            np.savetxt(pathdir+self.filename+"_test",test_data)

        PA = ParetoSet()
        ######################################################################
        # Run the code on the train data
        PA = self._run_AI_all(pathdir,
                        self.filename+'_train',
                        BF_try_time=self.BF_try_time,
                        BF_ops_file_type=self.BF_ops_file_type,
                        polyfit_deg=self.polyfit_deg,
                        NN_epochs=self.NN_epochs,
                        PA=PA)
        ######################################################################
        PA_list = PA.get_pareto_points()
        self.PA_list_before_snap_ = PA_list
        np.savetxt(self.pathdir+"results/solution_before_snap_%s.txt" %self.filename,
                   PA_list,
                   fmt="%s"
                  )

        # Run zero, integer and rational snap on the resulted equations
        for j in range(len(PA_list)):
            PA = add_snap_expr_on_pareto(pathdir,
                                         self.filename,
                                         PA_list[j][-1],
                                         PA, "")

        PA_list = PA.get_pareto_points()
        self.PA_list_first_snap_ = PA_list
        np.savetxt(self.pathdir+"results/solution_first_snap_%s.txt" %self.filename,
                    PA_list,fmt="%s")

        # Run gradient descent on the data one more time
        for i in range(len(PA_list)):
            try:
                #WGL: this should be called with data as follows:
                # gd_update = final_gd(data,PA_list[i][-1])
                gd_update = final_gd(pathdir,self.filename,PA_list[i][-1])
                PA.add(Point(x=gd_update[1],y=gd_update[0],data=gd_update[2]))
            except Exception as e:
                print(__file__,__LINE__,e)
                continue

        PA_list = PA.get_pareto_points()
        for j in range(len(PA_list)):
            PA = add_snap_expr_on_pareto(pathdir,self.filename,PA_list[j][-1],
                    PA, DR_file)

        list_dt = np.array(PA.get_pareto_points())
        # WGL
        # data_file_len = len(np.loadtxt(pathdir+self.filename))
        data_file_len = len(input_data)
        log_err = []
        log_err_all = []
        for i in range(len(list_dt)):
            log_err = log_err + [np.log2(float(list_dt[i][1]))]
            log_err_all = log_err_all + [data_file_len*np.log2(float(list_dt[i][1]))]
        log_err = np.array(log_err)
        log_err_all = np.array(log_err_all)

        # Try the found expressions on the test data
        if DR_file=="" and test_data.size != 0:
            test_errors = []
            # input_test_data = np.loadtxt(pathdir+self.filename+"_test")


            for i in range(len(list_dt)):
                # test_errors = test_errors + [get_symbolic_expr_error(input_test_data,str(list_dt[i][-1]))]
                test_errors = test_errors + [get_symbolic_expr_error(test_data,str(list_dt[i][-1]))]
            test_errors = np.array(test_errors)
            # Save all the data to file
            save_data = np.column_stack((test_errors,log_err,log_err_all,list_dt))
        else:
            save_data = np.column_stack((log_err,log_err_all,list_dt))
        # WGL: store model with lowest test_errors
        best_idx = np.argmin(log_err)
        self.best_model_ = list_dt[best_idx][-1]
        self.save_data_ = save_data
        self.pareto_points_ = list_dt
        np.savetxt(self.pathdir+"results/solution_%s" %filename_orig,save_data,fmt="%s")
        print('done training.')

        try:
            shutil.rmtree(self.pathdir)
        except Exception as e:
            print(__file__,__LINE__,e)
            pass

    def _get_symbolic_expr_output(self, data, expr):
        """Adapted from get_symbolic_expr_error"""
        try:
            N_vars = data.shape[1] #WGL: no label passed
            variables = ["x%s" %i for i in np.arange(N_vars)]
            eq = parse_expr(expr)
            f = lambdify(variables, N(eq))
            output = f(*[x for x in data.T])
 
            return output
        except Exception as e: 
            print(__file__,__LINE__,e)
            raise e
        return []
    
    def _time_limit(self):
        limit_reached = (time.time() - self.start_time_) > self.max_time
        if limit_reached:
            print('times up, returning')
        return limit_reached 

    def _run_AI_all(self, pathdir, filename, BF_try_time=60,
                   BF_ops_file_type="14ops", 
                   polyfit_deg=4, NN_epochs=4000, PA=PA):
        print('self._run_AI_all')
        try:
            os.mkdir(self.pathdir+"results/")
        except Exception as e:
            print(__file__,__LINE__,e)
            pass

        # time limit
        if self._time_limit(): return PA
        # load the data for different checks
        data = np.loadtxt(pathdir+filename)

        # Run bf and polyfit
        PA = run_bf_polyfit(pathdir,
                            pathdir,
                            filename,
                            BF_try_time,
                            BF_ops_file_type, 
                            PA, polyfit_deg,
                            og_pathdir=self.pathdir
                            )
        PA = get_squared(pathdir, self.pathdir+"results/mystery_world_squared/",
                        filename, BF_try_time,BF_ops_file_type, PA, polyfit_deg,
                        og_pathdir=self.pathdir
                        )   
        if self._time_limit(): return PA

        # Run bf and polyfit on modified output

        PA = get_acos(pathdir,self.pathdir+"results/mystery_world_acos/",filename,
                BF_try_time,BF_ops_file_type, PA, polyfit_deg,
                og_pathdir=self.pathdir)
        if self._time_limit(): return PA
        print(self.pathdir)
        PA = get_asin(pathdir,self.pathdir+"results/mystery_world_asin/",filename,
                BF_try_time,BF_ops_file_type, PA, polyfit_deg,
                og_pathdir=self.pathdir)
        if self._time_limit(): return PA
        print(self.pathdir)
        PA = get_atan(pathdir,self.pathdir+"results/mystery_world_atan/",filename,
                BF_try_time,BF_ops_file_type, PA, polyfit_deg,
                og_pathdir=self.pathdir)
        if self._time_limit(): return PA
        print(self.pathdir)
        PA = get_cos(pathdir,self.pathdir+"results/mystery_world_cos/",filename,
                BF_try_time,BF_ops_file_type, PA, polyfit_deg,
                og_pathdir=self.pathdir)
        if self._time_limit(): return PA
        PA = get_exp(pathdir,self.pathdir+"results/mystery_world_exp/",filename,
                BF_try_time,BF_ops_file_type, PA, polyfit_deg,
                og_pathdir=self.pathdir)
        if self._time_limit(): return PA
        PA = get_inverse(pathdir,self.pathdir+"results/mystery_world_inverse/",filename,
                BF_try_time,BF_ops_file_type, PA, polyfit_deg,
                og_pathdir=self.pathdir)
        if self._time_limit(): return PA
        PA = get_log(pathdir,self.pathdir+"results/mystery_world_log/",filename,
                BF_try_time,BF_ops_file_type, PA, polyfit_deg,
                og_pathdir=self.pathdir)
        if self._time_limit(): return PA
        PA = get_sin(pathdir,self.pathdir+"results/mystery_world_sin/",filename,
                BF_try_time,BF_ops_file_type, PA, polyfit_deg,
                og_pathdir=self.pathdir)
        if self._time_limit(): return PA
        PA = get_sqrt(pathdir,self.pathdir+"results/mystery_world_sqrt/",filename,
                BF_try_time,BF_ops_file_type, PA, polyfit_deg,
                og_pathdir=self.pathdir)
        if self._time_limit(): return PA
        PA = get_squared(pathdir,self.pathdir+"results/mystery_world_squared/",
                filename,BF_try_time,BF_ops_file_type, PA, polyfit_deg,
                og_pathdir=self.pathdir)
        if self._time_limit(): return PA
        PA = get_tan(pathdir,self.pathdir+"results/mystery_world_tan/",filename,
                BF_try_time,BF_ops_file_type, PA, polyfit_deg,
                og_pathdir=self.pathdir)
        if self._time_limit(): return PA

    ###########################################################################
        # check if the NN is trained. If it is not, train it on the data.
        if len(data[0])<3:
            print("Just one variable!")
            pass
        elif os.path.exists(self.pathdir+"results/NN_trained_models/models/" 
                            + filename + ".h5"):# or len(data[0])<3:
            print("NN already trained \n")
            print("NN loss: ", NN_eval(pathdir,filename,
                                       og_pathdir=self.pathdir)[0], "\n")
            model_feynman = NN_eval(pathdir,filename,
                                    og_pathdir=self.pathdir)[1]
        elif os.path.exists(self.pathdir+"results/NN_trained_models/models/" 
                            + filename + "_pretrained.h5"):
            print("Found pretrained NN \n")
            model_feynman = NN_train(pathdir,
                                     filename,
                                     NN_epochs/2,
                                     lrs=1e-3,
                                     N_red_lr=3,
                     pretrained_path=(self.pathdir
                                      +"results/NN_trained_models/models/" 
                                      + filename 
                                      + "_pretrained.h5"),
                                     og_pathdir=self.pathdir
                                    )
            print("NN loss after training: ", 
                  NN_eval(pathdir,filename, og_pathdir=self.pathdir), "\n")
        else:
            print("Training a NN on the data... \n")
            model_feynman = NN_train(pathdir,filename,NN_epochs,
                                     og_pathdir=self.pathdir)
            print("NN loss: ", 
                  NN_eval(pathdir,filename, og_pathdir=self.pathdir), "\n")

        
        # Check which symmetry/separability is the best
        # Symmetries
        print("Checking for symmetries...")
        symmetry_minus_result = check_translational_symmetry_minus(
                                    pathdir,filename,og_pathdir=self.pathdir)
        symmetry_divide_result = check_translational_symmetry_divide(
                                    pathdir,filename,og_pathdir=self.pathdir)
        symmetry_multiply_result = check_translational_symmetry_multiply(
                                    pathdir,filename,og_pathdir=self.pathdir)
        symmetry_plus_result = check_translational_symmetry_plus(
                                    pathdir,filename,og_pathdir=self.pathdir)
        print("")

        print("Checking for separabilities...")
        # Separabilities
        separability_plus_result = check_separability_plus(
                                    pathdir,filename,og_pathdir=self.pathdir)
        separability_multiply_result = check_separability_multiply(
                                    pathdir,filename,og_pathdir=self.pathdir)

        if symmetry_plus_result[0]==-1:
            idx_min = -1
        else:
            idx_min = np.argmin(np.array([symmetry_plus_result[0],
                                          symmetry_minus_result[0],
                                          symmetry_multiply_result[0],
                                          symmetry_divide_result[0],
                                          separability_plus_result[0],
                                          separability_multiply_result[0]]))

        print("")
        # Check if compositionality is better than the best so far
        if idx_min==0:
            mu, sigma = symmetry_plus_result[3:]
        elif idx_min==1:
            mu, sigma = symmetry_minus_result[3:]
        elif idx_min==2:
            mu, sigma = symmetry_multiply_result[3:]
        elif idx_min==3:
            mu, sigma = symmetry_divide_result[3:]
        elif idx_min==4:
            mu, sigma = separability_plus_result[3:]
        elif idx_min==5:
            mu, sigma = separability_multiply_result[3:]

        print("Checking for compositionality...")
        # Save the gradients for compositionality
        try:
            succ_grad = evaluate_derivatives(pathdir,filename,model_feynman, 
                                             self.pathdir)
        except Exception as e:
            print(__file__,__LINE__,e)
            succ_grad = 0

        idx_comp = 0
        if succ_grad == 1:
            #try:
            for qqqq in range(1):
                brute_force_comp(self.pathdir+"results/",
                                 "gradients_comp_%s.txt" %filename,
                                 600,
                                 "14ops.txt",
                                 og_pathdir=self.pathdir
                                 )
                bf_all_output = np.loadtxt(self.pathdir+"results_comp.dat", dtype="str")
                for bf_i in range(len(bf_all_output)):
                    idx_comp_temp = 0
                    try:
                        express = bf_all_output[:,1][bf_i]
                        idx_comp_temp, eqq, new_mu, new_sigma = \
                                check_compositionality(pathdir,
                                                       filename,
                                                       model_feynman,
                                                       express,
                                                       mu,
                                                       sigma,
                                                       nu=10)
                        if idx_comp_temp==1:
                            idx_comp = 1
                            math_eq_comp = eqq
                            mu = new_mu
                            sigma = new_sigma
                    except Exception as e:
                        print(__file__,__LINE__,e)
                        continue
            #except:
            #    idx_comp = 0
        else:
            idx_comp = 0
        print("")
        
        if idx_comp==1:
            idx_min = 6


        print("Checking for generalized symmetry...")
        # Check if generalized separabilty is better than the best so far
        idx_gen_sym = 0
        for kiiii in range(1):
            if len(data[0])>3:
                # find the best separability indices
                decomp_idx = identify_decompositions(pathdir,
                                                     filename,
                                                     model_feynman,
                                                     og_pathdir=self.pathdir
                                                     )
                brute_force_gen_sym(self.pathdir+"results/",
                                    "gradients_gen_sym_%s" %filename,
                                    600,"14ops.txt", og_pathdir=self.pathdir)
                bf_all_output = np.loadtxt(self.pathdir+"results_gen_sym.dat",
                                           dtype="str")
                
                for bf_i in range(len(bf_all_output)):
                    idx_gen_sym_temp = 0
                    try:
                        express = bf_all_output[:,1][bf_i]
                        idx_gen_sym_temp, eqq, new_mu, new_sigma = \
                                check_gen_sym(pathdir,
                                              filename,
                                              model_feynman,
                                              decomp_idx,
                                              express,
                                              mu,
                                              sigma,
                                              nu=10)
                        if idx_gen_sym_temp==1:
                            idx_gen_sym = 1
                            math_eq_gen_sym = eqq
                            mu = new_mu
                            sigma = new_sigma
                    except Exception as e:
                        print(__file__,__LINE__,e)
                        continue

        if idx_gen_sym==1:
            idx_min = 7
        print("")

        # Apply the best symmetry/separability and rerun the main function on 
        # this new file
        if idx_min == 0:
            print("Translational symmetry found for variables:", 
                    symmetry_plus_result[1],symmetry_plus_result[2])
            new_pathdir, new_filename = do_translational_symmetry_plus(
                                            pathdir,
                                            filename,
                                            symmetry_plus_result[1],
                                            symmetry_plus_result[2],
                                            og_pathdir=self.pathdir
                                            )
            PA1_ = ParetoSet()
            PA1 = self._run_AI_all(new_pathdir, new_filename, BF_try_time,
                             BF_ops_file_type,
                             polyfit_deg,
                             NN_epochs, 
                             PA1_)
            PA = add_sym_on_pareto(pathdir,filename,PA1,
                    symmetry_plus_result[1],symmetry_plus_result[2],PA,"+")
            return PA

        elif idx_min == 1:
            print("Translational symmetry found for variables:", 
                    symmetry_minus_result[1],symmetry_minus_result[2])
            new_pathdir, new_filename = do_translational_symmetry_minus(
                                            pathdir,
                                            filename,
                                            symmetry_minus_result[1],
                                            symmetry_minus_result[2],
                                            og_pathdir=self.pathdir
                                            )
            PA1_ = ParetoSet()
            PA1 = self._run_AI_all(new_pathdir,
                             new_filename,
                             BF_try_time,
                             BF_ops_file_type,
                             polyfit_deg,
                             NN_epochs,
                             PA1_)
            PA = add_sym_on_pareto(pathdir,
                                   filename,
                                   PA1,
                                   symmetry_minus_result[1],
                                   symmetry_minus_result[2],PA,"-")
            return PA

        elif idx_min == 2:
            print("Translational symmetry found for variables:",
                    symmetry_multiply_result[1],
                    symmetry_multiply_result[2])
            new_pathdir, new_filename = do_translational_symmetry_multiply(
                                            pathdir,
                                            filename,
                                            symmetry_multiply_result[1],
                                            symmetry_multiply_result[2],
                                            og_pathdir=self.pathdir
                                            )
            PA1_ = ParetoSet()
            PA1 = self._run_AI_all(new_pathdir,
                             new_filename,
                             BF_try_time,
                             BF_ops_file_type,
                             polyfit_deg,
                             NN_epochs,
                             PA1_)
            PA = add_sym_on_pareto(pathdir,
                                   filename,
                                   PA1,
                                   symmetry_multiply_result[1],
                                   symmetry_multiply_result[2],
                                   PA,"*")
            return PA

        elif idx_min == 3:
            print("Translational symmetry found for variables:",
                    symmetry_divide_result[1],
                    symmetry_divide_result[2])
            new_pathdir, new_filename = do_translational_symmetry_divide(
                                            pathdir,
                                            filename,
                                            symmetry_divide_result[1],
                                            symmetry_divide_result[2],
                                            og_pathdir=self.pathdir
                                            )
            PA1_ = ParetoSet()
            PA1 = self._run_AI_all(new_pathdir,
                             new_filename,
                             BF_try_time,
                             BF_ops_file_type,
                             polyfit_deg,
                             NN_epochs,
                             PA1_)
            PA = add_sym_on_pareto(pathdir,
                                   filename,
                                   PA1,
                                   symmetry_divide_result[1],
                                   symmetry_divide_result[2],
                                   PA,
                                   "/")
            return PA

        elif idx_min == 4:
            print("Additive separability found for variables:",
                    separability_plus_result[1],separability_plus_result[2])
            new_pathdir1, new_filename1, new_pathdir2, new_filename2,  = \
                    do_separability_plus(pathdir,
                                         filename,
                                         separability_plus_result[1],
                                         separability_plus_result[2],
                                         og_pathdir=self.pathdir
                                         )
            PA1_ = ParetoSet()
            PA1 = self._run_AI_all(new_pathdir1,
                             new_filename1,
                             BF_try_time,
                             BF_ops_file_type,
                             polyfit_deg,
                             NN_epochs,
                             PA1_)
            PA2_ = ParetoSet()
            PA2 = self._run_AI_all(new_pathdir2,
                             new_filename2,
                             BF_try_time,
                             BF_ops_file_type,
                             polyfit_deg,
                             NN_epochs,
                             PA2_)
            combine_pareto_data = np.loadtxt(pathdir+filename)
            PA = combine_pareto(combine_pareto_data,
                                PA1,
                                PA2,
                                separability_plus_result[1],
                                separability_plus_result[2],
                                PA,
                                "+")
            return PA

        elif idx_min == 5:
            print("Multiplicative separability found for variables:",
                    separability_multiply_result[1],
                    separability_multiply_result[2])
            new_pathdir1, new_filename1, new_pathdir2, new_filename2,  = \
                    do_separability_multiply(pathdir,
                                             filename,
                                             separability_multiply_result[1],
                                             separability_multiply_result[2],
                                             og_pathdir=self.pathdir
                                             )
            PA1_ = ParetoSet()
            PA1 = self._run_AI_all(new_pathdir1,
                             new_filename1,
                             BF_try_time,
                             BF_ops_file_type,
                             polyfit_deg,
                             NN_epochs,
                             PA1_)
            PA2_ = ParetoSet()
            PA2 = self._run_AI_all(new_pathdir2,
                             new_filename2,
                             BF_try_time,
                             BF_ops_file_type,
                             polyfit_deg,
                             NN_epochs,
                             PA2_)
            combine_pareto_data = np.loadtxt(pathdir+filename)
            PA = combine_pareto(combine_pareto_data,
                                PA1,
                                PA2,
                                separability_multiply_result[1],
                                separability_multiply_result[2],
                                PA,
                                "*")
            return PA

        elif idx_min == 6:
            print("Compositionality found")
            new_pathdir, new_filename = do_compositionality(pathdir,
                    filename,
                    math_eq_comp,og_pathdir=self.pathdir)
            PA1_ = ParetoSet()
            PA1 = self._run_AI_all(new_pathdir,
                             new_filename,
                             BF_try_time,
                             BF_ops_file_type,
                             polyfit_deg,
                             NN_epochs,
                             PA1_)
            PA = add_comp_on_pareto(PA1,PA,math_eq_comp)
            return PA

        elif idx_min == 7:
            print("Generalized symmetry found")
            new_pathdir, new_filename = do_gen_sym(pathdir,
                                                   filename,
                                                   decomp_idx,
                                                   math_eq_gen_sym,
                                                   og_pathdir=self.pathdir
                                                   )
            PA1_ = ParetoSet()
            PA1 = self._run_AI_all(new_pathdir,
                             new_filename,
                             BF_try_time,
                             BF_ops_file_type,
                             polyfit_deg,
                             NN_epochs,
                             PA1_)
            PA = add_gen_sym_on_pareto(PA1,PA, decomp_idx, math_eq_gen_sym)
            return PA
        else:
            return PA
