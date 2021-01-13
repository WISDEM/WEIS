__author__ = ["Nikhar Abbas", "Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = ["Nikhar Abbas", "Jake Nunemaker"]
__email__ = ["nikhar.abbas@nrel.gov", "jake.nunemaker@nrel.gov"]


def batch_processing(self):
    """
    Run a full batch processing case!
    """

    raise NotImplementedError()
    # # ------------------ Input consistancy checks ------------------ #
    # # Do we have a list of data?
    # N = len(self.OpenFAST_outfile_list)
    # if N == 0:
    #     raise ValueError('Output files not defined! Populate: "FastPost.OpenFAST_outfile_list". \n Quitting FAST_Processing.')

    # # Do all the files exist?
    # files_exist = True
    # for i, flist in enumerate(self.OpenFAST_outfile_list):
    #     if isinstance(flist, str):
    #         if not os.path.exists(flist):
    #             print('Warning! File "{}" does not exist.'.format(
    #                 flist))
    #             self.OpenFAST_outfile_list.remove(flist)
    #     elif isinstance(flist, list):
    #         for fname in flist:
    #             if not os.path.exists(fname):
    #                 files_exist = False
    #                 if len(self.dataset_names) > 0:
    #                     print('Warning! File "{}" from {} does not exist.'.format(
    #                         fname, self.dataset_names[i]))
    #                     flist.remove(fname)
    #                 else:
    #                     print('Warning! File "{}" from dataset {} of {} does not exist.'.format(
    #                         fname, i+1, N))
    #                     flist.remove(fname)

    # # # load case matrix data to get descriptive case naming
    # # if self.fname_case_matrix == '':
    # #     print('Warning! No case matrix file provided, no case descriptions will be provided.')
    # #     self.case_desc = ['Case ID %d' % i for i in range(M)]
    # # else:
    # #     cases = load_case_matrix(self.fname_case_matrix)
    # #     self.case_desc = get_dlc_label(cases, include_seed=True)

    # # get unique file namebase for datasets
    # self.namebase = []
    # if len(self.dataset_names) > 0:
    #     # use filename safe version of dataset names
    #     self.namebase = ["".join([c for c in name if c.isalpha() or c.isdigit() or c in [
    #                                 '_', '-']]).rstrip() for i, name in zip(range(N), self.dataset_names)]
    # elif len(self.OpenFAST_outfile_list) > 0:
    #     # use out file naming
    #     if isinstance(self.OpenFAST_outfile_list[0], list):
    #         self.namebase = ['_'.join(os.path.split(flist[0])[1].split('_')[:-1])
    #                         for flist in self.OpenFAST_outfile_list]
    #     else:
    #         self.nsamebase = ['_'.join(os.path.split(flist)[1].split('_')[:-1])
    #                         for flist in self.OpenFAST_outfile_list]

    # # check that names are unique
    # if not len(self.namebase) == len(set(self.namebase)):
    #     self.namebase = []
    # # as last resort, give generic name
    # if not self.namebase:
    #     if isinstance(self.OpenFAST_outfile_list[0], str):
    #         # Just one dataset name for single dataset
    #         self.namebase = ['dataset1']
    #     else:
    #         self.namebase = ['dataset' + ('{}'.format(i)).zfill(len(str(N-1))) for i in range(N)]

    # # Run design comparison if filenames list has multiple lists
    # if (len(self.OpenFAST_outfile_list) > 1) and (isinstance(self.OpenFAST_outfile_list[0], list)):
    #     # Load stats and load rankings for design comparisons
    #     stats, load_rankings = self.design_comparison(self.OpenFAST_outfile_list)

    # else:
    #     # Initialize Analysis
    #     loads_analysis = Analysis.Loads_Analysis()
    #     loads_analysis.verbose = self.verbose
    #     loads_analysis.t0 = self.t0
    #     loads_analysis.tf = self.tf

    #     # run analysis in parallel
    #     if self.parallel_analysis:
    #         pool = mp.Pool(self.parallel_cores)
    #         try:
    #             stats_separate = pool.map(
    #                 partial(loads_analysis.full_loads_analysis, get_load_ranking=False), self.OpenFAST_outfile_list)
    #         except:
    #             stats_separate = pool.map(partial(loads_analysis.full_loads_analysis, get_load_ranking=False), self.OpenFAST_outfile_list[0])
    #         pool.close()
    #         pool.join()

    #         # Re-sort into the more "standard" dictionary/dataframe format we like
    #         stats = [pdTools.dict2df(ss).unstack() for ss in stats_separate]
    #         dft = pd.DataFrame(stats)
    #         dft = dft.reorder_levels([2, 0, 1], axis=1).sort_index(axis=1, level=0)
    #         stats = pdTools.df2dict(dft)

    #         # Get load rankings after stats are loaded
    #         load_rankings = loads_analysis.load_ranking(stats,
    #                                     names=self.dataset_names, get_df=False)

    #     # run analysis in serial
    #     else:
    #         # Initialize Analysis
    #         loads_analysis = Analysis.Loads_Analysis()
    #         loads_analysis.verbose = self.verbose
    #         loads_analysis.t0 = self.t0
    #         loads_analysis.tf = self.tf

    #         stats, load_rankings = loads_analysis.full_loads_analysis(self.OpenFAST_outfile_list, get_load_ranking=True)

    # if self.save_SummaryStats:
    #     if isinstance(stats, dict):
    #         fname = self.namebase[0] + '_stats.yaml'
    #         if self.verbose:
    #             print('Saving {}'.format(fname))
    #         save_yaml(self.results_dir, fname, stats)
    #     else:
    #         for namebase, st in zip(self.namebase, stats):
    #             fname = namebase + '_stats.yaml'
    #             if self.verbose:
    #                 print('Saving {}'.format(fname))
    #             save_yaml(self.results_dir, fname, st)
    # if self.save_LoadRanking:
    #     if isinstance(load_rankings, dict):
    #         fname = self.namebase[0] + '_LoadRanking.yaml'
    #         if self.verbose:
    #             print('Saving {}'.format(fname))
    #         save_yaml(self.results_dir, fname, load_rankings)
    #     else:
    #         for namebase, lr in zip(self.namebase, load_rankings):
    #             fname = namebase + '_LoadRanking.yaml'
    #             if self.verbose:
    #                 print('Saving {}'.format(fname))
    #             save_yaml(self.results_dir, fname, lr)

    # return stats, load_rankings


def design_comparison(self, filenames):
    """
    Compare design runs

    Parameters:
    ----------
    filenames: list
        list of lists, where the inner lists are of equal length.

    Returns:
    --------
    stats: dict
        dictionary of summary statistics data
    load_rankings: dict
        dictionary of load rankings
    """

    raise NotImplementedError()

    # # Make sure datasets are the same length
    # ds_len = len(filenames[0])
    # if any(len(dataset) != ds_len for dataset in filenames):
    #     raise ValueError('The datasets for filenames corresponding to the design comparison should all be the same size.')

    # fnames = np.array(filenames).T.tolist()
    # # Setup FAST_Analysis preferences
    # loads_analysis = Analysis.Loads_Analysis()
    # loads_analysis.verbose=self.verbose
    # loads_analysis.t0 = self.t0
    # loads_analysis.tf = self.tf
    # loads_analysis.ranking_vars = self.ranking_vars
    # loads_analysis.ranking_stats = self.ranking_stats

    # if self.parallel_analysis: # run analysis in parallel
    #     # run analysis
    #     pool = mp.Pool(self.parallel_cores)
    #     stats_separate = pool.map(partial(loads_analysis.full_loads_analysis, get_load_ranking=False), fnames)
    #     pool.close()
    #     pool.join()

    #     # Re-sort into the more "standard" dictionary/dataframe format we like
    #     stats = [pdTools.dict2df(ss).unstack() for ss in stats_separate]
    #     dft = pd.DataFrame(stats)
    #     dft = dft.reorder_levels([2, 0, 1], axis=1).sort_index(axis=1, level=0)
    #     stats = pdTools.df2dict(dft)

    #     # Get load rankings after stats are loaded
    #     load_rankings = loads_analysis.load_ranking(stats)

    # else: # run analysis in serial
    #     stats = []
    #     load_rankings = []
    #     for file_sets in filenames:
    #         st, lr = loads_analysis.full_loads_analysis(file_sets, get_load_ranking=True)
    #         stats.append(st)
    #         load_rankings.append(lr)

    # return stats, load_rankings
