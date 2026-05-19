# SRP-2-semester
These codes incarnate the work of Hysteretic Hopfield Neural Network (HHNN) with all modifications made studied in paper.

The program called "nqueen_final_test(N=30).py" was used to conduct final tests for the case when N=30.
The optimal parameters alpha_learn_rate = 0.004 and beta_learn_rate = 0.002 were obtained by using the program "nqueen_GS30.py" via grid search. 

To find optimal parameters for the case N = 50 due to significant expansion of the search area more precise instrument, namely Optuna library, was used.
This library uses Tree-structured Parzen Estimator algorithm and pruning what allows it to build a probabilistic mathematical model of where good results are and where bad ones are
and this significantly speeds up the search for global parameters. 
At first the optimal discretization step was found for HNN by using program "nqueen_HNN_OptunaSearch.py". After that with optimal discretization step optimal alpha and beta learning rates were found
by using program "nqueen_OptunaSearch50.py". With the obtained optimal parameters, final tests were carried out by running the "nqueen_final50.py" program.
