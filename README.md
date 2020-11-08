# Generating color-coded scatterplot matrices in matplotlib


## scatterplot_matrix.py
This script generates scatterplot matrices for an arbitrary number of data columns, given as a pandas dataframe.
The user can choose which columns are used for color-coding the scatterplots, separately for the upper and lower triangle of the matrix.
Moreover, the user can chose a data transformation (Percentiles, standardization, column-wise (0,1) scaling)

Exemplary results using the Boston House Prices Dataset (https://scikit-learn.org/stable/datasets/index.html#boston-dataset)

use_ranks=True:

<img width="500" alt="java 8 and prio java 8  array review example" src="https://github.com/johannesuhl/scatterplot_matrix/blob/main/scatterplot_matrix_ranks.jpg">

transform_to_01=True:

<img width="500" alt="java 8 and prio java 8  array review example" src="https://github.com/johannesuhl/scatterplot_matrix/blob/main/scatterplot_matrix_01.jpg">

standardize=True:

<img width="500" alt="java 8 and prio java 8  array review example" src="https://github.com/johannesuhl/scatterplot_matrix/blob/main/scatterplot_matrix_std.jpg">

no transformation:

<img width="500" alt="java 8 and prio java 8  array review example" src="https://github.com/johannesuhl/scatterplot_matrix/blob/main/scatterplot_matrix_raw.jpg">


## scatterplot_crosscorr_matrix.py
This script generates combined scatterplot-crosscorrelation matrices for an arbitrary number of data columns, given as a pandas dataframe.
The user can choose which columns are used for color-coding the scatterplots, in the upper triangle of the matrix, the lower triangle contains the crosscorrelation matrix usign Pearson's correlation coefficient.
Moreover, the user can chose a data transformation (Percentiles, standardization, column-wise (0,1) scaling) for the scatterplots.

Exemplary results using the Boston House Prices Dataset (https://scikit-learn.org/stable/datasets/index.html#boston-dataset)

use_ranks=True:

<img width="500" alt="java 8 and prio java 8  array review example" src="https://github.com/johannesuhl/scatterplot_matrix/blob/main/scatterplot_matrix_w_crosscorr_ranks.jpg">

transform_to_01=True:

<img width="500" alt="java 8 and prio java 8  array review example" src="https://github.com/johannesuhl/scatterplot_matrix/blob/main/scatterplot_matrix_w_crosscorr_01.jpg">

standardize=True:

<img width="500" alt="java 8 and prio java 8  array review example" src="https://github.com/johannesuhl/scatterplot_matrix/blob/main/scatterplot_matrix_w_crosscorr_std.jpg">

no transformation:

<img width="500" alt="java 8 and prio java 8  array review example" src="https://github.com/johannesuhl/scatterplot_matrix/blob/main/scatterplot_matrix_w_crosscorr_raw.jpg">




