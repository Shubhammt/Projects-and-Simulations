the code generates 3 .csv files
1)predictions of standard linear regression in "standard_linear_preds.csv"
2)predictions of overfiting case of standard linear regression in "standard_linear_preds_overfit.csv"
3)predictions of ridge regression in "ridge_preds.csv"

The code is divided into 3 segements for each of the above cases, separated by a "########..."

To simplify the structure of code we defined a linear_reg class and put all the necessary functions
as its member functions. The functions are
1) RSS: it calculates the residual sum of squares (Y-XB)^T(Y-XB)
2) basis_expand: it calculates the polynomial terms of degree n, of the input vector X
3) standard_lin_reg: it solves for the coefficients of the least squared solution of standard linear
by
	B=(XTX)^-1XTY
4) ridge_lin_reg: it solves for the coefficients of the least squared solution of ridge rgression
by
	B=(XTX+lambda I)^-1XTY