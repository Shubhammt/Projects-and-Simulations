the code generates 4 .csv files
1)predictions of perceptron algorithm in "perceptron_preds.csv"
2)predictions of discrimant function when Cov=(sigma^2)I in  "discrminant_same_variance_preds.csv"
3)predictions of discrimant function when C1=C2 in "discrminant_same_covariance_preds.csv"
4)predictions of discrimant function when C1!=C2 in "discrminant_diff_covariance_preds.csv"


The code is divided into 4 segements for each of the above cases, separated by a "########..."

To simplify the structure of code we defined a classification class and put all the necessary functions
as its member functions. The functions are
1) percptron: returns the weight vector with norm 1, obtained by implementing the perceptron algorithm
over the training data
2) cov_and_mu_estimate: it returns the following quantities
	a) mean of feature vector with class label 1
	b) mean of feature vector with class label 0
	c) pooled estimate for the case when Cov=(sigma^2)I
	d) covariance for the case when C1=C2
	e) covariance for the case when C1!=C2
3) discrminant_same_variance: calculates the discriminant function for the case Cov=(sigma^2)I
and C1=C2
4) discrminant_diff_variance: calculates the discriminant function for the case C1!=C2
5) bound_disc_same_variance: calculates the discriminant classifier boundary for the case Cov=(sigma^2)I
and C1=C2
6) bound_disc_diff_variance: calculates the discriminant classifier boundary for the case C1!=C2
7) confusion_perceptron: calculates confusion matrix of perceptron
8) confusion_disc_same_var: calculates confusion matrix of discriminant classifier for the case Cov=(sigma^2)I
and C1=C2
9) confusion_disc_diff_var: calculates confusion matrix of discriminant classifier for the case C1!=C2
