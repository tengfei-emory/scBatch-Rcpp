//#include <Rcpp.h>

#include <RcppArmadillo.h>
// [[Rcpp::depends( RcppArmadillo)]]
using namespace arma;


// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::export]]
Rcpp::List derif(arma::mat c, arma::mat w, arma::mat d, arma::mat core, arma::uvec idx){
                      //NumericMatrix d, NumericMatrix core, NumericVector idx) {
  int n = c.n_cols;
  arma::mat nc = c * w;
  arma::mat a = 1-cor(nc);
  arma::mat amd = a-d;
  double f; f = accu(pow(amd,2));
  
  //arma::mat W = sqrt(w.t() * core * w);
  arma::mat W = w.t() * core * w;
  arma::uvec idxw = arma::find(W < 0);
  W(idxw) = W(idxw)- W(idxw);
  W = sqrt((W+W.t())/2);
  arma::mat M = core * w;
  // create some intermediate quatities
  arma::mat invW = 1/W;
  arma::mat invWdiag = invW.diag();
  arma::vec ln(n); ln.fill(1);
  
  //arma::mat a1 = kron(invW.diag(),M/(ln*W.diag().t()));
  arma::mat a1 = M/(ln*W.diag().t());
  arma::vec b1 = vectorise(M/(ln*pow(W.diag(),3).t()));
  arma::mat b20 = pow(W,2)/(ln*pow(W.diag(),2).t());
  arma::vec b2 = vectorise(b20.t());
  
  arma::mat df(n,n); df.fill(0);
  int lidx = idx.n_elem;
  
  for(int k = 0; k < lidx; ++k) {
    int i = idx(k)+1;
    arma::mat a11 = invWdiag(i)*a1;
    //arma::mat D = a1.rows(n*(i-1),n*(i)-1) - (b1.subvec(n*(i-1),n*(i)-1)*b2.subvec(n*(i-1),n*(i)-1).t());
    arma::mat D = a11 - (b1.subvec(n*(i-1),n*(i)-1)*b2.subvec(n*(i-1),n*(i)-1).t());
    df.col(i-1) = D.t() * amd.col(i-1);
    arma::mat damd = D.t() * amd;
    df(i-1,i-1) = sum(damd.diag());
  }
  
  df = -df;
  
  Rcpp::List ret;
  ret["f"] = f;
  ret["df"] = df;
  return ret;
  
  return ret;
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R

*/
