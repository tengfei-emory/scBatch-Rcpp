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
arma::mat scBatch(arma::mat c, arma::mat w, arma::mat d, int m, double max, double step, double tol, Rcpp::Function derif){
  //NumericMatrix d, NumericMatrix core, NumericVector idx) {
  arma::wall_clock timer;
  
  int p = c.n_rows; int n = c.n_cols;
  arma::mat core = c.t()*(eye(p,p)-ones(p,p)/p)*(eye(p,p)-ones(p,p)/p)*c;
  //core = t(count.mat)%*%t((diag(p) - matrix(1,p,p)/p))%*%(diag(p) - matrix(1,p,p)/p)%*%count.mat
  core = (core+core.t())/2;
  
  for(int i = 0; i < max; ++i) {
    arma::vec group = randi<vec>(n,arma::distr_param(0,m-1));
    for(int k = 0; k < m; ++k){
      timer.tic();
      arma::uvec idx = arma::find(group == k);
      Rcpp::List fdf = derif(c,w,d,core,idx);
      double f = fdf["f"];
      arma::mat df = fdf["df"];
      for(int j = 0; j < 5; ++j){
        arma::mat u = w - step*df;
        arma::vec ln(n); ln.fill(1);
        u = u/(ln*(arma::max(arma::abs(u),0)));
        arma::mat nc = c*u;
        arma::mat A = 1-cor(nc);
        double fnew = accu(pow(A-d,2));
        
        if (fnew >= f){
          step = 0.5*step;
        }else{
          step = 1.5*step;
          w = u;
          double n = timer.toc();
          
          cout << k << " time elapsed: " << n << " L: " << fnew << " step size: " << step << endl;
          //cout << fnew << endl;
          break;
        }
      }
    }
    if (step < tol){
      break;
    }
  }
  
  //Rcpp::List ret;
  //ret["f"] = f;
  //ret["df"] = df;
  //return ret;
  
  arma::mat nc = c*w;
  
  return nc;
}


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R

*/
