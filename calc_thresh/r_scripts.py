import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

robjects.numpy2ri.activate()

robjects.r("""

require(mixtools)
require(parallel)
options("mc.cores"=8);

getmixtoolsRes <- function(combinedFCSSet, clusteringColumns){
mixtoolsRes <- mclapply(clusteringColumns, function(mrk, combinedFCSSet){
    for (i in 4:2) {
        lambdaInit <- rep(1/i, i)
        markerExpression<-tryCatch({
            mixResult = NULL
            if(i == 2){
                mixResult = normalmixEM(combinedFCSSet[,mrk], k = i ,epsilon = 1e-05, mu=c(min(combinedFCSSet[,mrk]), max(combinedFCSSet[,mrk])),verb=FALSE)
            } else {
                mixResult = normalmixEM(combinedFCSSet[,mrk], k = i ,epsilon = 1e-05, lambda = lambdaInit,verb=FALSE)
            }
            mixResult
        }, error = function(cond) {return(NULL)})
        if(!is.null(markerExpression)){
            break
        }
    }
    if (is.null(markerExpression)) {
        return(NaN)
    } else {
        # apply a k-mean with k=2 to derive the gate:
        k = kmeans(markerExpression$x, centers = c(min(markerExpression$x), max(markerExpression$x)))
        kMeansThresh = max(markerExpression$x[k[[1]] == 1])
        #assignment of each cell to the relevant Gaussian:
        assignments <- apply(markerExpression$posterior[,order(markerExpression$mu)],1,which.max)
        #order the Gaussians according to their mu's
        models <- data.frame(mu = markerExpression$mu[order(markerExpression$mu)], sigma = markerExpression$sigma[order(markerExpression$mu)],
                             lambda = sapply(1:i, function(x){sum(assignments == x)})/length(assignments))
        #calculate accumulated lambda (as the accumulated proportion of cells in each Gaussian)
        accum <- c(0,models$lambda[1])
        for(i in 2:nrow(models)){
          accum <- c(accum, accum[i] + models$lambda[i])
        }
        if (accum[length(accum)] > 1) {
          accum[length(accum)] = 1
        }
        #get the actual thresholds in the data according to the accumulated proportions:
        GaussiansThresh <- quantile(markerExpression$x, accum)
        #get the Gaussian that contains the k-means threshold:
        contGaus <- max(which(GaussiansThresh <= kMeansThresh))
        if(contGaus == -Inf) {
            contGaus = 1
        }
        if(contGaus == length(GaussiansThresh)) {
        return(GaussiansThresh[length(GaussiansThresh)-1])
        }
        # get the values of the cells included in this Gaussian
        valuesControversialModel <- markerExpression$x[markerExpression$x >= GaussiansThresh[contGaus] & markerExpression$x < GaussiansThresh[contGaus + 1]]
        #get the ratio between the values of cells in the controversial Gaussian and the k-means threshold:
        propUnderKMeans <- sum(valuesControversialModel < kMeansThresh)/length(valuesControversialModel)
        #if most cells in the controversial Gaussian are below the k-means threshold, consider this Gaussian as the negative population.
        #elseif most cells in this Gaussian are above the k-means threshold, consider this Gaussian as a positive population.
        if (propUnderKMeans < 0.5){
          finalThresh = GaussiansThresh[contGaus]
          tag <- 1:nrow(models) >=  contGaus
        }
        if (propUnderKMeans >= 0.5){
          finalThresh = GaussiansThresh[contGaus + 1]
          tag <- 1:nrow(models) >  contGaus
        }
        return(finalThresh)
      }
    }, combinedFCSSet)
  return(mixtoolsRes)
}
""")
