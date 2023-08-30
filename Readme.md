## Double Machine Learning (DoubleML): A Simulation Study of the Effect of Hyperparameter Variation on the Estimation of Causal Parameters

The modeling of partial linear regression models (PLR) is done based on the double/debiased machine learning framework by Chernozhukov et. al. (2018):

$$Y = {\theta_{0}D + g_{0}(X) + \zeta}$$

$$D = {m_{0}(X) + V}$$

where $E[\zeta|D,X]=0$ and $E[V|X]=0$.

### References

Chernozhukov, V.,  Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). “Double/debiased machine learning for treatment and structural parameters”. In: The Econometrics Journal 21.1, pp. C1–C68, DOI: 10.1111/ectj.120.97
