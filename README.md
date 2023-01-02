# PINN-for-Black-Scholes

This Neural network algorithm solves Black-Scholes (BS) equation. While there are already analytical and numerical solution being used for this equation, the Physics-inforemd Neural Network offers unique advantages. It can capture the dynamics where the Call price valuation deviates from the BS equation. 

![image (4)](https://user-images.githubusercontent.com/35528280/210229095-3220be5c-2585-4e96-91c0-136354d7f5b0.png)

As evident in the figure, the numerical and analytical model follows an ideal Black-Scholes curve, but the real data might deviate from it. The datapoints are not sufficient enough, to do an independent data-driven prediction. PINN model considers both the ideal black Scholes curve and the deviated datapoints and thus develop a realistic model
