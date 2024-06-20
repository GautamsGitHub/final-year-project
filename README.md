# Polymatrix Games

Finding the Nash Equilibria of games


## nf_and_polymatrix.py

Provides classes for describing Normal Form Games and
Polymatrix Games and provides many useful methods on them.


## ADIDAS

Implements the ADIDAS algorithm proposed by the paper:
Sample-based Approximation of Nash in Large Many-Player Games via Gradient Descent
https://doi.org/10.48550/arXiv.2106.01285


## Polymatrix Game Linear Complementarity Solver.

Implements Howson's algorithm for finding equilibrium in
a Polymatrix Game by solving a LCP. Utilizes methods from QuantEcon.
All other Python implementations so far seem to only support
Bimatrix Games. GameTracer has an implementation in C.

Howson's paper:
Equilibria of Polymatrix Games
https://www.jstor.org/stable/2634798


## Autograd Attempt

Makes our LCP solver compatible with JAX so that we can
try to use automatic differentiation on it. We then use
the solver as a layer in a neural network.


## Quadratic Programming

Converts our LCP into QP form but games in general do not
lead to nice QPs. We try a couple of QP solvers.


## ***LCP Layer in NEural Network***

Uses InferOpt to wrap our LCP solver and puts it into a Neural Network
where we combine it with our own loss estimater and loss gradients
to optimise the Neural Network to predict Polymatrix approximations of
Normal Form Games. ***This is novel***
