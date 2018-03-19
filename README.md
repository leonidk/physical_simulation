# Using Function Optimization To Find Policies: Ball Run
[CMU RI 16-745: Dynamic Optimization: Assignment 2](http://www.cs.cmu.edu/~cga/dynopt/ass2/)

Leonid Keselman and Alex Spitzer

## Part 1
We found bubble ball to be somewhat fun.

<img src="images/rotate_bball.jpg?raw=true">

## Part 2
We've decided to work on Box2D (which this repository is a fork of). We have fixed the common Ubuntu "GLSL 3.3 not supported" error that the normal Box2D repository has.

## Part 3
Make an optimizer that learns to move the ball as far to the right as possible with one obstacle bar 0.1m long. We use a physics coefficent of restitution of 0.75. Implement several optimization approaches, including a gradient-based approach, CMA-ES, and another non-derivative-based approach
| Optimizer | Runtime (ms) | Best | Mean | Std Dev |
|-------------------------------|--------------|-------|------|---------|
| CMA-ES (n=100) | 168 | -14.0 | 18.4 | 17.7 |
| CMA-ES (n=1000) | 318 | -15.1 | 12.5 | 21.4 |
| CMA-ES (n=1,popsize=80) | 60000 | -15.5 | None | None |
| Random (n=10) | 57 | -7.4 | 26.9 | 8.3 |
| Random (n=100) | 575 | -12.4 | 13.9 | 14.5 |
| Conjugate Gradient (eps=1e-1) | 92 | -6.6 | 29.5 | 4.7 |
| Conjugate Gradient (eps=1e-8) | 50 | 30.0 | 30.1 | 1.0 |
| SLSQP (eps=1e-1) | 45 | -8.1 | 29.6 | 3.8 |
| SLSQP (eps=1e-8) | 30 | 17.0 | 30.0 | 1.7 |
| Diff Evolution (n=10) | 1970 | -14.7 | 6.4 | 18.1 |
| Diff Evolution (n=100) | 59618 | -15.4 | 6.1 | 21.0 |

## Part 4
A vision system sees obstacles and a goal at obstacles.txt in pixel coordinates (so positive Y is down, you should fix that). The ball is dropped at (431, 181) in pixel coordinates. The obstacles are really 0.25m long and 0.037m high, so you also need to convert the pixel values to meters. Use optimization to find a simulated setup that is "similar" that gets the ball in the goal. A video (slow motion) of the actual ball on this run.

## Part 5
For the system in part 4, an observed trial has the following ball trajectory: trajectory.txt, sampled at 30 frames per second in pixel coordinates. Use optimization to adjust the parameters of the simulation so that the simulated trajectory matches the observed trajectory, with the obstacles back in the observed positions. Parameters to change might include gravity, air resistance, something to do with rolling vs. sliding (friction, moment of inertia of the ball, some parameters you make up, ...), and in the ODE simulation the bounce parameters in dynamics.cpp:

## Part 6
Construct an interesting simulated ball run with four obstacles, and send Chris the obstacle locations (in metric coordinates). We will run it in reality, and send you back a video and ball trajectory that actually happened. You then modify the obstacle locations, and we repeat, until the desired behavior happens in reality.

## Part 7
{**IMPLEMENT BUBBLE BALL LEVEL?**}

