# TrajGen_Py

Python Implementation for Trajectory Generator.

## requirements (tested)

- Python 3.8, it should work up at least Python 3.5
- CasADi >= v3.5.1, for solving QP/OP
- qpsolvers and cvxopt, they provide more efficient way to handle QP problem directly.

  ```bash
    conda/pip install qpsolvers cvxopt
  ```

- mosek 9.2, can also solve this task by converting the original problem to a conic problem.

## file declaration

1. *poly_3d_example.py*

    This file is based on the algorithm in paper [Minimum snap trajectory generation and control for quadrotors](https://ieeexplore.ieee.org/document/5980409).

    It utilizes the class which is defined in *traj_gen/poly_trajectory.py*. The inputs of this trajectory generator are the user-defined positions **Xs** and their time stamp **knots** and **ts**. These fixed positions will be converted into fixed constrains for the optimization algorithm. One can also provide some veclocity/acceleration constrains or even flexible position limitations with time, which should be stay within the fixed time sequence.

    By default, the algorithm has calculated the 4-th derivation through the objective weight vector **objWeights**. One can also change this vector values according to the requirements.

    Two possible algorithms are implemented, one utilizes the original polynomial coefficient algorithm, and the second is based on the above paper which has better performance. One can select the algorithm by defining the parameter **optimTarget** using 'poly-coeff' or 'end-derivative', respectively.

    Additionally, a updated version using **qpsolvers** and **cvxopt** is implemented in *traj_gen/poly_trajectory_qp.py*. They solve the same problem more efficiently. Besides, a MOSEK based version is also provided. However, you need to set up the licence according to the official website from [MOSEK](https://www.mosek.com).

2. *chomp_3d_example.py*

    This file is based on the work in [CHOMP: Gradient Optimization Techniques for Efficient Motion Planning](https://www.ri.cmu.edu/pub_files/2009/5/icra09-chomp.pdf). It only requires the user to define the time step when the trajectory begins and ends, using **knots**. The rest of the code is similar to *poly_3d_example.py*.

3. *uav_4d_traj.py*

   Based on the poly_3d_example, this example only shows possible application of the polynomial trajectory generator. Since the features of the quadcopter, the yaw angle should not change rapidly. Therefore, for the yaw trajectory one doesn't need to calculate high degree polynomial.
