#!/usr/bin/env python
# coding=utf-
'''
Author: Wei Luo
Date: 2020-07-30 00:18:45
LastEditors: Wei Luo
LastEditTime: 2020-08-31 17:11:50
Note: Note
'''

import numpy as np
import time
from traj_gen import uav_poly_trajectory as pt

if __name__ == "__main__":
    # time_knots = np.array([0.0, 14.0, 16.5, 38.0, 40.0])
    time_knots = np.array([0.0, 5.0, 6.5, 8.0, 10.0])
    # time_knots = np.array([0.0, 1.68, 2.22, 2.76, 3.54])
    # time_knots = np.array([0.0, 1.1, 3.57])
    order = [8, 4]
    optimTarget = 'end-derivative'
    maxConti = [4, 2]
    pos_objWeights = np.array([0, 0, 0, 1])
    ang_objWeights = np.array([0, 1])
    pTraj = pt.UAVTrajGen(time_knots, order, maxContiOrder_=maxConti)

    # 2. Pin
    # ts = np.array([0.0, 4.0])
    # ts = np.array([0.0, 4.0, 6.5, 8.0, 10.0])
    ts = time_knots.copy()
    # ts = np.array([0.0, 1.68, 2.22, 2.76, 3.54])
    # ts = np.array([0.0, 1.1, 3.57])
    Xs = np.array([
                    [0.0, 0.0, 3.0, 0.0],
                    [0.5, 1.3, 2.0, -np.pi/2],
                    [0.8, 0.8, 1.5, 3*np.pi/4],
                    [0.6, 0.6, 1.0, 3*np.pi/4],
                    [0.4, 0.4, .5, 4*np.pi/4]
                   ])
    Xdot = np.array([0.2, 0, 0, 0])
    Xddot = np.array([0, 0, 0, 0])
    Xenddot = np.array([-0.1, -0.1, 0, 0])
    Xendddot = np.array([0, 0, 0, 0])
    # create pin dictionary
    for i in range(Xs.shape[0]):
        pin_ = {'t':ts[i], 'd':0, 'X':Xs[i]}
        pTraj.addPin(pin_)
    pin_ = {'t':ts[0], 'd':1, 'X':Xdot,}
    pTraj.addPin(pin_)
    pin_ = {'t':ts[0], 'd':2, 'X':Xddot,}
    pTraj.addPin(pin_)
    pin_ = {'t':ts[-1], 'd':1, 'X':Xenddot,}
    pTraj.addPin(pin_)
    pin_ = {'t':ts[-1], 'd':2, 'X':Xendddot,}
    pTraj.addPin(pin_)


    # # solve
    pTraj.setDerivativeObj(pos_objWeights, ang_objWeights)
    print("solving")
    time_start = time.time()
    pTraj.solve()
    time_end = time.time()
    # print(time_end - time_start)
    print(pTraj.pos_polyCoeffSet[0])
    # plot
    # showing trajectory
    print("trajectory")
    pTraj.showTraj(4)
    print('path')
    fig_title ='poly order : {0} / max continuity: {1} / minimized derivatives order: {2}'.format(pTraj.pos_order, pTraj.maxContiOrder, np.where(pTraj.pos_weight_mask>0)[0].tolist())
    pTraj.showUAVPath(fig_title)