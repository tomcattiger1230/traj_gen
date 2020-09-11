#!/usr/bin/env python
# coding=utf-8

from .traj_gen_base import TrajGen
import numpy as np
import casadi as ca # using casadi for the optimization problem
from qpsolvers import solve_qp # using qpsolvers library for the optimization problem
from scipy.linalg import block_diag, solve


class UAVTrajGen(TrajGen):
    def __init__(self, knots_:np.array, order_:list, dim_=4, maxContiOrder_=[4, 3], pos_dim=3):
        """
        initialize the class
            Args:
                knots_: time knots to define the fixed pins
                order_: derivetiy order of the polynomial equation; here should be a list which
                        contains the position derivative and angle derivative
                dim_: as default the dimension of the uav trajectory has four (x, y, z, yaw)
                maxContiOrder_: max continuity order
        """
        super().__init__(knots_, dim_)
        self.pos_order, self.ang_order = order_ # polynomial order
        self.position_dim = pos_dim
        self.maxContiOrder = maxContiOrder_
        self.num_segments = knots_.shape[0] - 1 # segments which is knots - 1
        self.num_pos_variables = (self.pos_order+1) * self.num_segments
        self.num_ang_variables = (self.ang_order+1) * self.num_segments
        self.pos_polyCoeffSet = np.zeros((self.position_dim, self.pos_order+1, self.num_segments))
        self.ang_polyCoeffSet = np.zeros((dim_-self.position_dim, self.ang_order+1, self.num_segments))
        self.segState = np.zeros((self.num_segments, 3)) # 0 dim -> how many fixed pins in this segment,
                                              # muss smaller than the polynomial order+1
                                              # more fixed pins (higher order) will be ignored.
                                              # 1 dim -> continuity degree. should be defined by
                                              # user (maxContiOrder_+1)

    ## math functions
    def scaleMat(self, delT, num_polys):
        mat_ = np.diag([delT**i for i in range(num_polys)])
        return mat_

    def scaleMatBigInv(self, poly_type:str):
        """
        inverse matrix of time
            Args:
                poly_type: the type of the polynomial (the number of the)
        """
        mat_ = None
        num_polys_ = self.pos_order + 1 if poly_type == 'pos' else self.ang_order + 1
        for m in range(self.num_segments):
            matSet_ = self.scaleMat(1/(self.Ts[m+1]-self.Ts[m]), num_polys_)
            if mat_ is None:
                mat_ = matSet_.copy()
            else:
                mat_ = block_diag(mat_, matSet_)
        return mat_

    ## functional definition
    def setDerivativeObj(self, pos_weights, ang_weights):
        """
        Setup the which derivaties will be calculated in the cost function

        """
        if pos_weights.shape[0] > self.pos_order:
            print("Position order of derivative objective > order of poly. Higher terms will be ignored.")
            self.pos_weight_mask = pos_weights[:self.pos_order]
        else:
            self.pos_weight_mask = pos_weights

        if ang_weights.shape[0] > self.ang_order:
            print("Angle order of derivative objective > order of poly. Higher terms will be ignored.")
            self.ang_weight_mask = ang_weights[:self.ang_order]
        else:
            self.ang_weight_mask = ang_weights

    def findSegInteval(self, t_):
        idx_ = np.where(self.Ts<=t_)[0]
        if idx_.shape[0]>0:
            m_ = np.max(idx_)
            if m_ >= self.num_segments:
                if t_ != self.Ts[-1]:
                    print('Eval of t : geq TM. eval target = last segment')
                m_ = self.num_segments-1
        else:
            print('Eval of t : leq T0. eval target = 1st segment')
            m_ = 0
        tau_ = (t_-self.Ts[m_])/(self.Ts[m_+1]-self.Ts[m_])
        return m_, tau_

    def addPin(self, pin_):
        t_ = pin_['t']
        X_ = pin_['X']
        super().addPin(pin_)
        m, _ = self.findSegInteval(t_)
        if len(X_.shape) == 2: # 2 dimension ==> loose pin
            if m in self.loosePinSet.keys():
                self.loosePinSet[m].append(pin_)
            else:
                self.loosePinSet[m] = [pin_]
        elif len(X_.shape) == 1: # vector ==> fix pin
            assert (t_==self.Ts[m] or t_==self.Ts[-1]), 'Fix pin should be imposed only knots'
            if self.segState[m, 0] <= self.num_pos_variables+1:
                if m in self.fixPinSet.keys():
                    self.fixPinSet[m].append(pin_)
                    self.fixPinOrder[m].append(pin_['d'])
                else:
                    self.fixPinSet[m] = [pin_]
                    self.fixPinOrder[m] = [pin_['d']]
                self.segState[m, 0] += 1

            else:
                print('FixPin exceed the dof of this segment. Pin ignored')
        else:
            print('Dim of pin value is invalid')

    def nthCeoff(self, n, d):
        """ Returns the nth order ceoffs (n=0...N) of time vector of d-th
        derivative.

        Args:
            n(int): target order
            d(int): order derivative

        Returns:
        val_: n-th ceoffs
        """
        if d == 0:
            val_ = 1
        else:
            accumProd_ = np.cumprod(np.arange(n, n-d, -1))
            val_ = accumProd_[-1]*(n>=d)
        return val_

    def IntDerSquard(self, d, poly_order):
        """
        {x^(d)(t)}^2  = (tVec(t,d)'*Dp)'*(tVec(t,d)'*Dp)

        Args:
            d(int): order derivative
            poly_order(int): max order of the polynomial

        Returns:
            mat_: matrix of the cost function
        """
        mat_ = np.zeros((poly_order+1, poly_order+1))
        if d > poly_order:
            print("Order of derivative > poly order, return zeros-matrix \n")
        for i in range(d, poly_order+1):
            for j in range(d, poly_order+1):
                mat_[i,j] = self.nthCeoff(i, d) * self.nthCeoff(j, d) / (i+j-2*d+1)
        return mat_

    def tVec(self, t_, d_, poly_order):
        # time vector evaluated at time t with d-th order derivative.
        vec_ = np.zeros((poly_order+1, 1))
        for i in range(d_, poly_order+1):
            vec_[i] = self.nthCeoff(i, d_)*t_**(i-d_)
        return vec_

    def fixPinMatSet(self, pin, PolyType):
        t_ = pin['t']
        X_ = pin['X']
        d_ = pin['d']
        m_, tau_ = self.findSegInteval(t_)
        dTm_ = self.Ts[m_+1] - self.Ts[m_]
        if PolyType == 'pos':
            poly_order = self.pos_order
            idxStart_ = m_*(poly_order+1)
            idxEnd_ = (m_+1)*(poly_order+1)
            aeqSet_ = np.zeros((self.position_dim, self.num_pos_variables))
            beqSet_ = np.zeros((self.position_dim, 1))
            for dd in range(self.position_dim):
                aeqSet_[dd, idxStart_:idxEnd_] = self.tVec(tau_, d_, poly_order).flatten()/dTm_**d_#
                beqSet_[dd] = X_[dd]
        elif PolyType == 'ang':
            poly_order = self.ang_order
            idxStart_ = m_*(poly_order+1)
            idxEnd_ = (m_+1)*(poly_order+1)
            aeqSet_ = np.zeros((self.dim-self.position_dim, self.num_ang_variables))
            beqSet_ = np.zeros((self.dim-self.position_dim, 1))
            for dd in range(self.dim-self.position_dim):
                aeqSet_[dd, idxStart_:idxEnd_] = self.tVec(tau_, d_, poly_order).flatten()/dTm_**d_#
                beqSet_[dd] = X_[self.position_dim+dd]
        else:
            print("ERROR: please define polynomial for 'pos' or 'ang'")
        return aeqSet_, beqSet_

    def contiMat(self, m_, dmax, poly_order):
        """
        ensure in dmax derivative degree the curve should be continued.
        from 0 to dmax derivative
        Args:
            m_: index of the segment <= M-1
            dmax: max conti-degree
            poly_order: max poly order
        """
        dmax_ = int(dmax)
        if poly_order == self.pos_order:
            aeq_ = np.zeros((dmax_+1, self.num_pos_variables))
        else:
            aeq_ = np.zeros((dmax_+1, self.num_ang_variables))
        beq_ = np.zeros((dmax_+1, 1)) # different of the eq should be zero
        idxStart_ = m_*(poly_order+1)
        idxEnd_ = (m_+2)*(poly_order+1) # end of the next segment
        dTm1_ = self.Ts[m_+1] - self.Ts[m_]
        dTm2_ = self.Ts[m_+2] - self.Ts[m_+1]
        for d in range(dmax_+1):
            # the end of the first segment should be the same as the begin of the next segment at each derivative
            aeq_[d, idxStart_:idxEnd_] = np.concatenate((self.tVec(1, d, poly_order)/dTm1_**d, - self.tVec(0, d, poly_order)/dTm2_**d), axis=0).flatten() #
        return aeq_, beq_

    def loosePinMatSet(self, pin_, poly_order):
        """
        loose pin setup
        """
        t_ = pin_['t']
        X_ = pin_['X']
        d_ = pin_['d']
        m_, tau_ = self.findSegInteval(t_)
        dTm_ = self.Ts[m_+1] - self.Ts[m_]
        idxStart_ = m_*(poly_order+1)
        idxEnd_ = (m_+1)*(poly_order+1)

        if poly_order == self.pos_order:
            aSet_ = np.zeros((self.position_dim, 2, self.num_pos_variables))
            bSet_ = np.zeros((self.position_dim, 2, 1))
            for dd in range(self.position_dim):
                aSet_[dd, :, idxStart_:idxEnd_] = np.array([self.tVec(tau_, d_, poly_order)/dTm_**d_,-self.tVec(tau_, d_, poly_order)/dTm_**d_]).reshape(2, -1) #
                bSet_[dd, :] = np.array([X_[dd, 1], -X_[dd, 0]]).reshape(2, -1)
        else:
            aSet_ = np.zeros((self.dim-self.position_dim, 2, self.num_ang_variables))
            bSet_ = np.zeros((self.dim-self.position_dim, 2, 1))
            for dd in range(self.dim-self.position_dim):
                aSet_[dd, :, idxStart_:idxEnd_] = np.array([self.tVec(tau_, d_, poly_order)/dTm_**d_,-self.tVec(tau_, d_, poly_order)/dTm_**d_]).reshape(2, -1) #
                bSet_[dd, :] = np.array([X_[dd, 1], -X_[dd, 0]]).reshape(2, -1)
        return aSet_, bSet_

    def coeff2endDerivatives(self, Aeq_):
            assert Aeq_.shape[1] <= self.num_pos_variables, 'Pin + continuity constraints are already full. No dof for optim.'
            mapMat_ = Aeq_.copy()
            for m in range(self.num_segments):
                freePinOrder_ = np.setdiff1d(np.arange(self.pos_order+1), self.fixPinOrder[m]) # free derivative (not defined by fixed pin)
                dof_ = self.pos_order+1 - np.sum(self.segState[m, :2])
                freeOrder = freePinOrder_[:int(dof_)]
                for order in freeOrder:
                    virtualPin_ = {'t':self.Ts[m], 'X':np.zeros((self.position_dim, 1)), 'd':order}
                    aeqSet_, _ = self.fixPinMatSet(virtualPin_, 'pos')
                    aeq_ = aeqSet_[0] # only one dim is taken.
                    mapMat_ = np.concatenate((mapMat_, aeq_.reshape(-1, self.num_pos_variables)), axis=0)
            return mapMat_

    def getQPset(self, PolyType):
        AeqSet = None
        ASet = None
        BeqSet = None
        BSet = None
        if PolyType == 'pos':
            QSet = np.zeros((self.position_dim, self.num_pos_variables, self.num_pos_variables))
            for dd in range(self.position_dim):
                Q_ = np.zeros((self.num_pos_variables, self.num_pos_variables))
                for d in range(1, self.pos_weight_mask.shape[0]+1):
                    if self.pos_weight_mask[d-1] > 0:
                        Qd_ = None
                        for m in range(self.num_segments):
                            dT_ = self.Ts[m+1] - self.Ts[m]
                            Q_m_ = self.IntDerSquard(d, self.pos_order)/dT_**(2*d-1)
                            if Qd_ is None:
                                Qd_ = Q_m_.copy()
                            else:
                                Qd_ = block_diag(Qd_, Q_m_)
                        Q_ = Q_ + self.pos_weight_mask[d-1]*Qd_
                QSet[dd] = Q_

            for m in range(self.num_segments):
                ## fix pin
                if m in self.fixPinSet.keys():
                    for pin in self.fixPinSet[m]:
                        aeqSet, beqSet = self.fixPinMatSet(pin, 'pos')
                        if AeqSet is None:
                            AeqSet = aeqSet.reshape(self.position_dim, -1, self.num_pos_variables)
                            BeqSet = beqSet.reshape(self.position_dim, -1, 1)
                        else:
                            AeqSet = np.concatenate((AeqSet, aeqSet.reshape(self.position_dim, -1, self.num_pos_variables)), axis=1)
                            BeqSet = np.concatenate((BeqSet, beqSet.reshape(self.position_dim, -1, 1)), axis=1)
                    ## continuity
                    if m < self.num_segments-1:
                        contiDof_ = min(self.maxContiOrder[0]+1, self.num_pos_variables+1-self.segState[m, 0])
                        self.segState[m, 1] = contiDof_
                        if contiDof_ != self.maxContiOrder[0]+1:
                            print('Connecting segment ({0},{1}) : lacks {2} dof  for imposed {3} th continuity'.format(m, m+1, self.maxContiOrder[0]-contiDof_, self.maxContiOrder[0]))
                        if contiDof_ >0:
                            aeq, beq = self.contiMat(m, contiDof_-1, self.pos_order)
                            AeqSet = np.concatenate((AeqSet, aeq.reshape(1, -1, self.num_pos_variables).repeat(self.position_dim, axis=0)), axis=1)
                            BeqSet = np.concatenate((BeqSet, beq.reshape(1, -1, 1).repeat(self.position_dim, axis=0)), axis=1)
                if m in self.loosePinSet.keys():
                    for pin in self.loosePinSet[m]:
                        aSet, bSet = self.loosePinMatSet(pin, self.pos_order)
                        if ASet is None:
                            ASet = aSet.copy()
                            BSet = bSet.copy()
                        else:
                            ASet = np.concatenate((ASet, aSet), axis=1)
                            BSet = np.concatenate((BSet, bSet), axis=1)

        elif PolyType == 'ang':
            QSet = np.zeros((self.dim-self.position_dim, self.num_ang_variables, self.num_ang_variables))
            for dd in range(self.dim-self.position_dim):
                Q_ = np.zeros((self.num_ang_variables, self.num_ang_variables))
                for d in range(1, self.ang_weight_mask.shape[0]+1):
                    if self.ang_weight_mask[d-1] > 0:
                        Qd_ = None
                        for m in range(self.num_segments):
                            dT_ = self.Ts[m+1] - self.Ts[m]
                            Q_m_ = self.IntDerSquard(d, self.ang_order)/dT_**(2*d-1)
                            if Qd_ is None:
                                Qd_ = Q_m_.copy()
                            else:
                                Qd_ = block_diag(Qd_, Q_m_)
                        Q_ = Q_ + self.ang_weight_mask[d-1]*Qd_
                QSet[dd] = Q_
            for m in range(self.num_segments):
                ## fix pin
                if m in self.fixPinSet.keys():
                    for pin in self.fixPinSet[m]:
                        aeqSet, beqSet = self.fixPinMatSet(pin, 'ang')
                        if AeqSet is None and aeqSet is not None:
                            AeqSet = aeqSet.reshape(self.dim-self.position_dim, -1, self.num_ang_variables)
                            BeqSet = beqSet.reshape(self.dim-self.position_dim, -1, 1)
                        elif aeqSet is not None:
                            AeqSet = np.concatenate((AeqSet, aeqSet.reshape(self.dim-self.position_dim, -1, self.num_ang_variables)), axis=1)
                            BeqSet = np.concatenate((BeqSet, beqSet.reshape(self.dim-self.position_dim, -1, 1)), axis=1)
                        else:
                            pass
                    # continuity
                    if m < self.num_segments-1:
                        contiDof_ = min(self.maxContiOrder[1]+1, self.num_ang_variables+1-self.segState[m, 0])
                        self.segState[m, 2] = contiDof_
                        if contiDof_ != self.maxContiOrder[1]+1:
                            print('Connecting segment ({0},{1}) : lacks {2} dof  for imposed {3} th continuity'.format(m, m+1, self.maxContiOrder[1]-contiDof_, self.maxContiOrder[1]))
                        if contiDof_ >0:
                            aeq, beq = self.contiMat(m, contiDof_-1, self.ang_order)
                            AeqSet = np.concatenate((AeqSet, aeq.reshape(1, -1, self.num_ang_variables).repeat(self.dim-self.position_dim, axis=0)), axis=1)
                            BeqSet = np.concatenate((BeqSet, beq.reshape(1, -1, 1).repeat(self.dim-self.position_dim, axis=0)), axis=1)
        else:
            print("ERROR: please use 'pos' or 'ang'")

        return QSet, ASet, BSet, AeqSet, BeqSet

    def mapQP(self, QSet_, ASet_, BSet_, AeqSet_, BeqSet_):
        Afp_ = self.coeff2endDerivatives(AeqSet_[0]) # sicne all Aeq in each dim are the same
        AfpInv_ = np.linalg.inv(Afp_)
        Nf_ = int(AeqSet_[0].shape[0])
        Qtemp_ = np.dot(np.dot(AfpInv_.T, QSet_[0]), AfpInv_)
        # Qff_ = Qtemp_[:Nf_, :Nf_]
        Qfp_ = Qtemp_[:Nf_, Nf_:]
        Qpf_ = Qtemp_[Nf_:, :Nf_]
        Qpp_ = Qtemp_[Nf_:, Nf_:]
        QSet = np.zeros((self.position_dim, self.num_pos_variables-Nf_, self.num_pos_variables-Nf_))
        HSet = np.zeros((self.position_dim, self.num_pos_variables-Nf_))
        # check ASet ?
        if ASet_ is not None:
            ASet = np.zeros((self.position_dim, ASet_.shape[1], self.num_pos_variables-Nf_))
            BSet = BSet_.copy()
            dp_ = None
            for dd in range(self.position_dim):
                df_ = BeqSet_[dd]
                QSet[dd] = 2*Qpp_
                HSet[dd] = np.dot(df_.T, (Qfp_+Qpf_.T))
                A_ = np.dot(ASet_[dd], AfpInv_)
                ASet[dd] = A_[:, Nf_:]
                BSet[dd] = BSet_[dd] - np.dot(A_[:, :Nf_], df_)
        else:
            ASet = None
            BSet = None
            # directly solving the problem without making an optimization problem
            dp_ = np.zeros((self.position_dim, self.num_pos_variables-Nf_))
            for dd in range(self.position_dim):
                df_ = BeqSet_[dd]
                dp_[dd] = np.dot(np.dot(-np.linalg.inv(Qpp_), Qfp_.T), df_).flatten()

        return QSet, HSet, ASet, BSet, dp_

    def check_angle_data(self,):
        # currently only use yaw angle
        # temp_ = np.zersos(self.dim-self.position_dim)
        temp_ = 0
        for m in range(self.num_segments):
            if m in self.fixPinSet.keys():
                for pin in self.fixPinSet[m]:
                    if pin['d'] == 0:
                        current_yaw = pin['X'][-1]
                        if current_yaw - temp_ - np.pi > 0.0:
                            temp_ = -2*np.pi + current_yaw
                            pin['X'][-1] = temp_.copy()
                        elif temp_ - current_yaw - np.pi > 0.0:
                            # print('2 {0}'.format( current_yaw - temp_))
                            temp_ = 2*np.pi + current_yaw
                            pin['X'][-1] = temp_.copy()
                        else:
                            temp_ = current_yaw


    def solve(self, ):
        self.isSolved = True
        pos_flag = False
        ang_flag = False

        # check yaw angle
        self.check_angle_data()

        # # get original QP
        QSet_pos, ASet_pos, BSet_pos, AeqSet_pos, BeqSet_pos = self.getQPset('pos')


        mapMat = self.coeff2endDerivatives(AeqSet_pos[0])
        QSet_pos, HSet_pos, ASet_pos, BSet_pos, dp_e = self.mapQP(QSet_pos, ASet_pos, BSet_pos, AeqSet_pos, BeqSet_pos)

        for dd in range(self.position_dim):
            print('solving {} th dimention of the position'.format(dd))
            if ASet_pos is not None:
                result = solve_qp(P=QSet_pos[dd], q=HSet_pos[dd], G=ASet_pos[dd], h=BSet_pos[dd], solver='cvxopt')
                dP_ = result.reshape(-1, 1)
                dF_ = BeqSet_pos[dd]
                Phat_ = solve(mapMat, np.concatenate((dF_, dP_)))
                pos_flag = True
            else:
                dP_ = dp_e.copy()
                dF_ = BeqSet_pos[dd]
                Phat_ = solve(mapMat, np.concatenate((dF_, dP_[dd].reshape(-1, 1))))
                pos_flag = True
            if pos_flag:
                print('position {}-th dimension success!'.format(dd))
                P_ = np.dot(self.scaleMatBigInv('pos'), Phat_)
                self.pos_polyCoeffSet[dd] = P_.reshape(-1, self.pos_order+1).T

        QSet_ang, _, _, AeqSet_ang, BeqSet_ang= self.getQPset('ang')

        for dd in range(self.dim- self.position_dim):
            result = solve_qp(P=QSet_ang[dd], q=np.zeros((QSet_ang[dd].shape[0])), A=AeqSet_ang[dd], b=BeqSet_ang[dd], solver='cvxopt')
            Phat_ = result
            if Phat_ is not None:
                ang_flag = True
                P_ = np.dot(self.scaleMatBigInv('ang'), Phat_)
                self.ang_polyCoeffSet[dd] = P_.reshape(-1, self.ang_order+1).T
            else:
                ang_flag = False

        print(self.pos_polyCoeffSet.shape)
        print(self.ang_polyCoeffSet.shape)

    def eval(self, t_, d_):
        val_pos = np.zeros((self.position_dim, t_.shape[0]))
        for dd in range(self.position_dim):
            for idx in range(t_.shape[0]):
                t_i = t_[idx]
                if t_i < self.Ts[0] or t_i > self.Ts[-1]:
                    print("WARNING: Eval of t: out of bound. Extrapolation\n")
                m, _ = self.findSegInteval(t_i)
                # dTm = self.Ts[m+1] - self.Ts[m]
                val_pos[dd, idx] = np.dot(self.tVec(t_i-self.Ts[m], d_, self.pos_order).T, self.pos_polyCoeffSet[dd, :, m])

        val_ang = np.zeros((self.dim-self.position_dim, t_.shape[0]))
        for dd in range(self.dim-self.position_dim):
            for idx in range(t_.shape[0]):
                t_i = t_[idx]
                if t_i < self.Ts[0] or t_i > self.Ts[-1]:
                    print("WARNING: Eval of t: out of bound. Extrapolation\n")
                m, _ = self.findSegInteval(t_i)
                # dTm = self.Ts[m+1] - self.Ts[m]
                val_ang[dd, idx] = np.dot(self.tVec(t_i-self.Ts[m], d_, self.ang_order).T, self.ang_polyCoeffSet[dd, :, m])
        return np.concatenate((val_pos, val_ang))
