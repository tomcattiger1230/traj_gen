#!/usr/bin/env python
# coding=utf-8
'''
Author: Wei Luo
Date: 2020-08-21 18:02:38
LastEditors: Wei Luo
LastEditTime: 2020-08-31 13:53:51
Note: Note
'''
from .traj_gen_base import TrajGen
import numpy as np
from scipy.linalg import block_diag, solve
import mosek as mk
import mosek.fusion as mkfs
from qpsolvers import solve_qp
import sys
import casadi as ca
import time


class PolyTrajGen(TrajGen):
    def __init__(self, knots_, order_, algo_, dim_, maxContiOrder_):
        """ Initialize the class of the trajectory generator"""
        super().__init__(knots_, dim_)
        self.N = order_ # polynomial order
        self.algorithm = algo_

        self.M = knots_.shape[0] - 1 # segments which is knots - 1
        self.maxContiOrder = maxContiOrder_
        self.num_variables =  (self.N+1) * self.M
        self.inf = np.inf
        self.segState = np.zeros((self.M, 2)) # 0 dim -> how many fixed pins in this segment,
                                              # muss smaller than the polynomial order+1
                                              # more fixed pins (higher order) will be ignored.
                                              # 1 dim -> continuity degree. should be defined by
                                              # user (maxContiOrder_+1)
        self.polyCoeffSet = np.zeros((self.dim, self.N+1, self.M))

    ## math functions
    def scaleMat(self, delT):
        mat_ = np.diag([delT**i for i in range(self.N+1)])
        return mat_

    def scaleMatBigInv(self,):
        mat_ = None
        for m in range(self.M):
            matSet_ = self.scaleMat(1/(self.Ts[m+1]-self.Ts[m]))
            if mat_ is None:
                mat_ = matSet_.copy()
            else:
                mat_ = block_diag(mat_, matSet_)
        return mat_

    ## functional definition
    def setDerivativeObj(self, weights):
        if weights.shape[0] > self.N:
            print("Order of derivative objective > order of poly. Higher terms will be ignored.")
            self.weight_mask = weights[:self.N]
        else:
            self.weight_mask = weights

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
            if self.segState[m, 0] <= self.N+1:
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

    def IntDerSquard(self, d):
        """
        {x^(d)(t)}^2  = (tVec(t,d)'*Dp)'*(tVec(t,d)'*Dp)

        Args:
            d(int): order derivative

        Returns:
            mat_: matrix of the cost function
        """
        mat_ = np.zeros((self.N+1, self.N+1))
        if d > self.N:
            print("Order of derivative > poly order, return zeros-matrix \n")
        for i in range(d, self.N+1):
            for j in range(d, self.N+1):
                # if i+j-2*d+1 > 0:
                mat_[i,j] = self.nthCeoff(i, d) * self.nthCeoff(j, d) / (i+j-2*d+1)
        return mat_

    def findSegInteval(self, t_):
        idx_ = np.where(self.Ts<=t_)[0]
        if idx_.shape[0]>0:
            m_ = np.max(idx_)
            if m_ >= self.M:
                if t_ != self.Ts[-1]:
                    print('Eval of t : geq TM. eval target = last segment')
                m_ = self.M-1
        else:
            print('Eval of t : leq T0. eval target = 1st segment')
            m_ = 0
        tau_ = (t_-self.Ts[m_])/(self.Ts[m_+1]-self.Ts[m_])
        return m_, tau_

    def tVec(self, t_, d_):
        # time vector evaluated at time t with d-th order derivative.
        vec_ = np.zeros((self.N+1, 1))
        for i in range(d_, self.N+1):
            vec_[i] = self.nthCeoff(i, d_)*t_**(i-d_)
        return vec_

    def fixPinMatSet(self, pin):
        t_ = pin['t']
        X_ = pin['X']
        d_ = pin['d']
        m_, tau_ = self.findSegInteval(t_)
        idxStart_ = m_*(self.N+1)
        idxEnd_ = (m_+1)*(self.N+1)
        dTm_ = self.Ts[m_+1] - self.Ts[m_]
        aeqSet_ = np.zeros((self.dim, self.num_variables))
        beqSet_ = np.zeros((self.dim, 1))
        for dd in range(self.dim):
            aeqSet_[dd, idxStart_:idxEnd_] = self.tVec(tau_, d_).flatten()/dTm_**d_#
            beqSet_[dd] = X_[dd]
        return aeqSet_, beqSet_

    def contiMat(self, m_, dmax):
        """
        ensure in dmax derivative degree the curve should be continued.
        from 0 to dmax derivative
        Args:
            m_: index of the segment <= M-1
            dmax: max conti-degree
        """
        dmax_ = int(dmax)
        aeq_ = np.zeros((dmax_+1, self.num_variables))
        beq_ = np.zeros((dmax_+1, 1)) # different of the eq should be zero
        idxStart_ = m_*(self.N+1)
        idxEnd_ = (m_+2)*(self.N+1) # end of the next segment
        dTm1_ = self.Ts[m_+1] - self.Ts[m_]
        dTm2_ = self.Ts[m_+2] - self.Ts[m_+1]
        for d in range(dmax_+1):
            # the end of the first segment should be the same as the begin of the next segment at each derivative
            aeq_[d, idxStart_:idxEnd_] = np.concatenate((self.tVec(1, d)/dTm1_**d, - self.tVec(0, d)/dTm2_**d), axis=0).flatten() #

        return aeq_, beq_

    def loosePinMatSet(self, pin_):
        aSet_ = np.zeros((self.dim, 2, self.num_variables))
        bSet_ = np.zeros((self.dim, 2, 1))
        t_ = pin_['t']
        X_ = pin_['X']
        d_ = pin_['d']
        m_, tau_ = self.findSegInteval(t_)
        dTm_ = self.Ts[m_+1] - self.Ts[m_]
        idxStart_ = m_*(self.N+1)
        idxEnd_ = (m_+1)*(self.N+1)
        for dd in range(self.dim):
            aSet_[dd, :, idxStart_:idxEnd_] = np.array([self.tVec(tau_, d_)/dTm_**d_,-self.tVec(tau_, d_)/dTm_**d_]).reshape(2, -1) #
            bSet_[dd, :] = np.array([X_[dd, 1], -X_[dd, 0]]).reshape(2, -1)
        return aSet_, bSet_


    def getQPset(self,):
        # objective
        QSet = np.zeros((self.dim, self.num_variables, self.num_variables))
        for dd in range(self.dim):
            Q_ = np.zeros((self.num_variables, self.num_variables))

            for d in range(1, self.weight_mask.shape[0]+1):
                if self.weight_mask[d-1] > 0:
                    Qd_ = None
                    for m in range(self.M):
                        dT_ = self.Ts[m+1] - self.Ts[m]
                        Q_m_ = self.IntDerSquard(d)/dT_**(2*d-1)
                        if Qd_ is None:
                            Qd_ = Q_m_.copy()
                        else:
                            Qd_ = block_diag(Qd_, Q_m_)
                    Q_ = Q_ + self.weight_mask[d-1]*Qd_
            QSet[dd] = Q_

        # constraint
        AeqSet = None
        ASet = None
        BSet = None
        BeqSet = None

        for m in range(self.M): # segments
            ## fix pin
            if m in self.fixPinSet.keys():
                for pin in self.fixPinSet[m]:
                    aeqSet, beqSet = self.fixPinMatSet(pin)
                    if AeqSet is None:
                        AeqSet = aeqSet.reshape(self.dim, -1, self.num_variables)
                        BeqSet = beqSet.reshape(self.dim, -1, 1)
                    else:
                        AeqSet = np.concatenate((AeqSet, aeqSet.reshape(self.dim, -1, self.num_variables)), axis=1)
                        BeqSet = np.concatenate((BeqSet, beqSet.reshape(self.dim, -1, 1)), axis=1)

                ## continuity
                if m < self.M-1:
                    contiDof_ = min(self.maxContiOrder+1, self.N+1-self.segState[m, 0])
                    self.segState[m, 1] = contiDof_
                    if contiDof_ != self.maxContiOrder+1:
                        print('Connecting segment ({0},{1}) : lacks {2} dof  for imposed {3} th continuity'.format(m, m+1, self.maxContiOrder-contiDof_, self.maxContiOrder))
                    if contiDof_ >0:
                        aeq, beq = self.contiMat(m, contiDof_-1)
                        AeqSet = np.concatenate((AeqSet, aeq.reshape(1, -1, self.num_variables).repeat(self.dim, axis=0)), axis=1)
                        BeqSet = np.concatenate((BeqSet, beq.reshape(1, -1, 1).repeat(self.dim, axis=0)), axis=1)
            else:
                pass # not pin in this interval

            ## loose pin
            if m in self.loosePinSet.keys():
                for pin in self.loosePinSet[m]:
                    aSet, bSet = self.loosePinMatSet(pin)
                    if ASet is None:
                        ASet = aSet
                        BSet = bSet
                    else:
                        ASet = np.concatenate((ASet, aSet), axis=1)
                        BSet = np.concatenate((BSet, bSet), axis=1)


        return QSet, ASet, BSet, AeqSet, BeqSet

    def coeff2endDerivatives(self, Aeq_):
        assert Aeq_.shape[1] <= self.num_variables, 'Pin + continuity constraints are already full. No dof for optim.'
        mapMat_ = Aeq_.copy()
        for m in range(self.M):
            freePinOrder_ = np.setdiff1d(np.arange(self.N+1), self.fixPinOrder[m]) # free derivative (not defined by fixed pin)
            dof_ = self.N+1 - np.sum(self.segState[m])
            freeOrder = freePinOrder_[:int(dof_)]
            for order in freeOrder:
                virtualPin_ = {'t':self.Ts[m], 'X':np.zeros((self.dim, 1)), 'd':order}
                # print('virtual Pin {}'.format(virtualPin_))
                aeqSet_, _ = self.fixPinMatSet(virtualPin_)
                aeq_ = aeqSet_[0] # only one dim is taken.
                mapMat_ = np.concatenate((mapMat_, aeq_.reshape(-1, self.num_variables)), axis=0)
        return mapMat_

    def mapQP(self, QSet_, ASet_, BSet_, AeqSet_, BeqSet_):
        Afp_ = self.coeff2endDerivatives(AeqSet_[0]) # sicne all Aeq in each dim are the same
        AfpInv_ = np.linalg.inv(Afp_)
        Nf_ = int(AeqSet_[0].shape[0])
        Qtemp_ = np.dot(np.dot(AfpInv_.T, QSet_[0]), AfpInv_)
        # Qff_ = Qtemp_[:Nf_, :Nf_]
        Qfp_ = Qtemp_[:Nf_, Nf_:]
        Qpf_ = Qtemp_[Nf_:, :Nf_]
        Qpp_ = Qtemp_[Nf_:, Nf_:]
        QSet = np.zeros((self.dim, self.num_variables-Nf_, self.num_variables-Nf_))
        HSet = np.zeros((self.dim, self.num_variables-Nf_))
        # check ASet ?
        if ASet_ is not None:
            ASet = np.zeros((self.dim, ASet_.shape[1], self.num_variables-Nf_))
            BSet = BSet_.copy()
            dp_ = None
            for dd in range(self.dim):
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
            dp_ = np.zeros((self.dim, self.num_variables-Nf_))
            for dd in range(self.dim):
                df_ = BeqSet_[dd]
                dp_[dd] = np.dot(np.dot(-np.linalg.inv(Qpp_), Qfp_.T), df_).flatten()

        return QSet, HSet, ASet, BSet, dp_

    def qp_mk_solver(self, P, q=None, G=None, h=None, A=None, b=None, lb=None, ub=None):
        '''
        description:
            using MOSEK to solve a qp problem
        param {type}
        return {type}
        '''
        num_x = P.shape[0]
        num_c = 0

        bound_key_cons = []
        bound_low_cons = []
        bound_up_cons = []

        A_sum = None
        xx = None
        # print('solving using mosek')

        ## only for print optimizer states
        # def streamprinter(text):
        #     sys.stdout.write(text + '\n')
        #     sys.stdout.flush()
        # prepare data
        num_x = P.shape[0]
        if lb is None and ub is None:
            bound_low_x = [-self.inf]*num_x
            bound_up_x = [self.inf]*num_x
            bound_key_x = [mk.boundkey.fr]*num_x
        elif lb is None and ub is not None:
            bound_key_x = [mk.boundkey.up]*num_x
        elif lb is not None and ub is None:
            bound_key_x = [mk.boundkey.lo]*num_x
        else:
            bound_key_x = [mk.boundkey.ra]*num_x
        if G is not None:
            num_c += G.shape[0]
            bound_key_cons = bound_key_cons + [mk.boundkey.up]*G.shape[0]
            bound_low_cons = bound_low_cons + [-np.inf]*G.shape[0]
            bound_up_cons = bound_up_cons + h.flatten().tolist()
            if A_sum is None:
                A_sum = G
            else:
                A_sum = np.concatenate((A_sum, G))
        if A is not None:
            num_c += A.shape[0]
            bound_key_cons = bound_key_cons + [mk.boundkey.fx]*A.shape[0]
            bound_low_cons = bound_low_cons + b.flatten().tolist()
            bound_up_cons = bound_up_cons + b.flatten().tolist()
            if A_sum is None:
                A_sum = A
            else:
                A_sum = np.concatenate((A_sum, A))

        with mk.Env() as env:
            with env.Task(0, 0) as task:
                # task.set_Stream(mk.streamtype.log, streamprinter) # for print solver information
                task.appendvars(num_x)
                for i in range(num_x):
                    task.putvarbound(i, bound_key_x[i], bound_low_x[i], bound_up_x[i])
                if q is not None:
                    for i in range(num_x):
                        task.putcj(i, q[i]) # add q vector
                for i in range(num_x):
                    for j in range(num_x):
                        if j <= i: # only the lower triangle matrix needs to be defined
                            task.putqobjij(i, j, P[i, j])

                task.appendcons(num_c)
                for i in range(num_c):
                    task.putconbound(i, bound_key_cons[i], bound_low_cons[i], bound_up_cons[i])
                for i in range(num_c):
                    for j in range(num_x):
                        task.putaij(i, j, A_sum[i, j])
                        # task.putaij(i, j, A[i, j])
                task.putobjsense(mk.objsense.minimize)
                task.optimize()
                # task.solutionsummary(mk.streamtype.msg)
                prosta = task.getprosta(mk.soltype.itr)
                solsta = task.getsolsta(mk.soltype.itr)
                xx = [0.]*num_x
                task.getxx(mk.soltype.itr, xx)
                if solsta == mk.solsta.optimal:
                    print("Optimal solution found.")
                elif solsta == mk.solsta.dual_infeas_cer:
                    print("Primal or dual infeasibility.\n")
                elif solsta == mk.solsta.prim_infeas_cer:
                    print("Primal or dual infeasibility.\n")
                elif mk.solsta.unknown:
                    print("Unknown solution status")
                else:
                    print("Other solution status")

        return xx

    def qp_mk_fusion_solver(self, P, q=None, G=None, h=None, A=None, b=None, lb=None, ub=None):
        '''
        description:
        param {type}
        return {type}
        '''
        num_x = P.shape[0]
        num_c_eq = 0
        xx = None
        print('solving using mosek fusion')

        ## only for print optimizer states
        # def streamprinter(text):
        #     sys.stdout.write(text + '\n')
        #     sys.stdout.flush()
        # prepare data
        if G is not None:
            num_c_ieq = G.shape[0]

        if A is not None:
            num_c_eq = A.shape[0]

        with mkfs.Model('test') as M:
            x = M.variable('x', num_x, mkfs.Domain.unbounded())
            t_0 = M.variable('t0', 1, mkfs.Domain.unbounded())
            # F_chol, d, _ = ldl(P, lower=True)
            try:
                F_chol = np.linalg.cholesky(P)
            except:
                F_chol =np.linalg.cholesky(P+np.diag(np.ones(P.shape[0]))*1e-7)
            # result of the cholesky decomposition
            F_ = M.parameter('F', [num_x, num_x])
            # set up value of the mksf parameter
            F_.setValue(F_chol.T)
            quad_cost = mkfs.Expr.vstack(t_0, mkfs.Expr.mul(F_, x))
            M.constraint("lc", quad_cost, mkfs.Domain.inQCone())

            if A is not None:
                A_ = M.parameter('A', [num_c_eq, num_x])
                b_ = M.parameter('b', num_c_eq)
                A_.setValue(A)
                b_.setValue(b.flatten())
                M.constraint(mkfs.Expr.sub(mkfs.Expr.mul(A_, x), b_), mkfs.Domain.equalsTo(0.0))
            if G is not None:
                A_ieq_ = M.parameter('A_ieq', [num_c_ieq, num_x])
                b_ieq_ = M.parameter('b_ieq', num_c_ieq)
                A_ieq_.setValue(G)
                b_ieq_.setValue(h.flatten())
                M.constraint(mkfs.Expr.sub(mkfs.Expr.mul(A_ieq_, x), b_ieq_), mkfs.Domain.lessThan(0.0))

            M.objective('obj', mkfs.ObjectiveSense.Minimize, t_0)
            t1 = time.time()
            M.solve()
            sol_x = x.level()
            return sol_x

    def solve(self,):
        self.isSolved = True
        # prepare QP
        QSet, ASet, BSet, AeqSet, BeqSet = self.getQPset()

        if self.algorithm == 'end-derivative':# and ASet is not None:
            mapMat = self.coeff2endDerivatives(AeqSet[0])
            QSet, HSet, ASet, BSet, dp_e = self.mapQP(QSet, ASet, BSet, AeqSet, BeqSet)
        elif self.algorithm == 'poly-coeff': # or ASet is None:
            pass

        for dd in range(self.dim):
            print('soving {}th dimension ...'.format(dd))
            if self.algorithm == 'poly-coeff': # or ASet is None:
                try:
                    if ASet is not None:
                        t1 = time.time()
                        result = solve_qp(P=QSet[dd], q=np.zeros((QSet[dd].shape[0])), G=ASet[dd],
                            h=BSet[dd], A=AeqSet[dd], b=BeqSet[dd], solver='cvxopt')
                        print("solve qp time {}".format(time.time() - t1))
                        t2 = time.time()
                        result = self.qp_mk_solver(P=QSet[dd], G=ASet[dd], h=BSet[dd], A=AeqSet[dd], b=BeqSet[dd])
                        print("mosek using api {0}".format(time.time() - t2))
                        t3 = time.time()
                        result = self.qp_mk_fusion_solver(P=QSet[dd], G=ASet[dd], h=BSet[dd], A=AeqSet[dd], b=BeqSet[dd])
                        print("mosek using fusion {0}".format(time.time() - t3))
                        # print("result qpsolver {0} \n and result mosek {1}".format(result, result_test))
                    else:
                        if dd == 3:
                            print(BeqSet[dd].shape)
                        # print(QSet[dd].shape)
                        # t1 = time.time()
                        result = solve_qp(P=QSet[dd], q=np.zeros((QSet[dd].shape[0])), A=AeqSet[dd], b=BeqSet[dd], solver='cvxopt')
                        # print(time.time() - t1)
                        # t2 = time.time()
                        result = self.qp_mk_solver(P=QSet[dd], A=AeqSet[dd], b=BeqSet[dd])
                        result = self.qp_mk_fusion_solver(P=QSet[dd], A=AeqSet[dd], b=BeqSet[dd])
                        print(result)
                        # print("saving")
                        # np.savetxt("P_"+str(dd)+".csv", QSet[dd], delimiter=",")
                        # np.savetxt("A_"+str(dd)+".csv", AeqSet[dd], delimiter=",")
                        # np.savetxt("B_"+str(dd)+".csv", BeqSet[dd], delimiter=",")
                        # np.save("P_"+str(dd)+".npy", QSet[dd])
                        # np.save("A_"+str(dd)+".npy", AeqSet[dd])
                        # np.save("B_"+str(dd)+".npy", BeqSet[dd])
                        # print(time.time()-t2)
                    Phat_ = result
                    if Phat_ is not None:
                        flag_ = True
                    else:
                        flag_ = False
                except:
                    Phat_ = None
                    flag_ = False
            else: # using end-derivative method
                if ASet is not None:
                    # result = solve_qp(P=QSet[dd], q=HSet[dd], G=ASet[dd],
                        # h=BSet[dd], solver='cvxopt')
                    # dP_ = result.reshape(-1, 1)
                    t1 = time.time()
                    result = self.qp_mk_solver(P=QSet[dd], q=HSet[dd], G=ASet[dd], h=BSet[dd])
                    print(time.time()-t1)
                    dP_ = np.array(result).reshape(-1, 1)
                    dF_ = BeqSet[dd]
                    Phat_ = solve(mapMat, np.concatenate((dF_, dP_)))

                    flag_ = True
                else:
                    # without considering the optimization problem, get the result directly
                    dP_ = dp_e.copy()
                    dF_ = BeqSet[dd]
                    Phat_ = solve(mapMat, np.concatenate((dF_, dP_[dd].reshape(-1, 1))))
                    flag_ = True
                # ## ipopt version [this is an alternative choice to solve the opt problem, however it is slower than the sigle qp problem]
                # t3 = time.time()
                # x_sym = ca.SX.sym('x', QSet[0].shape[0])
                # opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}
                # obj = 0.5* ca.mtimes([x_sym.T, QSet[dd], x_sym]) + ca.mtimes([HSet[dd].reshape(1, -1), x_sym])
                # Ax_sym = ca.mtimes([ASet[dd], x_sym])
                # nlp_prob = {'f': obj, 'x': x_sym, 'g':Ax_sym}
                # solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)
                # try:
                #     result = solver(ubg=BSet[dd],)
                #     dP_ = result['x']
                #     # print(dP_)
                #     dF_ = BeqSet[dd]
                #     Phat_ = solve(mapMat, np.concatenate((dF_, dP_)))
                #     flag_ = True
                # except:
                #     dP_ = None
                #     flag_ = False
                # print(time.time() - t3)
                # except:
                #     dP_ = None
                #     flag_ = False
                # else:
                #     # without considering the optimization problem, get the result directly
                #     dP_ = dp_e.copy()
                #     dF_ = BeqSet[dd]
                #     Phat_ = solve(mapMat, np.concatenate((dF_, dP_[dd].reshape(-1, 1))))
                #     flag_ = True
            if flag_:
                print("success !")
                # print('phat shape: {}'.format(Phat_.shape))
                P_ = np.dot(self.scaleMatBigInv(), Phat_)
                self.polyCoeffSet[dd] = P_.reshape(-1, self.N+1).T
                # print('for dd {0}, Phat {1}, P_{2}, result {3}'.format(dd, Phat_, P_, self.polyCoeffSet[dd]))
        print("done")

    def eval(self, t_, d_):
        val_ = np.zeros((self.dim, t_.shape[0]))
        for dd in range(self.dim):
            for idx in range(t_.shape[0]):
                t_i = t_[idx]
                if t_i < self.Ts[0] or t_i > self.Ts[-1]:
                    print("WARNING: Eval of t: out of bound. Extrapolation\n")
                m, _ = self.findSegInteval(t_i)
                # dTm = self.Ts[m+1] - self.Ts[m]
                val_[dd, idx] = np.dot(self.tVec(t_i-self.Ts[m], d_).T, self.polyCoeffSet[dd, :, m])

        return val_