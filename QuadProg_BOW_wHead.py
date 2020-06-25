# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
#from scipy.linalg import logm, norm, sqrtm

#from pyquaternion import Quaternion
import quadprog
 

def getqp_H(J, Vd, Epsilon1, Epsilon2 ,Epsilon3,Epsilon4):
    # Epsilon1: alpha close to 1
    # Epsilon2: weighting on dual arm motion
    # Epsilon3: q close to previous
    n = 12

    H1 = np.dot(np.hstack((J,np.zeros((n,1)))).T,np.hstack((J,np.zeros((n,1)))))
    
    H2 = np.dot(np.hstack((np.zeros((n,17)),Vd)).T,np.hstack((np.zeros((n,17)),Vd)))

    H3 = -2*np.dot(np.hstack((J,np.zeros((n,1)))).T, np.hstack((np.zeros((n,17)),Vd)))
    
    #C = np.vstack(( np.zeros((15,18)),np.hstack((np.zeros((3,14)),np.identity(3),np.zeros((3,1)))) ) )  
    C = np.zeros((18,18))
    C[14,14]=1
    C[15,15]=1
    C[16,16]=1
    
    H4 = np.dot(np.hstack((np.zeros((1,17)),np.sqrt(Epsilon1).reshape([1,1]))).T,np.hstack((np.zeros((1,17)),np.sqrt(Epsilon1).reshape([1,1]))))

    H5 = Epsilon2*np.dot(C.T,C)

    H6 = Epsilon3*np.dot(C.T,C)
    
    C2 = np.zeros((18,18))
    C2[14,14]=1
    H7 = Epsilon4*np.dot(C2.T,C2)
    
    H = 2*(H1+H2+H3+H4+H5+H6+H7)

    return H.astype('double')


def getqp_f(Epsilon1,Epsilon3,Epsilon4,dq_pre,Ktheta_h):
    C = np.zeros((18,18))
    C[14,14]=1
    C[15,15]=1
    C[16,16]=1

    C2 = np.zeros((18,18))
    C2[14,14]=1
    
    f1 = -2*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Epsilon1]).reshape(18, 1)
    f2 = -2*Epsilon3*np.dot(np.dot(C.T,C),np.vstack((dq_pre,0)).reshape(18, 1))
    f3 = -2*Epsilon4*np.dot(C2,np.vstack((np.zeros([14,1]),Ktheta_h,0,0,0)).reshape(18, 1))
    f = f1+f2+f3
    
    return f.astype('double')


def inequality_bound(h,c,eta,epsilon,e):
    sigma = np.zeros((h.shape))
    h2 = h - eta
    sigma[np.array(h2 >= epsilon)] = -np.tan(c*np.pi/2)
    sigma[np.array(h2 >= 0) & np.array(h2 < epsilon)] = -np.tan(c*np.pi/2/epsilon*h2[np.array(h2 >= 0) & np.array(h2 < epsilon)])
    sigma[np.array(h >= 0) & np.array(h2 < 0)] = -e*h2[np.array(h >= 0) & np.array(h2 < 0)]/eta
    sigma[np.array(h < 0)] = e
    
    return sigma

#def collision_check(select):
#    if select == 'Left':
#        Closest_Pt_env = np.array([1000,1000,1000])
#        Closest_Pt = np.array([0,0,0])
#    elif select == 'Right':
#        Closest_Pt_env = np.array([1000,1000,1000])
#        Closest_Pt = np.array([0,0,0])        
#    else:
#        Closest_Pt_env = np.array([1000,1000,1000])
#        Closest_Pt = np.array([0,0,0])        
#        
#    return Closest_Pt_env , Closest_Pt


def QP_bow_head(JT,Vd,Epsilon1,Epsilon2,q,Epsilon3, Epsilon4, Ktheta_h, dq_pre, Closest_Pt_L , Closest_Pt_env_L, Closest_Pt_R , Closest_Pt_env_R):
    # JT is 12x17
    # Vd is 12 x1
    # q is 17x1
    
    # joint limits
    dt = 0.1
    n_q = 17
    upper_joint_limit = (np.array([97.5, 60, 175, 150, 175, 120, 175])-25)*np.pi/180
    lower_joint_limit = (np.array([-97.5,-123, -175, -2.5, -175, -90, -175])+25)*np.pi/180
    upper_limit = np.hstack((upper_joint_limit,upper_joint_limit,np.array([10000,10000,10000])))
    lower_limit = np.hstack((lower_joint_limit,lower_joint_limit,np.array([-10000,-10000,-10000])))

    # inequality constraints
    h = np.zeros((n_q*2+2, 1))
    sigma = np.zeros((n_q*2+2, 1))
    dhdq = np.vstack((np.hstack((np.eye(n_q), np.zeros((n_q, 1)))), np.hstack((-np.eye(n_q), np.zeros((n_q, 1)))), np.zeros((2, n_q+1))))

    
    # parameters for inequality constraints
    c = 0.3#0.5
    eta = 0.25#0.1
    epsilon_in = 0.25#0.15
    E = 0.00001#0.005
    dmin = 0.01


    Q = getqp_H(JT, Vd, Epsilon1, Epsilon2 ,Epsilon3, Epsilon4)
    # make sure Q is symmetric
    Q = 0.5*(Q + Q.T)+np.eye(18)*1e-10
    
    f = getqp_f(Epsilon1,Epsilon3,Epsilon4,dq_pre,Ktheta_h)
    f = f.reshape((n_q+1, ))

    bound = np.array([2.,2.,2.,2.,4.,4.,4.,2.,2.,2.,2.,4.,4.,4.,1.,0.5/dt,0.5/dt]) # joint speed limits and base speed limits
    bound[0:-3]=bound[0:-3]*2
    LB = np.vstack((-dt*bound.reshape(n_q, 1),0))
    UB = np.vstack((dt*bound.reshape(n_q, 1),1))
        
    # inequality constrains A and b
    h[0:n_q] = q - lower_limit.reshape(n_q, 1)
    h[n_q:2*n_q] = upper_limit.reshape(n_q, 1) - q


#    Closest_Pt_env_L , Closest_Pt_L = collision_check('Left')
#    Closest_Pt_env_R , Closest_Pt_R = collision_check('Right')
    dv_L = Closest_Pt_env_L - Closest_Pt_L
    dist_L = np.sqrt(dv_L[0]**2 + dv_L[1]**2 + dv_L[2]**2)            
    # derivative of dist w.r.t time
    der_L = dv_L/dist_L #np.array([dx*(dist)**(-1), dy*(dist)**(-1), dz*(dist)**(-1)])
    dv_R = Closest_Pt_env_R - Closest_Pt_R
    dist_R = np.sqrt(dv_R[0]**2 + dv_R[1]**2 + dv_R[2]**2)            
    # derivative of dist w.r.t time
    der_R = dv_R/dist_R #np.array([dx*(dist)**(-1), dy*(dist)**(-1), dz*(dist)**(-1)])       
    
    
    h[2*n_q] = dist_L - dmin
    h[2*n_q+1] = dist_R - dmin
    """ """ """ """
    #dhdq[12, 0:6] = np.dot(-der.reshape(1, 3), J_eef2[3:6,:])
    dhdq[2*n_q, 0:n_q] = np.dot(-der_L[None, :], JT[3:6,:])
    dhdq[2*n_q+1, 0:n_q] = np.dot(-der_R[None, :], JT[9:12,:])
    
    sigma[0:2*n_q] =inequality_bound(h[0:2*n_q], c, eta, epsilon_in, E)
    sigma[2*n_q:2*n_q+2] = inequality_bound(h[2*n_q:2*n_q+2], c, eta, epsilon_in, E)           
    
    A = dhdq
    b = sigma
    
    A = np.vstack((A, np.eye(n_q+1), -np.eye(n_q+1)))
    b = np.vstack((b, LB, -UB))
    b = b.reshape((4*n_q+4, ))

    # solve the quadprog problem

    dq_sln = quadprog.solve_qp(Q, -f, A.T, b)[0]

        
#    if len(dq_sln) < n:
#        obj.params['controls']['dq'] = np.zeros((6,1))
#        V_scaled = 0
#        print 'No Solution'
#        dq_sln = np.zeros((int(n),1))
#    else:
#        obj.params['controls']['dq'] = dq_sln[0: int(obj.params['defi']['n'])]
#        obj.params['controls']['dq'] = obj.params['controls']['dq'].reshape((6, 1))
#        #print dq_sln
##        V_scaled = dq_sln[-1]*V_desired
##        vr_scaled = dq_sln[-2]*vr.reshape(3,1)
#        
#        #print np.dot(np.linalg.pinv(J_eef),v)
    if any(np.isnan(dq_sln)):
        dq_sln = np.zeros([18,])     
        
    return dq_sln

                           
        
#if __name__ == '__main__':
#    q = np.array([0,0,0,0,np.pi/2,0]).reshape(6, 1)
#    v = np.array([0,0,0,0,0,0.1])
#    a=QP_abbirb6640(q,v)
#    print a
    
