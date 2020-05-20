# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:47:49 2019

@author: d0v0b
"""
import numpy as np
from numpy.linalg import inv
import math
# from scipy.linalg import  sqrtm
from scipy.spatial.transform import Rotation

# from general_robotics_toolbox: https://github.com/rpiRobotics/general-robotics-toolbox
# generate cross product matrix for 3 x 1 vector

# expecting k as a 
def hat(k):
    khat = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return khat

# generate 3 x 3 rotation matrix for theta degrees about k
def rot(k, theta):
    I = np.identity(3)
    khat = hat(k)
    khat2 = khat.dot(khat)
    return I + math.sin(theta)*khat + (1.0 - math.cos(theta))*khat2

# jacobian of forward kinematics for a kinematic chain
def robotjacobian(H, P, joint_type, theta):
    hi = np.zeros(H.shape)
    pOi = np.zeros(P.shape)
    
    p = P[:,[0]]
    R = np.identity(3)
    
    pOi[:,[0]] = p
    
    for i in range(0, joint_type.size):
        if (joint_type[i] == 0 or joint_type[i] == 2):
            r = Rotation.from_rotvec(theta[i] * H[:,i].reshape([3,]))
            R = R.dot(r.as_dcm())
            #R = R.dot(rot(H[:,[i]],theta[i]))
        elif (joint_type[i] == 1 or joint_type[i] == 3):
            p = p + theta[i] * R.dot(H[:,[i]])
        p = p + R.dot(P[:,[i+1]])
        pOi[:,[i+1]] = p
        hi[:,[i]] = R.dot(H[:,[i]])
    
    pOT = pOi[:,[joint_type.size]]
    J = np.zeros([6,joint_type.size])
    i = 0
    j = 0
    while (i < joint_type.size):
        if (joint_type[i] == 0):
            J[0:3,[j]] = hi[:,[i]]
            J[3:6,[j]] = hat(hi[:,[i]]).dot(pOT - pOi[:,[i]])
        elif (joint_type[i] == 1):
            J[3:6,[j]] = hi[:,[i]]
        elif (joint_type[i] == 3):
            r = Rotation.from_rotvec(theta[i+2] * hi[:,[i+2]].reshape([3,]))
            J[3:6,[j]] = r.as_dcm().dot(hi[:,[i]])            
            
            #J[3:6,[j]] = rot(hi[:,[i+2]], theta[i+2]).dot(hi[:,[i]])
            J[0:3,[j+1]] = hi[:,[i+2]]
            J[3:6,[j+1]] = hat(hi[:,[i+2]]).dot(pOT - pOi[:,[i+2]])
            J = J[:,0:-1]
            i = i + 2
            j = j + 1
        
        i = i + 1
        j = j + 1
    return J

# find closest rotation matrix 
# A=A*inv(sqrt(A'*A))   
#def Closest_Rotation(R):
#    R_n = np.dot(R, inv(sqrtm(np.dot(R.T, R))))
#    
#    return R_n

def fwdkin_alljoints(q, ttype, H, P, n):
    R=np.eye(3)
    p=np.zeros((3,1))
    RR = np.zeros((3,3,n+1))
    pp = np.zeros((3,n+1))
    
    for i in range(n):
        h_i = H[0:3,i]
       
        if ttype[i] == 0:
        #rev
            pi = P[0:3,i].reshape(3, 1)
            p = p+np.dot(R,pi)
            
            r = Rotation.from_rotvec(q[i] * h_i)
            R = R.dot(r.as_dcm())                  
#            Ri = rot(h_i,q[i])
#            R = np.dot(R,Ri)
#            R = Closest_Rotation(R)
        elif ttype[i] == 1: 
        #pris
            pi = (P[:,i]+q[i]*h_i).reshape(3, 1)
            p = p+np.dot(R,pi)
        else: 
	    # default pris
	        pi = (P[:,i]+q[i]*h_i).reshape(3, 1)
	        p = p+np.dot(R,pi)
        
        pp[:,[i]] = p
        RR[:,:,i] = R
    
    # end effector T
    p=p+np.dot(R, P[0:3,n].reshape(3, 1))
    pp[:,[n]] = p
    RR[:,:,n] = R
    
    return pp, RR

def getJT(JL,JR,p_BTL,p_BTR):
    
    C1 = np.vstack((np.hstack((JL,np.zeros(JL.shape))),np.hstack((np.zeros(JR.shape),JR))))
    C2 = np.hstack((z0,-np.cross(p_BTL,z0),z0,-np.cross(p_BTR,z0))).reshape([12,1])
    C3 = np.hstack((zed,x0,zed,x0)).reshape([12,1])
    C4 = np.hstack((zed,y0,zed,y0)).reshape([12,1])
    JT = np.hstack([C1,C2,C3,C4])
    
    return JT

def getJT_OneArm(J,p_BT):
    
    C2 = np.hstack((z0,-np.cross(p_BT,z0))).reshape([6,1])
    C3 = np.hstack((zed,x0)).reshape([6,1])
    C4 = np.hstack((zed,y0)).reshape([6,1])
    JT = np.hstack([J,C2,C3,C4])
    
    return JT


x0 = np.array([1,0,0])
y0 = np.array([0,1,0])
z0 = np.array([0,0,1])
zed = np.array([0,0,0])

class defineArm():
    
    def __init__(self,H,P,joint_type,upper_joint_limit,lower_joint_limit):
        self.H = H
        self.P = P
        self.joint_type = joint_type
        self.upper_joint_limit = upper_joint_limit
        self.lower_joint_limit = lower_joint_limit

class defineBaxter():
    
    def __init__(self):
        # Joint limits
        upper_joint_limit = np.array([97.5, 60, 175, 150, 175, 120, 175])*np.pi/180
        lower_joint_limit = np.array([-97.5,-123, -175, -2.5, -175, -90, -175])*np.pi/180
        # Baxter origin
        baxter_origin_p = np.array([0.20,0,0.59]) #np.zeros(3)
        baxter_origin_R = np.identity(3)
        
        # Left Arm
        H = np.dot(baxter_origin_R,rot(z0,np.pi/4)).dot(np.column_stack((z0, y0, x0, y0, x0, y0, x0)))
        P = np.column_stack(([0.06375,0.25888,0.119217],
                                [0.069,0,0.27035],
                                zed, [0.36435,0,-0.069],
                                zed, [0.37429,0,-0.01],
                                zed, 0.229525*x0))
        P[:,1:] = np.dot(rot(z0,np.pi/4),P[:,1:])
        P = np.dot(baxter_origin_R,P)
        P[:,0] = baxter_origin_p + P[:,0]
        joint_type = np.zeros(7)
        self.left = defineArm(H,P,joint_type,upper_joint_limit,lower_joint_limit)
        
        # Right Arm
        H = np.dot(baxter_origin_R,rot(z0,-np.pi/4)).dot(np.column_stack((z0, y0, x0, y0, x0, y0, x0)))
        P = np.column_stack(([0.06375,-0.25888,0.119217],
                                [0.069,0,0.27035],
                                zed, [0.36435,0,-0.069],
                                zed, [0.37429,0,-0.01],
                                zed, 0.229525*x0))
        P[:,1:] = np.dot(rot(z0,-np.pi/4),P[:,1:])
        P = np.dot(baxter_origin_R,P)
        P[:,0] = baxter_origin_p + P[:,0]
        joint_type = np.zeros(7)
        self.right = defineArm(H,P,joint_type,upper_joint_limit,lower_joint_limit)
        
        
#if __name__ == '__main__':
#       baxter = defineBaxter()  
#       q0 = np.array([0.3804,0.0579,-1.6341,1.2805,0.1603,1.3948,0.0667])
#       q0L = q0
#       q0R = q0*[-1,1,-1,1,-1,1,-1]
#
#       q = np.hstack([q0L,q0R,np.array([0,0,0])]).reshape([17,1])
#
#       JL = robotjacobian(baxter.left.H, baxter.left.P, baxter.left.joint_type, q0L)
#       JR = robotjacobian(baxter.right.H, baxter.right.P, baxter.right.joint_type, q0R)
##       print JL
##       print JR
#       
#       pp,RR = fwdkin_alljoints(q0L, baxter.left.joint_type, baxter.left.H, baxter.left.P, 7)
##       print pp[:, -1], RR[:, :, -1]
#       p_BTL = pp[:, -1]
#       pp,RR = fwdkin_alljoints(q0R, baxter.right.joint_type, baxter.right.H, baxter.right.P, 7)
##       print pp[:, -1], RR[:, :, -1]
#       p_BTR = pp[:, -1]
#
#       JT = getJT(JL,JR,p_BTL,p_BTR)
##       print JT
#       
#       Vd_L = np.array([0,0,0,0.5,0.5,0])
#       Vd_R = np.array([0,0,0,0.5,0.5,0])
#       Vd = np.hstack([Vd_L,Vd_R]).reshape([12,1])
#       Lambda = 0.01
#       Epsilon1 = 1
#       Epsilon2 = 2
#       q_dot_pre = np.zeros([17,1])
#       dq_sln = QP_bow(JT,Vd,Lambda,Epsilon1,q,Epsilon2,q_dot_pre)
#       print dq_sln
#       print np.dot(np.linalg.pinv(JT),Vd)


        