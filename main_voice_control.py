from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import sys, os, traceback

if sys.platform == 'win32' or sys.platform == 'win64':
    os.environ['SDL_VIDEO_CENTERED'] = '1'
from math import *
pygame.display.init()
pygame.font.init()
import numpy as np
import timeit
import scipy.io as sio
from QuadProg_BOW import QP_bow
from baxter_info import *

screen_size = [800,600]
multisample = 16
#icon = pygame.Surface((1,1)); icon.set_alpha(0); pygame.display.set_icon(icon)
pygame.display.set_caption("Cloth Demo")
if multisample:
    pygame.display.gl_set_attribute(GL_MULTISAMPLEBUFFERS,1)
    pygame.display.gl_set_attribute(GL_MULTISAMPLESAMPLES,multisample)
pygame.display.set_mode(screen_size,OPENGL|DOUBLEBUF)

import thread
import speech_recognition as sr

minimal_create_interface="""
service experimental.minimal_create

object create_obj
    function void run()
    property int32 read
    
end object
"""
def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.

    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    recognizer.energy_threshold = 150
    with microphone as source:
        # recognizer.adjust_for_ambient_noise(source)
        
        audio = recognizer.listen(source)
        print (recognizer.energy_threshold)

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response

#glEnable(GL_BLEND)
#glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

#glEnable(GL_TEXTURE_2D)
#glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_MODULATE)
#glTexEnvi(GL_POINT_SPRITE,GL_COORD_REPLACE,GL_TRUE)

glHint(GL_PERSPECTIVE_CORRECTION_HINT,GL_NICEST)
glEnable(GL_DEPTH_TEST)

glPointSize(4)

def subtract(vec1,vec2):
    return [vec1[i]-vec2[i] for i in [0,1,2]]
def get_length(vec):
    return sum([vec[i]*vec[i] for i in [0,1,2]])**0.5

cloth_offset = [ 0.9213,-0.4592, 1.2194] # =p_OR0
cloth_scale = 0.9186
R_K2B = np.array([[0.0,0.0,1.0],[1.0,0.0,0.0],[0.0,1.0,0.0]])
R_K2B_inv = np.array([[0., 1., 0.],[0., 0., 1.],[1., 0., 0.]])

Table_O = np.dot(R_K2B_inv,(np.array([2,1,0])-cloth_offset))/cloth_scale+np.array([0.,1.0,0.]) 
Table_dim = np.dot(R_K2B_inv,(np.array([0.5,0.5,0.79])))/cloth_scale#np.array([0.5,0.79,0.5])/cloth_scale
Table_lower_bound = [Table_O[0]-Table_dim[0],Table_O[1],Table_O[2]-Table_dim[2]]
Table_upper_bound = [Table_O[0]+Table_dim[0],Table_O[1]+Table_dim[1],Table_O[2]+Table_dim[2]]


class Particle(object):
    def __init__(self,pos):
        self.pos = pos
        self.last_pos = list(self.pos)
        self.accel = [0.0,0.0,0.0]
        self.force = [0.0,0.0,0.0]
        self.vel = [0.0,0.0,0.0]
        self.last_vel = [0.0,0.0,0.0]
        self.streched = False
        self.constrained = False
        
    def move(self,dt):
        #Don't move constrained particles
      
        
        if self.constrained: return
        #Move
        
        for i in [0,1,2]:
            #Verlet
            temp = 2*self.pos[i] - self.last_pos[i] + self.accel[i]*dt*dt
            self.last_pos[i] = self.pos[i]
            self.pos[i] = temp
            
            
        if self.pos[0]> Table_lower_bound[0] and self.pos[0] < Table_upper_bound[0]:
            if self.pos[2]> Table_lower_bound[2] and self.pos[2] < Table_upper_bound[2]:
                if self.pos[1] < Table_upper_bound[1]:
                    self.pos[1] = Table_upper_bound[1]
#        else: 
#            for i in [0,1,2]:
#                self.pos[i] = self.last_pos[i]
#            if self.pos[i] < 0.0: self.pos[i] = 0.0
#            elif self.pos[i] > 1.0: self.pos[i] = 1.0
    def draw(self):
        glVertex3fv(self.pos)
        
class Edge(object):
    def __init__(self, p1,p2, tolerance=0.1):
        self.p1 = p1
        self.p2 = p2

        self.tolerance = tolerance
        
        self.rest_length = get_length(subtract(self.p2.pos,self.p1.pos))
        self.lower_length = self.rest_length*(1.0-self.tolerance)
        self.upper_length = self.rest_length*(1.0+self.tolerance)
        
    def constrain(self):
        vec = [self.p2.pos[i]-self.p1.pos[i] for i in [0,1,2]]
        length = get_length(vec)

        if   length > self.upper_length:
            target_length = self.upper_length
            strength = 1
        elif length < self.rest_length:
            target_length = self.lower_length
            strength = 0
        else: 
            target_length = self.rest_length
            strength = (length - self.rest_length) / ( 10*self.rest_length)

        
#        print ('Test:', self.rest_length)
        movement_for_each = strength * (length - target_length) / 0.2#0.5#2.0
        if movement_for_each>0.1:
            movement_for_each = 0.1
            self.p1.streched = True
            self.p2.streched = True
            
        K_d = 0.04
        for i in [0,1,2]:
            
            tmp = movement_for_each*vec[i] - K_d*(self.p1.last_vel[i]-self.p2.last_vel[i])
            self.p1.force[i] += tmp
            self.p2.force[i] -= tmp
            
            if not self.p1.constrained: self.p1.pos[i] += tmp
            if not self.p2.constrained: self.p2.pos[i] -= tmp

            
            
class ClothCPU(object):
    def __init__(self, res):
        self.res = res

        corners = [
            [-1,-1],        [ 1,-1],

            [-1, 1],        [ 1, 1]
        ]
        edges = [
                    [ 0,-1],
            [-1, 0],        [ 1, 0],
                    [ 0, 1]
        ]
        self.rels = edges + corners
        for rel in self.rels:
            length = sum([rel[i]*rel[i] for i in [0,1]])**0.5
            rel.append(length/float(self.res))

        self.reset()
    def reset(self):
        self.particles = []
        for z in range(self.res):
            row = []
            for x in range(self.res):
                row.append(Particle([
                    float(x)/float(self.res-1),
                    1.0,
                    float(z)/float(self.res-1)
                ]))
            self.particles.append(row)
        self.particles[         0][         0].constrained = True
        self.particles[self.res-1][         0].constrained = True
        self.particles[         0][self.res-1].constrained = True
        self.particles[self.res-1][self.res-1].constrained = True   
        
        self.particles[         0][         1].constrained = True
        self.particles[         0][self.res-2].constrained = True        
        #self.particles[         0][self.res-1].constrained = True

        self.edges = []
        for z1 in range(self.res):
            for x1 in range(self.res):
                p1 = self.particles[z1][x1]
                for rel in self.rels:
                    x2 = x1 + rel[0]
                    z2 = z1 + rel[1]
                    if x2 < 0 or x2 >= self.res: continue
                    if z2 < 0 or z2 >= self.res: continue
                    p2 = self.particles[z2][x2]

                    found = False
                    for edge in self.edges:
                        if edge.p1 == p2:
                            found = True
                            break
                    if found: continue

                    self.edges.append(Edge(p1,p2))

    def constrain(self, n):
        #Gravity
        for row in self.particles:
            for particle in row:
                particle.accel = [0.0, gravity, 0.0]
                particle.force = [0.0,0.0,0.0]
                for i in [0,1,2]:
                    particle.last_vel[i] = particle.pos[i]-particle.last_pos[i]  
                    
        for constraint_pass in range(n):
            for edge in self.edges:
                edge.constrain()
    def update(self,dt):
              
        #Move everything
        for row in self.particles:
            for particle in row:
                particle.move(dt)
    
    def update_mouse(self,dt):            

        self.particles[ point_n-1][         0].accel = [vx, vy, vz]#[mouse_rel[0]/10.0, -mouse_rel[1]/10.0, 2*mouse_rel[2]]
        self.particles[ point_n-1][ point_n-1].accel = [vx, vy, vz]#[mouse_rel[0]/10.0, -mouse_rel[1]/10.0, 2*mouse_rel[2]]  
        for i in [0,1,2]:
            self.particles[ point_n-1][         0].last_pos[i] = self.particles[ point_n-1][         0].pos[i]
            self.particles[ point_n-1][ point_n-1].last_pos[i] = self.particles[ point_n-1][ point_n-1].pos[i]
            self.particles[ point_n-1][         0].pos[i] = self.particles[ point_n-1][         0].pos[i]+self.particles[ point_n-1][         0].accel[i]*dt
            self.particles[ point_n-1][ point_n-1].pos[i] = self.particles[ point_n-1][ point_n-1].pos[i]+self.particles[ point_n-1][ point_n-1].accel[i]*dt                   
        #Move everything

        for row in self.particles:
            for particle in row:
                particle.move(dt)


    def draw(self):
        glBegin(GL_POINTS)
        for row in self.particles:
            for particle in row:
                particle.draw()
        glEnd()
    def draw_wireframe(self):
        glBegin(GL_LINES)
        for edge in self.edges:
            glVertex3fv(edge.p1.pos)
            glVertex3fv(edge.p2.pos)
        glEnd()
    def draw_mesh(self):
        for z in range(self.res-1):
            glBegin(GL_QUAD_STRIP)
            for x in range(self.res):
                glVertex3fv(self.particles[z  ][x].pos)
                glVertex3fv(self.particles[z+1][x].pos)
            glEnd()
    
gravity = -9.8/4#-9.80665
point_n = 11
cloth = ClothCPU(point_n)
target_fps = 30
dt = 1.0/float(target_fps)

camera_rot = [70,23]
camera_radius = 8
camera_center = [0.5,0.5,0.5]

def get_input():
    global camera_rot, camera_radius
    keys_pressed = pygame.key.get_pressed()
    mouse_buttons = pygame.mouse.get_pressed()
    mouse_rel = pygame.mouse.get_rel()
    scroll_bar = 0
    for event in pygame.event.get():
        if   event.type == QUIT: return False
        elif event.type == KEYDOWN:
            if   event.key == K_ESCAPE: return False
            elif event.key == K_r: cloth.reset()
        elif event.type == MOUSEBUTTONDOWN:
            if   event.button == 4: scroll_bar -= 1
            elif event.button == 5: scroll_bar += 1
            else: scroll_bar=0


    cloth.constrain(3)
    cloth.update_mouse(dt)            

    return True

def update(dt):
    cloth.constrain(3)
    cloth.update(dt)
    
    
def draw():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    
    glViewport(0,0,screen_size[0],screen_size[1])
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45,float(screen_size[0])/float(screen_size[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    camera_pos = [
        camera_center[0] + camera_radius*cos(radians(camera_rot[0]))*cos(radians(camera_rot[1])),
        camera_center[1] + camera_radius                            *sin(radians(camera_rot[1])),
        camera_center[2] + camera_radius*sin(radians(camera_rot[0]))*cos(radians(camera_rot[1]))
    ]
    gluLookAt(
        camera_pos[0],camera_pos[1],camera_pos[2],
        camera_center[0],camera_center[1],camera_center[2],
        0,1,0
    )

    cloth.draw()

    glColor3f(0,0.2,0)
    cloth.draw_wireframe()

    glColor3f(1,0,0)
    glBegin(GL_LINES)
    points = []
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                points.append([x,y,z])
    for p1 in points:
        for p2 in points:
            unequal = sum([int(p1[i]!=p2[i]) for i in [0,1,2]])
            if unequal == 1:
                glVertex3fv(p1)
                glVertex3fv(p2)
    glEnd()

    glColor3f(1,1,1)
    
    pygame.display.flip()


def collision_check(pp_L,pp_R):
    
    body_radius = 0.35
    Closest_Pt_l = np.array([1000,1000,1000])
    Closest_Pt_env_l = np.array([0,0,0])
    Closest_Pt_r = np.array([0,0,0]) 
    Closest_Pt_env_r = np.array([1000,1000,1000])
    
    d_min = 1000
        
    for l in pp_L.T:
        for r in pp_R.T:
            dist = np.linalg.norm(l-r)
            if dist < d_min:
                d_min = dist
                Closest_Pt_l = l
                Closest_Pt_env_l = r
                Closest_Pt_r = r
                Closest_Pt_env_r = l     
                
    if (np.linalg.norm(pp_L[0:2,-1])-body_radius) < d_min:
        Closest_Pt_l = pp_L[:,-1]
        scale = body_radius/np.linalg.norm(pp_L[0:2,-1])
        Closest_Pt_env_l = np.array([scale*pp_L[0,-1],scale*pp_L[1,-1],pp_L[2,-1]])       

    if (np.linalg.norm(pp_R[0:2,-1])-body_radius) < d_min:
        Closest_Pt_r = pp_R[:,-1]
        scale = body_radius/np.linalg.norm(pp_R[0:2,-1])
        Closest_Pt_env_r = np.array([scale*pp_R[0,-1],scale*pp_R[1,-1],pp_R[2,-1]])
        
    return Closest_Pt_l , Closest_Pt_env_l, Closest_Pt_r , Closest_Pt_env_r
   


process = -1
vx = 0
vy = 0
vz = 0

def run_sr(recognizer, microphone):
    global process
    global vx,vy,vz 
    
    WORDS = ["stop","forward","backward","left","right","higher","lower","freeze","Tire","laughed","fries","freezing","free"]
    # ["stop", "start", "pull tight", "relax", "lift", "lower", "grab there", "letgo", "freeze", "follow me", "go", "turn", "tug", "reach"]   
    step = 0.1
    
    while(True):
        
        
        print('Speak!')        
        command = recognize_speech_from_mic(recognizer, microphone)
        

        # if there was an error, stop the game
        if command["error"]:
            print("ERROR: {}".format(command["error"]))
        
        else:
            if not command["transcription"]:            
                if command["success"]:
                    print("I didn't catch that. What did you say?\n")
                
            else:        
                # show the user the transcription
                print("You said: {}".format(command["transcription"]))

                # determine if guess is correct and if any attempts remain
                if command["transcription"] in WORDS:
                    process = WORDS.index(command["transcription"])
                    # print (process, command["transcription"])
                    vx = 0
                    vy = 0
                    vz = 0
                    if process == 1:
                        vz += 3*step
                    elif process == 2:
                        vz -= 3*step
                    elif process == 3 or process == 9:
                        vx += step
                    elif process == 4:
                        vx -= step   
                    elif process == 5 or process == 8:
                        vy += step
                    elif process == 6:
                        vy -= step
                         
        print ("Process:", process)
        
        if process == 0:
            print ("Process ends!")
            break 
        
            
def main():
    t_all=[]
    clock = pygame.time.Clock()

    

    # create recognizer and mic instances
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration = 1)
    print (recognizer.energy_threshold)
    
    recognizer.energy_threshold = 150    
    try:
       thread.start_new_thread( run_sr,(recognizer, microphone))
    except:
       print "Error: unable to start thread"    
    
        

    Bdef = defineBaxter()    
    q0 = np.array([0.77350981,  0.26461169, -1.40589339,  2.07777698,  0.33594179, -0.49547579, -0.03298059]) 
    q0L = q0
    q0R = q0*[-1,1,-1,1,-1,1,-1]
    qL = q0L
    qR = q0R
    qB = np.array([0.0,0.0,0.0]) 
    
    tmp = np.array([[0.,0.,0.],[0.,0.,0.],[0.145,0.,0.]])
    tmp1 = np.array([0.0212,0.0212,0]).reshape([3,1])
    bow_joint_type_left = np.hstack([[1,1,0],Bdef.left.joint_type,[1]])
    bow_H_left = np.hstack([np.eye(3),Bdef.left.H,Bdef.left.H[:,1].reshape(3,1)])
    bow_P_left = np.hstack([tmp,Bdef.left.P,tmp1])
    bow_P_left[:,3] += np.array([-0.0400,       0,   0.2100])
    bow_P_left[:,10] += np.array([0.0742, -0.0035,   0.0000])
       
    tmp2 = np.array([0.0212,-0.0212,0]).reshape([3,1])
    bow_joint_type_right = np.hstack([[1,1,0],Bdef.right.joint_type,[1]])
    bow_H_right = np.hstack([np.eye(3),Bdef.right.H,Bdef.right.H[:,1].reshape(3,1)])
    bow_P_right = np.hstack([tmp,Bdef.right.P,tmp2])
    bow_P_right[:,3] += np.array([-0.0400,       0,   0.2100])
    bow_P_right[:,10] += np.array([-0.0035, -0.0742,   0.0000])

    pp_L,RR = fwdkin_alljoints(np.hstack([qB,qL,0.05]), bow_joint_type_left, bow_H_left, bow_P_left, 11)
    p_BTL0_bow = pp_L[:, -1]    
    pp_R,RR = fwdkin_alljoints(np.hstack([qB,qR,0.05]), bow_joint_type_right, bow_H_right, bow_P_right, 11)
    p_BTR0_bow = pp_R[:, -1]        
        
    K_com = 0.1
    GraspForce = np.array([0,0,0,0.15,-0.15,0,0,0,0,0.15,0.15,0])
    ez = np.array([0,0,1])
#    dt = 0.05
    dq_pre = np.zeros([17,1]) 
    Lambda = 0      # Lambda: weighting on dual arm motion
    Epsilon1 = 1    # Epsilon1: alpha close to 1
    Epsilon2 = 0.5   # Epsilon2: q close to previous         


    human_L = np.array(cloth.particles[ point_n-1][         0].pos)
    human_R = np.array(cloth.particles[ point_n-1][ point_n-1].pos)
    p_L = np.dot(R_K2B,human_L)
    p_R = np.dot(R_K2B,human_R)
    p_LR_prev = (p_R-p_L).reshape([3,])
    Z = np.vstack([p_L,p_R]).reshape([6,1])        
    Xk = np.vstack([Z, np.zeros([6,1])]).reshape([12,1])    
    Xk_prev = Xk

       
          
    while True:
    #for i in range(1000):
        tic = timeit.default_timer()
        if not get_input(): break

        pos = []
        for x in range(point_n):
            for y in range(point_n):
                    pos.append(np.array(cloth.particles[x][y].pos))     

        #==============================Control=================================
        VT = np.zeros([12,1])            
        q = np.hstack([qL,qR,np.array([0,0,0])]).reshape([17,1])
        
        JL = robotjacobian(Bdef.left.H, Bdef.left.P, Bdef.left.joint_type, qL)
        JR = robotjacobian(Bdef.right.H, Bdef.right.P, Bdef.right.joint_type, qR)
        
        pp_L,RR = fwdkin_alljoints(qL, Bdef.left.joint_type, Bdef.left.H, Bdef.left.P, 7)
        p_BTL = pp_L[:, -1]
        pp_R,RR = fwdkin_alljoints(qR, Bdef.right.joint_type, Bdef.right.H, Bdef.right.P, 7)
        p_BTR = pp_R[:, -1]
        JT = getJT(JL,JR,p_BTL,p_BTR)          
        
   
        ### Force Control
        wrL = np.dot(R_K2B,np.array(cloth.particles[         0][         0].force))
        wrR = np.dot(R_K2B,np.array(cloth.particles[         0][ point_n-1].force))
        h_LR = p_BTR-p_BTL
        h_LR_norm = h_LR/np.linalg.norm(h_LR)
        h_LRv = np.cross(h_LR_norm,ez)
#        print ('Force: ',np.dot(h_LR_norm,wrL),np.dot(h_LRv,wrL))
        if np.dot(h_LR_norm,wrL)>GraspForce[4] and np.dot(h_LRv,wrL)<GraspForce[3]:
            vL_force = -K_com*(np.dot(h_LR_norm,wrL)-GraspForce[4])*h_LR_norm-K_com*(np.dot(h_LRv,wrL)-GraspForce[3])*h_LRv
            VT[3:6] += vL_force.reshape([3,1])
        if np.dot(h_LR_norm,wrR)<GraspForce[10] and np.dot(h_LRv,wrR)<GraspForce[9]:
            vR_force = -K_com*(np.dot(h_LR_norm,wrR)-GraspForce[10])*h_LR_norm-K_com*(np.dot(h_LRv,wrR)-GraspForce[9])*h_LRv
            VT[9:12] += vR_force.reshape([3,1])
            
        Closest_Pt_L , Closest_Pt_env_L, Closest_Pt_R , Closest_Pt_env_R = collision_check(pp_L,pp_R)
        human_L = np.array(cloth.particles[ point_n-1][         0].pos)
        human_R = np.array(cloth.particles[ point_n-1][ point_n-1].pos)
        
        p_L = np.dot(R_K2B,human_L)
        p_R = np.dot(R_K2B,human_R)
        
        ## Kalman Iteration
        Z = np.vstack([p_L,p_R]).reshape([6,1])        
        Xk = np.vstack([Z, np.zeros([6,1])]).reshape([12,1])    
        Vk = Xk - Xk_prev
        Xk_prev = Xk
                               
        # For the next iteration                                                                 
        p_LR = (Xk[3:6,:]-Xk[0:3,:]).reshape([3,])
        p_LR = p_LR/np.linalg.norm(p_LR)
        vc = 1*(Vk[3:6,:]+Vk[0:3,:]).reshape([3,1])
        wc = 10*(np.cross(p_LR_prev,p_LR)*np.arccos(np.clip(np.dot(p_LR_prev,p_LR),-1,1))).reshape([3,1])                        
        
        Vc = np.vstack([wc,vc])
        
        AL = np.vstack([np.hstack([np.eye(3),np.zeros([3,3])]),np.hstack([-hat(-h_LR*0.5),np.eye(3)])])
        AR = np.vstack([np.hstack([np.eye(3),np.zeros([3,3])]),np.hstack([-hat(h_LR*0.5),np.eye(3)])])
        
        VT[0:6] += 10*np.dot(AL,Vc)
        VT[6:12] += 10*np.dot(AR,Vc)
        
        p_LR_prev = p_LR
        # print ('VT[0:6]: ',Vc)
#        print ('Vk: ',Vk)
#        print ('human_L: ',human_L)
#        print ('wc: ',wc)
        if not any(np.isnan(VT)):
#            if all(Vc) ==0:
#                dq_sln = np.zeros([18,])
#            else:
            dq_sln = QP_bow(JT,VT,Lambda,Epsilon1,q,Epsilon2,dq_pre, Closest_Pt_L , Closest_Pt_env_L, Closest_Pt_R , Closest_Pt_env_R)    
#            VT = np.zeros([12,1])
#            VT[9] = 0.2
#            VT[3] = 0.2
#            dq_sln = np.dot(np.linalg.pinv(JT),VT).reshape([17,])
#            print ('dq_sln: ',dq_sln)
            dq_sln=dq_sln*(abs(dq_sln)>0.01)
            dq_pre = dq_sln[0:17].reshape([17,1])
            dqL = dq_sln[0:7]
            dqR = dq_sln[7:14]
            qL += dqL*dt
            qR += dqR*dt

            dqB = np.array([dq_sln[15],dq_sln[16],dq_sln[14]])#dq_sln[14:17]
            qB += dqB*dt                      


            dV=np.dot(JT,dq_sln[0:17])


            pp_L,RR = fwdkin_alljoints(np.hstack([qB,qL,0.05]), bow_joint_type_left, bow_H_left, bow_P_left, 11)
            p_BTL_bow = pp_L[:, -1]    
            pp_R,RR = fwdkin_alljoints(np.hstack([qB,qR,0.05]), bow_joint_type_right, bow_H_right, bow_P_right, 11)
            p_BTR_bow = pp_R[:, -1] 
    
            cloth.particles[         0][         0].pos = np.dot(R_K2B_inv,(p_BTR_bow-cloth_offset))/cloth_scale+np.array([0.,1.0,0.]) 
            cloth.particles[         0][ point_n-1].pos = np.dot(R_K2B_inv,(p_BTL_bow-cloth_offset))/cloth_scale+np.array([0.,1.0,0.])         
            cloth.particles[         0][         1].pos = np.dot(R_K2B_inv,(p_BTR_bow-cloth_offset))/cloth_scale+np.array([0.1,1.0,0.]) 
            cloth.particles[         0][ point_n-2].pos = np.dot(R_K2B_inv,(p_BTL_bow-cloth_offset))/cloth_scale+np.array([-0.1,1.0,0.])
        pos = np.array(pos)          
        sio.savemat('pos.mat', {'pos':pos,'qL':qL,'qR':qR,'qB':qB})
        draw()
        clock.tick(target_fps)
        # t_all.append(timeit.default_timer()-tic)
        # print ('Time: ', np.mean(t_all),np.std(t_all))
        
    #print (np.mean(t_all),np.std(t_all))    
    pygame.quit()
    
if __name__ == '__main__':
    try:
        main()
    except:
        traceback.print_exc()
        pygame.quit()
        input()
