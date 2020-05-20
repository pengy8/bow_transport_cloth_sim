% Creation of a pre-defined robot
% This is a test script to demonstrate how to take advantage of the
% pre-defined robots within the library.  In this example, we take a
% Rethink Robotics 'Baxter' robot and animate it following a prescribed
% path
%
% Be sure to use rigidbodyviz_setup() before running

clear variables; close all;

    gripper_aperture = 0.11;
    left_gripper_options.param = struct('aperture',gripper_aperture,  ...
                                                    'height',0.06);
    left_gripper_options.props = {};
    right_gripper_options.param = struct('aperture',gripper_aperture,  ...
                                                    'height',0.06);
    right_gripper_options.props = {};
    [bow_const, bow_structure] = defineBOW(...
                                'LeftGripper', left_gripper_options,  ...
                                'RightGripper', right_gripper_options);

                            
table = createTable(eye(3), [2;1;0], struct('surface_param', ...
                                                struct('width', 1, ...
                                                       'length', 1, ...
                                                       'height', 0.08), ...
                                                    'leg_param', ...
                                                struct('width', 0.05, ...
                                                       'length', 0.05, ...
                                                       'height', 0.7)));
% [bow_const, bow_structure] = defineBOW();

figure(1);
bow = createCombinedRobot(bow_const, bow_structure);
axis equal;
 axis([-2 6 -4 4 0 3]);
 view([75 10]);
grid on;
%% Simple Animation

% Time sequence
T = 4; dt = 0.1;
t = 0:dt:T;

q0 = [0.77350981,  0.26461169, -1.40589339,  2.07777698,  0.33594179, -0.49547579, -0.03298059]; 
qL0 = q0;
qR0 = q0.*[-1,1,-1,1,-1,1,-1];
qB0 = [0,0,0];
% Pre-allocate angle structure
% Expecting in order of left arm, right arm, and head
q = get_angle_structure(bow);
LEFT_ARM = strcmpi({q.name},'baxter_left_arm');
RIGHT_ARM = strcmpi({q.name},'baxter_right_arm');
RIDGEBACK = strcmpi({q.name},'ridgeback');

pause(1);
        

        
        
    q(3).state(:) = 0.05;
    q(4).state(:) = 0.05;
    q(6).state(:) = 0.05;
    q(7).state(:) = 0.05;        
%         h=1;
% for k=1:100
cloth_offset = [ 0.9213,-0.4592, 1.2194];
R = [0,0,1;1,0,0;0,1,0];
while(1)
    tic;
    
    left_arm_kin = get_kinematic_chain(bow_const, bow_structure, 'ridgeback', 'left_gripper_left_jaw');
    [ROL, pOL_O] = fwdkin(left_arm_kin,[q(RIDGEBACK).state q(LEFT_ARM).state q(3).state q(4).state]);
    right_arm_kin = get_kinematic_chain(bow_const, bow_structure, 'ridgeback', 'right_gripper_left_jaw');
    [ROR, pOR_O] = fwdkin(right_arm_kin,[q(RIDGEBACK).state q(RIGHT_ARM).state q(6).state q(7).state]);   
    
    try
        load('pos.mat');
        tmp = pos(:,2); pos(:,2)= pos(:,1); pos(:,1)=pos(:,3); pos(:,3)=tmp;

        pos=0.9186*(pos-[0,0,1])+cloth_offset;
       
    end    
    
    q(LEFT_ARM).state(:) = qL;%pi/4*sin(2*pi*t(k)); % s1 in left arm
    q(RIGHT_ARM).state(:) = qR;%-pi/4*sin(2*pi*t(k)); % s1 in right arm
%      q(8).state(1) = sin(2*pi*t(k)); % head pan

    
    q(RIDGEBACK).state(:) = qB;
    bow = updateRobot(q, bow);
    drawnow; 
    
     if exist('h')
        delete(h);
     end
    
        
    [t]=MyCrustOpen(pos);
    set(gcf, 'PaperSize', [8 5]);
    hold on;
    h=trisurf(t,pos(:,1),pos(:,2),pos(:,3),'facecolor','c','edgecolor','b');
    hold off;

     t1 = toc;
     while t1 < dt,   t1 = toc; end
end
