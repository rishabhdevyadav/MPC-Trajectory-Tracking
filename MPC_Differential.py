#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cvxpy
import scipy
import math
import numpy as np
import cubic_spline_planner

show_animation = True
NX = 3  # x = x, y, yaw
NU = 2  # a = [linear velocity,angular velocity ]
T = 5  # horizon length

# mpc parameters
R = np.diag([10000, .05])  # input cost matrix
Q = np.diag([10, 10,0.000])  # state cost matrix
Qf = np.diag([10, 10, 0.000]) # state final matrix

Rd = np.diag([10000, 0.01])

#Rd = np.diag([0.01, .1])  # input difference cost matrix

GOAL_DIS = 4  # goal distance
STOP_SPEED = 0.15   # stop speed
MAX_TIME = 200.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

N_IND_SEARCH = 10  # Search index number
DT = 0.25  # [s] time tick

TARGET_SPEED = 0.35   # [m/s] target speed
MAX_D = 5



class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw

def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi
    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi
    return angle


def get_linear_model_matrix(vref,phi):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[0, 2] = -vref*math.sin(phi)*DT
    A[1, 1] = 1.0
    A[1, 2] = vref*math.cos(phi)*DT
    A[2, 2] = 1.0

    B = np.zeros((NX, NU))
    B[0, 0] = DT * math.cos(phi)
    B[0, 1] = -0.5*DT*DT*math.sin(phi)*vref
    B[1, 0] = DT * math.sin(phi)
    B[1, 1] = 0.5*DT*DT*math.cos(phi)*vref
    B[2, 1] = DT

    return A, B


def update_state(state, v, omega):
    state.x = state.x + v * math.cos(state.yaw) * DT
    state.y = state.y + v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + omega * DT
    state.yaw = pi_2_pi(state.yaw)
    return state

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

def plot_arrow(x, y, yaw, length=0.2, width=0.5, fc="r", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def calc_nearest_index(state, cx, cy, cyaw, pind):
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    mind = min(d)
    ind = d.index(mind) + pind
    mind = math.sqrt(mind)
    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y
    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1
    return ind, mind


def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward
    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]
        move_direction = math.atan2(dy, dx)
        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0
        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed
    speed_profile[-1] = 0.0
    return speed_profile


def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]
        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]
        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]
    return yaw


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind,cur_vel):
    xref = np.zeros((NX, T + 1))
    vref = np.zeros((1, T + 1))
    ncourse = len(cx)
    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)
    if pind >= ind:
        ind = pind
    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = cyaw[ind]
    vref[0, 0] = sp[ind]
    travel = 0.0
    for i in range(T + 1):
        travel += abs(cur_vel) * DT
        dind = int(round(travel / dl))
        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = cyaw[ind + dind]
            vref[0, i] = sp[ind + dind]
            #print("if")
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = cyaw[ncourse - 1]
            vref[0, i] = sp[ncourse - 1]
            #print("else")
    return xref, ind, vref

def check_goal(state, goal, tind, nind,vi):
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)
    isgoal = (d <= GOAL_DIS)
    if abs(tind - nind) >= 5:
        isgoal = False
    isstop = (abs(vi) <= STOP_SPEED)
    if isgoal and isstop:
        return True
    return False

def get_forward_course(dl):
    #ax = [0.0, 0.5, 1.0, 2.0]
    #ay = [0.0, 0.0, 0.0, 0.0]
    #ax = [0.0, 1.0, 1.5, 2.0, 3.0]
    #ay = [0.0, 1.0, 2.0, 1.0, 0.0]   
    ax = [0.0, 1.0, 1.5, 3.0, 4.0]
    ay = [0.0, 1.0, 2.0, 1.0, 0.0] 
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=dl)
    return cx, cy, cyaw, ck


def iterative_linear_mpc_control(xref, x0, vref, ov, oomega, u0):
    if ov is None or oomega is None:
        ov = [0.0] * T
        oomega = [0.0] * T
    for i in range(MAX_ITER):
        xbar = predict_motion(x0, ov, oomega, xref)
        pov, poomega = ov[:], oomega[:]
        ov, omega, ox, oy, oyaw,ovr, ovl = linear_mpc_control(xref, xbar, x0, vref, u0)
        du = sum(abs(np.array(ov) - np.array(pov))) + sum(abs(np.array(oomega) - np.array(poomega)))
        if (du <= DU_TH):
            break   
    else:
        print("Iterative is max iter")
    return ov, omega, ox, oy, oyaw, ovr, ovl


def predict_motion(x0, ov, oomega, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0): #in-list #out-number,items
        xbar[i, 0] = x0[i]
    state = State(x=x0[0], y=x0[1], yaw=x0[2])
    for (i, v, omega) in zip(range(1, T + 1), ov, oomega):
        state = update_state(state, v, omega)
        xbar[0, i] = state.x
        xbar[1, i] = state.y 
        xbar[2, i] = state.yaw
    return xbar


def linear_mpc_control(xref, xbar, x0, vref, u0):    
    b,Radius = 0.2, 0.035     
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))
    vw = cvxpy.Variable((NU, T)) #to calculate vr and vl
 
    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t] ,R)
        if t != 0:
            cost += cvxpy.quad_form(x[:, t], Q)        
        A, B = get_linear_model_matrix(vref[0,t],xbar[2,t])  
        constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t]]        
        constraints += [vw[0,t] == ((1/Radius)*(u[0, t] + vref[0,t+1])) - ((b/(2*Radius))*(u[1, t]))]
        constraints += [vw[1,t] == ((1/Radius)*(u[0, t] + vref[0,t+1])) + ((b/(2*Radius))*(u[1, t]))]        
        if t < (T - 1):
            cost += cvxpy.quad_form((u[:, t + 1] - u[:, t]), Rd)
    constraints += [vw[:,:] <= 150/9.54 ]   # 140 rpm  60/(2*3.14)
    constraints += [vw[:,:] >= 0 /9.54]     
    cost += cvxpy.quad_form(x[:, T], Qf)
    constraints += [x[:, 0] == xref[:,0] - x0]    
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False,gp=False)
    #OSQP,CVXOPT, ECOS, scs

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        oyaw = get_nparray_from_matrix(x.value[2, :])
        ov = get_nparray_from_matrix(u.value[0, :])
        oomega = get_nparray_from_matrix(u.value[1, :])
        
        ovr = get_nparray_from_matrix(vw.value[0, :])
        ovl = get_nparray_from_matrix(vw.value[1, :])
        
        ox = ox + xref[0,:]
        oy = oy + xref[1,:]
        oyaw = oyaw + xref[2,:]
        ov = ov + vref[0,1:]
        oomega = -oomega

    else:
        print("Error: Cannot solve mpc..")
        ov, oomega, ox, oy, oyaw = None, None, None, None, None

    return ov, oomega, ox, oy, oyaw, ovr, ovl


def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
    goal = [cx[-1], cy[-1]]
    state = initial_state
    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [0.0]
    t = [0.0]
    omega = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)
    vi,vwheel,prev_v = 0,0,0
    prev_vwheel = 0
    alplp=0.4
    omegai = 0
    u0 = [0 ,0]
    oomega, ov, ox, oy, oyaw = None, None, None, None, None
    ovr, ovl = None, None
    cyaw = smooth_yaw(cyaw)
    

    while MAX_TIME >= time:
        xref, target_ind, vref = calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, target_ind,vi)       
        x0 = [state.x, state.y, state.yaw]  # current state        
        ov, oomega, ox, oy, oyaw, ovr, ovl = iterative_linear_mpc_control(xref, x0, vref, ov, oomega, u0)
        if oomega is not None:
            vi , omegai = ov[0], oomega[0]
            u0 = [ov[0], oomega[0]]
        vwheel = vi
        
        state = update_state(state, vwheel, omegai)
        time = time + DT
            
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(vi)
        t.append(time)
        omega.append(omegai)

        if check_goal(state, goal, target_ind, len(cx), vwheel):
            print("Goal")
            break       
        if target_ind % 1 == 0 and show_animation == True:
            plt.cla()
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[m/sec]:" + str(round(vwheel, 2))
                      + ",target index:" + str(target_ind))
            plt.pause(0.0003)
    return t, x, y, yaw, v, omega


if __name__ == '__main__':
    dl = 0.1 # course tick
    cx, cy, cyaw, ck = get_forward_course(dl)
    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
    
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0])
    t, x, y, yaw, v, omega = do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state)
    
    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()
    


