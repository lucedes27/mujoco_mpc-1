# Goal: Change the below code to work with the walker.xml model

import mujoco
import casadi
import numpy as np
import mediapy as media

# because of linearity purposes the derivative fcn needs to be defined first

class hopper_transition_jac(casadi.Callback):
    def __init__(self, name, opts={}):
        casadi.Callback.__init__(self)
        self.construct(name, opts)

    # for some reason casadi gives the output of the original fcn call to the derivative call
    def get_n_in(self): return 3
    # return two jacs, one wrt s and the other wrt u
    def get_n_out(self): return 2

    # define shape of input arguments
    def get_sparsity_in(self, n_in):
        # this will depend on the exact mujoco model we are using
        if n_in == 0: return casadi.Sparsity.dense(12)
        if n_in == 1: return casadi.Sparsity.dense(3)
        if n_in == 2: return casadi.Sparsity.dense(12)

    def get_sparsity_out(self, n_out):
        # dimensions of jac matrix we are returning
        # see docs on mjd_transitionFD for more info on jac matrix dim
        if n_out == 0: return casadi.Sparsity.dense(12, 12)
        if n_out == 1: return casadi.Sparsity.dense(12, 3)


    def init(self):
        # want to init the mujoco model here
        xml = open("walker.xml", 'r').read()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

    def eval(self, arg):
        # extract args
        # in order to copy them into mujoco data, they need to be cast to a numpy array
        # note the flattening to change shape from (n, 1) to (n,)
        s = np.array(arg[0])
        u = np.array(arg[1]).flatten()
        fcn_eval = arg[2] # i dont think we need this one, or at least mujuco doesn't need

        # mujoco handles qpos and qvel separately, so we must separate these states
        # you will probably need to change these numbers
        qpos = np.array(s[:6]).flatten()
        qvel = np.array(s[6:]).flatten()

        # we set the data to desired state and control
        self.data.qpos = qpos
        self.data.qvel = qvel
        self.data.ctrl = u

        output1 = np.ndarray((12,12), dtype=np.float64)
        output2 = np.ndarray((12,3), dtype=np.float64)

        # we discard the other outputs, dont care about sensor data
        mujoco.mjd_transitionFD(self.model, self.data, 0.001, 1, output1, output2, None, None)

        return [output1, output2]

# want to build state transition function with signature state x control -> updated state

class hopper_transition(casadi.Callback):
    def __init__(self, name, opts={}):
        casadi.Callback.__init__(self)
        self.construct(name, opts)

    def get_n_in(self): return 2
    def get_n_out(self): return 1

    # define shape of input arguments
    def get_sparsity_in(self, n_in):
        # this will depend on the exact mujoco model we are using
        # for hopper the state vector is of length 12 and control is length 3
        # these numbers will have to be changed for different models
        # note that the state consists of the qpos and the qvel
        if n_in == 0: return casadi.Sparsity.dense(12)
        if n_in == 1: return casadi.Sparsity.dense(3)
        print(n_in)

    def get_sparsity_out(self, n_out):
        return casadi.Sparsity.dense(12)

    def init(self):
        # want to init the mujoco model here
        xml = open("hopper.xml", 'r').read()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # set the renderer flags here for visualization purposes.
        self.visualization = False
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        # register derivative of function
        self.df = hopper_transition_jac('df')

    def eval(self, arg):
        # extract args
        # in order to copy them into mujoco data, they need to be cast to a numpy array
        # note the flattening to change shape from (n, 1) to (n,)
        s = np.array(arg[0])
        u = np.array(arg[1]).flatten()

        # mujoco handles qpos and qvel separately, so we must separate these states
        # you will probably need to change these numbers
        qpos = np.array(s[:6]).flatten()
        qvel = np.array(s[6:]).flatten()

        # some models will have a qact for activation. if your model has this lmk because
        # im not sure what to do about that one tbh

        # we set the data to desired state and control
        self.data.qpos = qpos
        self.data.qvel = qvel
        self.data.ctrl = u

        mujoco.mj_step(self.model, self.data)

        # visualization for debug purposes
        if self.visualization:
            with mujoco.Renderer(self.model) as renderer:
                renderer.update_scene(self.data, scene_option=self.scene_option)
                media.show_image(renderer.render())

        output = np.append(self.data.qpos, self.data.qvel)
        return [output]

    def has_jacobian(self, *args) -> bool:
        return True

    # return a casadi function handle to the jacobian
    def get_jacobian(self, name, inames, onames, opts={}):
        print(name, inames, onames)
        # input shapes
        x = casadi.MX.sym(inames[0], 12)
        u = casadi.MX.sym(inames[1], 3)
        out = casadi.MX.sym(inames[2], 12)
        jacs = self.df(x,u, out)
        return casadi.Function(name, [x, u, out], jacs)


# define time horizon in secs
T = 1
TIME_STEP = 0.02 # default mujoco stepping seems like
N = int(T/TIME_STEP)

state_dim = 12
ctrl_dim = 3

opti = casadi.Opti()

# declare variables
X = opti.variable(state_dim, N+1)
U = opti.variable(ctrl_dim, N)
P = opti.parameter(state_dim) # inital state

f = hopper_transition('f')

opti.minimize(-X[0, -1]) # maximize the height of the torso above ground
opti.subject_to(X[:, 0] == P) # initial state is fixed to parameter

for k in range(N):
    # subsequent states are governed by state transition
    opti.subject_to(X[:, k+1] == f(X[:, k], U[:, k]))

for k in range(N):
    opti.subject_to(U[:, k] <= [1,1,1])
    opti.subject_to([-1,-1,-1] <= U[:, k])

opts = {}
# neccesary to avoid hessian computations that we cannot produce with mujoco setup
opts['ipopt.hessian_approximation'] = 'limited-memory'

# the initial state the hopper is in
initial_state = np.array([0,   1.25, 0,  0,   0,   0,   0,   0,   0,   0,   0,   0,  ])

opti.solver('ipopt', opts)
opti.set_value(P, initial_state)
sol=opti.solve()
print(sol)

# the initial state the hopper is in
initial_state = np.array([0,   1.25, 0,  0,   0,   0,   0,   0,   0,   0,   0,   0,  ])
f = hopper_transition('f')

# Test
with open("hopper.xml", 'r') as xml_file:
    xml = xml_file.read()

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

x = np.append(data.qpos, data.qvel)
print(x, len(x))
u = np.array([0,0,0])

out = f(x, u)
print(out)