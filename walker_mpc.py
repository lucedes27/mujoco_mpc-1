# Goal: Change the below code to work with the walker.xml model

import mujoco
import casadi
import numpy as np
import mediapy as media

# because of linearity purposes the derivative fcn needs to be defined first

class walker_transition_jac(casadi.Callback):
    def __init__(self, name, opts={}):
        casadi.Callback.__init__(self)
        self.construct(name, opts)

    # for some reason casadi gives the output of the original fcn call to the derivative call
    def get_n_in(self): return 3
    # return two jacs, one wrt s and the other wrt u
    def get_n_out(self): return 2

    # define shape of input arguments
    def get_sparsity_in(self, n_in):
        # update dimensions based on the Walker task model
        if n_in == 0: return casadi.Sparsity.dense(18)  # Walker has 18 state variables
        if n_in == 1: return casadi.Sparsity.dense(6)   # Walker has 6 control inputs
        if n_in == 2: return casadi.Sparsity.dense(18)

    def get_sparsity_out(self, n_out):
        # dimensions of jac matrix we are returning
        if n_out == 0: return casadi.Sparsity.dense(18, 18)
        if n_out == 1: return casadi.Sparsity.dense(18, 6)

    def init(self):
        # initialize the mujoco model for Walker
        xml = open("walker.xml", 'r').read()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

    def eval(self, arg):
        # extract args
        s = np.array(arg[0])
        u = np.array(arg[1]).flatten()
        fcn_eval = arg[2] # i dont think we need this one, or at least mujuco doesn't need

        qpos = np.array(s[:9]).flatten()
        qvel = np.array(s[9:]).flatten()

        # set the data to desired state and control
        self.data.qpos = qpos
        self.data.qvel = qvel
        self.data.ctrl = u

        output1 = np.ndarray((18, 18), dtype=np.float64)
        output2 = np.ndarray((18, 6), dtype=np.float64)

        # discard other outputs, don't care about sensor data
        mujoco.mjd_transitionFD(self.model, self.data, 0.001, 1, output1, output2, None, None)

        return [output1, output2]

# want to build state transition function with signature state x control -> updated state

class walker_transition(casadi.Callback):
    def __init__(self, name, opts={}):
        casadi.Callback.__init__(self)
        self.construct(name, opts)

    def get_n_in(self): return 2
    def get_n_out(self): return 1

    # define shape of input arguments
    def get_sparsity_in(self, n_in):
        # update dimensions based on the Walker task model
        if n_in == 0: return casadi.Sparsity.dense(18)
        if n_in == 1: return casadi.Sparsity.dense(6)

    def get_sparsity_out(self, n_out):
        return casadi.Sparsity.dense(18)

    def init(self):
        # initialize the mujoco model for Walker
        xml = open("walker.xml", 'r').read()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # set the renderer flags here for visualization purposes.
        self.visualization = False
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        # register derivative of function
        self.df = walker_transition_jac('df')

    def eval(self, arg):
        # extract args
        s = np.array(arg[0])
        u = np.array(arg[1]).flatten()

        # update dimensions based on the Walker task model
        qpos = np.array(s[:9]).flatten()
        qvel = np.array(s[9:]).flatten()

        # set the data to desired state and control
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
        # input shapes
        x = casadi.MX.sym(inames[0], 18)
        u = casadi.MX.sym(inames[1], 6)
        out = casadi.MX.sym(inames[2], 18)
        jacs = self.df(x, u, out)
        return casadi.Function(name, [x, u, out], jacs)


# define time horizon in secs
T = 1
TIME_STEP = 0.02
N = int(T / TIME_STEP)

state_dim = 18  # update state dimension for Walker
ctrl_dim = 6    # update control dimension for Walker

opti = casadi.Opti()

# declare variables
X = opti.variable(state_dim, N + 1)
U = opti.variable(ctrl_dim, N)
P = opti.parameter(state_dim)  # initial state

f = walker_transition('f')

opti.minimize(-X[0, -1])  # maximize the height of the torso above ground
opti.subject_to(X[:, 0] == P)  # initial state is fixed to parameter

for k in range(N):
    # subsequent states are governed by state transition
    opti.subject_to(X[:, k + 1] == f(X[:, k], U[:, k]))

for k in range(N):
    opti.subject_to(U[:, k] <= [1] * ctrl_dim)
    opti.subject_to([-1] * ctrl_dim <= U[:, k])

opts = {}
# necessary to avoid hessian computations that we cannot produce with mujoco setup
opts['ipopt.hessian_approximation'] = 'limited-memory'

# initial state for the Walker task
initial_state = np.array([0, 1.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

opti.solver('ipopt', opts)
opti.set_value(P, initial_state)
sol = opti.solve()
print(sol)

# test code

with open("walker.xml", 'r') as xml_file:
    xml = xml_file.read()

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

x = np.append(data.qpos, data.qvel)
print(x, len(x))
u = np.zeros(ctrl_dim)

out = f(x, u)

out = f(out, u)
print(out)