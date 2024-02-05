import numpy as np
import matplotlib.pyplot as plt
import PyFoam
from numpy import linspace
import shutil
import json
from os import path
from scipy.special import factorial
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.Execution.AnalyzedRunner import AnalyzedRunner
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.event import DEFAULT_EVENTS, Events
from bayes_opt.logger import JSONLogger
from datetime import datetime

from PyFoam.LogAnalysis.SimpleLineAnalyzer import GeneralSimpleLineAnalyzer
from matplotlib import gridspec

class newJSONLogger(JSONLogger):
    def __init__(self, path):
        self._path = None
        super(JSONLogger, self).__init__()
        self._path = path if path[-5:] == ".json" else path + ".json"


class CompactAnalyzer(BoundingLogAnalyzer):
    def __init__(self):
        BoundingLogAnalyzer.__init__(self)
        self.addAnalyzer(
            "concentration",
            GeneralSimpleLineAnalyzer(
                r"averageConcentration", r"^[ ]*areaAverage\(outlet\) of s = (.+)$"
            ),
        )

def run_cfd(a):
    identifier= datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print('Starting to copy case')
    case= identifier
    print(case)
    source_dir= "serpentine"
    destin_dir= identifier
    shutil.copytree(source_dir,destin_dir)
    velBC= ParsedParameterFile(path.join(case,"0","U"))
    velBC["boundaryField"]["inlet"]["variables"][1] = '"amp= %.5f;"' % a
    velBC.writeFile()

    decomposer = UtilityRunner(
        argv=["decomposePar", "-case", case],
        logname="decomposePar",
    )
    decomposer.start()

    run_command=f"mpirun -np 16 pimpleFoam -parallel"

    run = AnalyzedRunner(
        CompactAnalyzer(),
        argv=[run_command, "-case", case],
        logname="Solution",
    )
    # running CFD
    run.start()
    # post processing concentrations
    times = run.getAnalyzer("concentration").lines.getTimes()
    values = run.getAnalyzer("concentration").lines.getValues("averageConcentration_0")
    time = np.array(times)  # list of times
    value = np.array(values)  # list of concentrations
    print(time,value)
    return time,value

def rtd_convert(time,value):
    peaks, _ = find_peaks(value, prominence=0.004)
    #peaks= peaks.astype(int)
    times_peaks = time[peaks]
    values_peaks = value[peaks]
    plt.plot(times_peaks,values_peaks)

    dt = np.diff(times_peaks)[0]
    et = values_peaks / (sum(values_peaks * dt))
    tau = (sum(times_peaks * values_peaks * dt)) / sum(values_peaks * dt)
    etheta= tau*et
    theta = times_peaks / tau

    return(theta,etheta)

def calc_etheta(N: float, theta: float) -> float:
    z = factorial(N - 1)
    xy = (N * ((N * theta) ** (N - 1))) * (np.exp(-N * theta))
    etheta_calc = xy / z
    return etheta_calc


def loss(N: list, theta: list, etheta: list) -> float:
    et = []
    for i in range(len(etheta)):
        et.append(calc_etheta(N, theta[i]))
    error_sq = (max(etheta) - max(et)) ** 2
    return error_sq

def calculate_N(theta, etheta): 
    s = 10000
    n0_list = np.logspace(np.log(1), np.log(50), s)

    best = np.Inf
    for n0 in n0_list:
        l = loss(n0, theta, etheta)
        if l < best:
            best = l
            N = n0

    #plt.scatter(theta, etheta, c="k", alpha=0.4, label="CFD")
    etheta_calc = []
    for t in theta:
        etheta_calc.append(calc_etheta(N, t))
    plt.plot(theta, etheta_calc, c="k", ls="dashed", label="Dimensionless")

    print('N value', N)
    return N

####################

# for _ in range(5):
#     next_point=optimizer.suggest(utility)
#     target= black_box_function(**next_point)
#     optimizer.register(params=next_point, target=target)

#     print(target, next_point)
# print(optimizer.max)

def eval_cfd(a):
    time,value= run_cfd(a)
    theta, etheta= rtd_convert(time,value)
    N= calculate_N(theta,etheta)
    return N

logger = newJSONLogger(path="logs.json")
utility_f= UtilityFunction(kind='ucb',kappa= 5, xi=0.0)
optimizer= BayesianOptimization(
    f= eval_cfd,
    pbounds={
        "a": (0.001,0.008), 
    },
    verbose= 2,
)

def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x,iteration):
    fig = plt.figure(figsize=(7, 4))
    plt.subplots_adjust(left=0.1,right=0.95,top=0.95)
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    axis.set_xticks([],[])
    
    x_obs = np.array([[res["params"]["a"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)

    axis.scatter(x_obs.flatten(), y_obs, marker='+',s=80,lw=1, label=u'Observations', color='k')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.1, fc='k', ec='None',label='95% confidence interval')
    
    axis.set_xlim((0.001,0.008))
    axis.set_ylim((None, None))
    axis.set_ylabel('N')
    
    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='k')
    acq.scatter(x[np.argmax(utility)], np.max(utility), marker='+', s=80, c='k',lw=1,
             label=u'Next Best Guess')
    acq.set_xlim((0.001, 0.008))
    acq.set_ylabel('UCB')
    acq.set_xlabel('a')
    
    axis.legend(frameon=False)
    acq.legend(frameon=False)
    fig.savefig(str(iteration)+'.png')
    return 

# Opening JSON file
try:
    logs = []
    with open("logs.json") as f:
        print(f)
        for line in f:
            print(line)
            logs.append(json.loads(line))
    print(logs)
    for log in logs:
        optimizer.register(params=log["params"], target=log["target"])
        print(optimizer)
except FileNotFoundError:
    print('FILE NOT FOUND')
    pass

x = np.linspace(0.001, 0.008, 1000).reshape(-1, 1)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

iteration= 0
n_init=2
init_points= np.linspace(0.001,0.008,n_init).reshape(-1,1)
keys=['a']
print('keys', keys)

for p in init_points:
    p_dict= {}
    for i in range(len(keys)):
        p_dict[keys[i]]= p[i]
    target= eval_cfd(**p_dict)
    optimizer.register(params=p_dict,target= target)
    iteration+=1
    plot_gp(optimizer,x,iteration)

max_iterations=5

while True:
    utility_p = utility_f.utility(x, optimizer._gp, 0)
    next_point = {}
    next_point['a'] = x[np.argmax(utility_p)]
    target = eval_cfd(**next_point)
    iteration += 1 
    optimizer.register(params=next_point, target=target)
    plot_gp(optimizer, x, iteration)

    # Check if the termination condition is met
    if iteration >= max_iterations:
        break  

# for i in param_space:
#     source_dir= "serpentine"
#     destin_dir= "serpentine-amp%.3f" %i
#     case= "serpentine-amp%.3f" %i
#     time,value= run_cfd(case)
#     theta, etheta= rtd_convert(time,value)
#     plt.scatter(theta, etheta, c="k", alpha=0.4, label="CFD")
#     N= calculate_N(theta,etheta)
#     print('N value', N)
#     #plt.plot(time,value)
    




    

