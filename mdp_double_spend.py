import mdptoolbox
import numpy as np
import networkx as nx
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import queue
from matplotlib.colors import LogNorm
import seaborn
seaborn.set(font_scale=2.3)
seaborn.set_style("whitegrid")
import sys

class State:
    def __init__(self, l_a, l_h, n):
        self.length_a = l_a
        self.length_h = l_h
        self.n = n

    def __hash__(self):
        return hash((self.length_a, self.length_h, self.n))

    def __eq__(self, other):
        try:
            return (self.length_a, self.length_h, self.n) == (other.length_a, other.length_h, other.n)
        except:
            return False

    def __ne__(self, other):
        return not(self == other)

    def __repr__(self):
        return "(%d, %d, %d)" % (self.length_a, self.length_h, self.n)

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def optimal_strategy(p, k, double_spend_value, m, max_blocks, cutoff, m_cost, block_reward, bribe):
    """
    p:                  attacker hashrate as fraction of total hashrate
    k:                  vitcim's security period. Min. k+1 blocks need to be found on the honest chain before the double spend can be executed
    double_spend_value: value the attacker gains from a successful double spend, 1 = block reward
    m:                  number of block funded upfront during an extend action. Default = 1
    max_blocks:         maximum number of total mined blocks considered
    gamma:              fraction of honest nodes that a block from the attacker reaches before a block from the honest network (when matching)
    cutoff:             maximum length of either chain (needed for finite MDP)
    m_cost:             cost of mining (per state transition)
    block_reward:       block reward (in BTC) incl. transaction fees. Default = 1
    bribe:              value of the bribe added on top of the block reward (measured as fraction of block reward)

    implicitly: 
        q = (1 - p) hashrate of honest network
    """
    states = {}
    states_inverted = {}
    q = 1.0-p
    
    # construct states
    states_counter = 0
    for l_a in range(cutoff + 1):
        for l_h in range(cutoff + 1):
            for n in range(cutoff + 1):
                
                state = State(l_a, l_h, n)
                states[states_counter] = state
                states_inverted[state] = states_counter
                states_counter += 1

    # exit state
    exit_idx = states_counter
    states_counter += 1
    adopt_idx = states_counter
    states_counter += 1

    # build transition matrices
    P_adopt    = np.zeros(shape=(states_counter, states_counter))
    P_extend    = np.zeros(shape=(states_counter, states_counter))
    P_wait     = np.zeros(shape=(states_counter, states_counter))
    P_exit     = np.zeros(shape=(states_counter, states_counter))
    
    # build reward matrices
    R_adopt    = np.zeros(shape=(states_counter, states_counter))
    R_extend    = np.zeros(shape=(states_counter, states_counter))
    R_wait     = np.zeros(shape=(states_counter, states_counter))
    R_exit     = np.zeros(shape=(states_counter, states_counter))

    # never leave exit state
    P_exit[exit_idx, exit_idx] = 1
    P_adopt[exit_idx, exit_idx] = 1
    P_extend[exit_idx, exit_idx] = 1
    P_wait[exit_idx, exit_idx] = 1

    # never leave the adopt state
    P_exit[adopt_idx, adopt_idx] = 1
    P_adopt[adopt_idx, adopt_idx] = 1
    P_extend[adopt_idx, adopt_idx] = 1
    P_wait[adopt_idx, adopt_idx] = 1
    
    #AZ - TODO  not sure what this does... seems it sets honest mining reward for exit state
    R_exit[exit_idx, exit_idx] = p - m_cost
    # AZ - added the same for the adopt state, as this is never left as well
    R_adopt[adopt_idx, adopt_idx] = p - m_cost

    for state_idx, state in states.items():
        l_a = state.length_a
        l_h = state.length_h
        n = state.n
        
        # exit
        if l_h > k and l_a > l_h:
            P_exit[state_idx, exit_idx] = 1
            R_exit[state_idx, exit_idx] = double_spend_value - n * bribe
        else:
            # AZ - TODO: Not sure why this is needed
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_exit[state_idx, state_idx] = 1
            R_exit[state_idx, state_idx] = -100


        # adopt
        P_adopt[state_idx, adopt_idx] = 1
        R_adopt[state_idx, adopt_idx] = -n * (block_reward + bribe)
        # AZ - TODO: not sure if we need the stochastic matrix fix from above 

        '''
        # Not needed since we never leave the adopt state
        # attacker mines next block
        P_adopt[state_idx, states_inverted[State(1, 0, n)]] = p
        R_adopt[state_idx, states_inverted[State(1, 0, n)]] = 0  
        # eclipsed node mines next block
        if lam != 0:
            P_adopt[state_idx, states_inverted[State(1, 0, 1, "irrelevant")]] = lam 
            R_adopt[state_idx, states_inverted[State(1, 0, 1, "irrelevant")]] = 0 - m_cost 
        # network mines next block
        P_adopt[state_idx, states_inverted[State(0, 1, 0, "relevant")]] = q*(1-stale)
        R_adopt[state_idx, states_inverted[State(0, 1, 0, "relevant")]] = 0 - m_cost 
        # network mines stale block
        P_adopt[state_idx, states_inverted[State(0, 0, 0, "irrelevant")]] = q*stale
        R_adopt[state_idx, states_inverted[State(0, 0, 0, "irrelevant")]] = 0 - m_cost
        '''

        # only allow adopt or exit after cutoff
        if l_a == cutoff or l_h == cutoff or n == cutoff:
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_extend[state_idx, state_idx] = 1
            R_extend[state_idx, state_idx] = -100
            P_wait[state_idx, state_idx] = 1
            R_wait[state_idx, state_idx] = -100
            continue
            
        # extend
        # attacker mines next block
        P_extend[state_idx, states_inverted[State(l_a + 1, l_h, n + m - 1)]] = p
        R_extend[state_idx, states_inverted[State(l_a + 1, l_h, n + m - 1)]] = 0 - m_cost 
        # network mines next block
        P_extend[state_idx, states_inverted[State(l_a, l_h + 1, n + m)]] = q
        R_extend[state_idx, states_inverted[State(l_a, l_h + 1, n + m)]] = 0 - m_cost

        # wait
        if(n > 0):
            # attacker mines next block
            P_wait[state_idx, states_inverted[State(l_a + 1, l_h, n - 1)]] = p
            R_wait[state_idx, states_inverted[State(l_a + 1, l_h, n - 1)]] = 0 - m_cost 
            # network mines next block
            P_wait[state_idx, states_inverted[State(l_a, l_h + 1, n)]] = q
            R_wait[state_idx, states_inverted[State(l_a, l_h + 1, n)]] = 0 - m_cost
        else:
            P_wait[state_idx, state_idx] = 1
            R_wait[state_idx, state_idx] = -100

    P = [P_wait, P_adopt, P_extend, P_exit]
    R = [R_wait, R_adopt, R_extend, R_exit]
    for i,p in enumerate(P):
        try:
            mdptoolbox.util.checkSquareStochastic(p)
        except:
            print("not stochastic:", i)
            #for l in p:
                #print l
    #mdp = mdptoolbox.mdp.FiniteHorizon(P, R, 0.999, max_blocks)
    #mdp = mdptoolbox.mdp.ValueIteration(P, R, 0.999)
    #mdp = mdptoolbox.mdp.QLearning(P, R, 0.999)
    mdp = mdptoolbox.mdp.PolicyIteration(P, R, 0.999)
    #mdp.setVerbose()
    mdp.run()
    return mdp, states

def state_graph(states, transitions, policy):
    policy_colors = ["blue", "red", "grey", "yellow", "green"]
    G = nx.DiGraph()
    q = queue.Queue()
    visited = [False]*len(states)
    visited[0] = True
    q.put(0)
    while not q.empty():
        state_idx = q.get()
        pol = policy[state_idx]
        G.add_node(states[state_idx], color=policy_colors[pol], style="filled")
        for i, p in enumerate(transitions[pol][state_idx]):
            if p > 0:
                if i == len(states):
                    G.add_edge(states[state_idx], "exit", label=p)
                else:
                    G.add_edge(states[state_idx], states[i], label=p)
                    if not visited[i]:
                        q.put(i)
                        visited[i] = True
    return G


def state_table(states, transitions, policy, cutoff):
    policy_letter = ["w", "a", "x", "e"]
    q = queue.Queue()
    table = [['*']*cutoff]*cutoff
    visited = [False]*len(states)
    visited[0] = True
    q.put(0)
    while not q.empty():
        state_idx = q.get()
        pol = policy[state_idx]
        state = states[state_idx]
        table[state.length_a][state.length_h] = policy_letter[pol]
        for i, p in enumerate(transitions[pol][state_idx]):
            if p > 0:
                if i == len(states):
                    pass
                else:
                    if not visited[i]:
                        q.put(i)
                        visited[i] = True
    return table

def markov_chain(states, transitions, policy):
    import pykov
    T = pykov.Chain()
    q = queue.Queue()
    visited = [False]*len(states)
    visited[0] = True
    q.put(0)
    start = pykov.Vector({states[0]:1})
    while not q.empty():
        state_idx = q.get()
        pol = policy[state_idx]
        for i, p in enumerate(transitions[pol][state_idx]):
            if p > 0:
                if i == len(states):
                    T[(states[state_idx], "exit")] = p
                    T[("exit", "exit")] = 1
                else:
                    T[(states[state_idx], states[i])] = p
                    if not visited[i]:
                        q.put(i)
                        visited[i] = True
    return T, start

def markov_matrix(transitions, policy):
    n = len(policy)
    P = np.zeros((n, n))
    q = queue.Queue()
    visited = [False]*n
    visited[0] = True
    q.put(0)
    while not q.empty():
        state_idx = q.get()
        pol = policy[state_idx]
        for i, p in enumerate(transitions[pol][state_idx]):
            if p > 0:
                P[state_idx, i] = p
                if not visited[i]:
                    q.put(i)
                    visited[i] = True
    return P

def exp_blocks_needed(p, k, double_spend_value, m=1, cutoff=20, m_cost=0, block_reward=1, bribe=0.1):
    mdp, states = optimal_strategy(p, k, double_spend_value, m, None, cutoff, m_cost, block_reward, bribe)
    P = markov_matrix(mdp.P, mdp.policy)
    l = len(mdp.policy)
    Q = P[0:l-1,0:l-1]
    I = np.eye(l-1)
    N = np.linalg.inv(I - Q)
    ones = np.ones((l-1,1))
    t = N.dot(ones)
    return t[0]


def hashrate_k_plot(cost, m, cutoff, block_reward, bribe):
    ps = np.arange(0.025, 0.5, 0.1)
    # ps = np.arange(0.025, 0.5, 0.025)

    # AZ - TODO: parametrize this correctly for actual run
    ks = np.arange(0, cutoff, 1)
    ds = np.zeros([len(ps), len(ks)])
    max_val = 1000000000
    eps = 0.1
    for p_idx, p in enumerate(ps):
        for k_idx, k in enumerate(ks):
            m_cost = cost*p
            double_spend_value = max_val/2
            if k_idx > 0 and ds[p_idx, k_idx-1] > max_val - eps:
                double_spend_value = max_val
            last_value = 0
            diff = max_val/2
            lower = 0
            upper = max_val
            while diff > eps:
                print(p, k, double_spend_value)
                mdp,states = optimal_strategy(p, k, double_spend_value, m, None, cutoff, m_cost, block_reward, bribe)
                
                '''
                G = state_graph(states, mdp.P, mdp.policy)
                diff = abs(last_value - double_spend_value)
                if G.has_node("exit"):
                    print("exit")
                    last_value = double_spend_value
                    upper = double_spend_value
                    double_spend_value -= (double_spend_value - lower)/2.0
                # AZ - TODO: not sure if this is needed
                elif G.has_node("adopt"):
                    print("adopt")
                    last_value = double_spend_value
                    upper = double_spend_value
                    double_spend_value -= (double_spend_value - lower)/2.0
                else:
                    last_value = double_spend_value
                    lower = double_spend_value
                    double_spend_value += (upper - double_spend_value)/2.0
                '''
            ds[p_idx, k_idx] = last_value

    np.save("hashrate_k_double_spend_co%dg%.2fs%.4fc%.2f.npy" % (cutoff, m, block_reward, bribe), ds)
    plt.pcolor(ps, ks, ds.T, norm=LogNorm(vmin=ds.min(), vmax=ds.max()))
    cbar = plt.colorbar()
    cbar.set_label("double spend value")
    plt.ylabel("k")
    plt.xlabel("p")
    plt.savefig("hashrate_k_double_spend_co%dg%.2fs%.4fc%.2f.png" %(cutoff, m, block_reward, bribe))
    plt.close()

# --------- START: OLD HELPER FUNCTIONS -----------
def print_table(table):
    l = len(table)
    print(r"\begin{tabular}{@{}c|"+ l*'c' +r"@{}}")
    print(r"\toprule")
    print('& ' + ' & '.join(str(x) for x in range(l)) + r'\\')
    for idx, line in enumerate(table):
        print(str(idx) + '& ' +  ' & '.join([''.join(x) for x in line]) + r'\\')
        if idx < l-1:
            print(r'\midrule')
    print(r"\bottomrule")
    print(r"\end{tabular}")

# --------- END: OLD HELPER FUNCTIONS     


def main():
    l = len(sys.argv)
    if l >=7:
        cost = float(sys.argv[1])
        block_reward = float(sys.argv[2])
        m =  float(sys.argv[3])
        cutoff = float(sys.argv[4])
        bribe = float(sys.argv[5])
        k = float(sys.argv[6])
    else:
        # TEST PARAMS
        print("RUNNING WITH TEST PARAMETERS...")
        cost = 0
        block_reward = 1
        m = 1
        cutoff = 6
        bribe = 0.1
        k = 1
        '''
        print("Not enough arguments")
        print("Usage: %s <cost> <block_reward> <m> <cutoff> <bribe>" %sys.argv[0])
        return
        '''
    #hashrate_k_plot(cost, m, cutoff, block_reward, bribe)

if __name__=="__main__":
    main()