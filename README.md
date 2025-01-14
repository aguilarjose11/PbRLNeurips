Code for paper "RA-PbRL: Provably Efficient Risk-Aware Preference-Based Reinforcement Learning"
==============================================================

Code Setup Documentation
-------------------------

### Libraries (>= Python 3.12.4)

For more information on the version specifics, see the [`environment.
yaml`](./environment.yaml) file. To import the environment, execute the 
following command prompt commands:

```bash
[mamba | conda | microbamba] create -n env python=3.12
[mamba | conda | microbamba] activate env
[mamba | conda | microbamba | pip] install numpy scipy pandas seaborn matplotlib jupyter gymnasium pytest

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
For the last step, we strongly recommend to 
[Follow Nvidia tutorial to install pytorch](https://pytorch.org/get-started/locally/)
. The code we provide to install pytorch is the one used for our 
project.


- Seaborn
- Matplotlib
- Jupyter
- scipy
- numpy
- pandas
- pytorch (see [Pytorch installation website](https://pytorch.org/get-started/locally/))
- Gymnasium by Farama (previously OpenAI gym)
- pytest (testing included)


---

Script Usage
------------------

Our codebase provides the code implementing and training RA-PbRL 
algorithm. In order to execute the experiments we implemented, refer to 


```bash
python3 src/paper_experiments/pbrl_experiment.py
```


Implemented Algorithms
======================
Bellow is a description and pseudo-code used to implement algorithms 
with the `experiment` API.

Risk-Aware Preference-baser Reinforcement Learning (RA-PbRL)
------------------------------------------------------------

RA-PbRL is a type of Policy-Iteration and "Confidence Bound" 
reinforcement learning algorithm designed for preference-based 
reinforcement learning while maximizing risk-awareness through 
Value-at-Risk penalties. The intuition behind the algorithm depends 
on the idea of confidence bounds. The algorithm begins by computing 
the confidence bound sets within which there is a 1-δ probability 
that the transition function ($\hat{P}_k$), reward function 
($\hat{r}_k (\cdot)$), and policies ($\pi$). The first two are 
computed by 
finding a center to the sets, and then using theoretical bounds to 
limit the set (see UCB algorithm in bandit theory.) As for the 
policy optimization, we optimize for two policies. We accomplish 
this by computing a policy confidence set that contains 
"almost-optimal" policies with a minor sub-optimality gap. Finally, 
the two "most exploratory" policies are chosen (choose most and 
least optimal policies in set.) 

```pseudocode
========================================================================
ALGORITHM - Risk Aware Preference-based Reinforcement Learning (RA-PbRL)
========================================================================
INPUT:      τ1:  list[tuple[State, Action, Reward]],
            τ2:  list[tuple[State, Action, Reward]],

PARAMETERS: K:   int - Number of episodes,
            H:   int - Horizon,
            S:   State - State space cardinality,
            A:   Action - Action space cardinality,
            δ:   float - Theoretical probabilistic guarantee (1-δ),
            n_k: Callable[[State, Action], int] - # times (s,a) visited
            
OUTPUT:     
------------------------------------------------------------------------
P_k   <= argmin_P |P[s, a] @ I[s, a]|^2
B^P_k <= {P'| |P_k|}
```

---

<sup><i>Oregon State University - Corvallis, OR.</sup></i>

