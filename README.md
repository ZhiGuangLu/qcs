# Quantum Correlation Solver (QCS)
QCS is an open-source Python code that allows to study the single-photon transmission and reflection, as well as the  *n*th-order equal-time correlation functions (ETCFs)
in driven-dissipative quantum systems. The handbook about the package and example notebooks can be found in the current directory.
## Applicable conditions
Here, the system needs to meet two conditions. One is that the system Hamiltonian that does't include the coherent driving part must satisfy U(1) symmetry, namely the 
total excitation number conservation. The other is that the driving strength must be small enough. Based on the two points, we could reduce the computation complexity of 
ETCFs from exponential to polynomial when the system contains multiple optical cavities, emitters, and both.

## Computable physical quantites and quantum effects in quantum optics
* Transmission and reflectance spectrums, ETCFs, cross-correlation function, and 2nd-order unequal-time correlation function.
* Photon blockade, antibunching and bunching effects.
* Dynamical photon blockade effects.
## The commonest systems
* Cavity QED system.
* Waveguide QED system.


## Installation
We've uploaded the open source package to *[PyPI](https://pypi.org/project/qcs-phy)*.
```python

pip install qcs_phy
```
## Use

```python
from qcs_phy import qcs
# creat the effective Hamiltonian
...
Heff
# creat the input and output channels
...
Input
Output
# calculate physical quantity
system = qcs(Heff, Input, Output)
gn_0 = system.calculate_quantity(Quantity)
```

For more details and examples on the use of *QCS* see the handbook and example folder.

<img src="https://github.com/ZhiGuangLu/Load-Figures/blob/main/Qcs.png" width="735px">

## Citation

We will have a paper on arXiv.

## Supplementary instruction
Due to the limited level of my programming ability, I can't provide a professional python package, so user first must download the whole files, 
and then call the Qcs.py file. Meanwhile, I've optimized the code to the best of my ability, and I guarantee its corectness
(see the tests folder). Finally, If there are some bugs running the code, please immediately contact me.

## License
QCS is licensed under the terms of the BSD license.

## Update log
`1.0.0` first release
