# Quantum Correlation Solver (QCS)
[![PyPI](https://img.shields.io/pypi/v/qcs_phy)](https://pypi.org/project/qcs-phy/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qcs_phy)](https://pypi.org/project/qcs-phy/)
[![PyPI - Status](https://img.shields.io/pypi/status/qcs_phy)](https://pypi.org/project/qcs-phy/)
[![PyPI - License](https://img.shields.io/pypi/l/qcs_phy)](https://pypi.org/project/qcs-phy/)
[![GitHub repo size](https://img.shields.io/github/repo-size/ZhiGuangLu/qcs)](https://github.com/ZhiGuangLu/qcs)
[![Downloads](https://static.pepy.tech/personalized-badge/qcs-phy?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/qcs-phy)


QCS is an open-source Python code that allows to study the single-photon transmission and reflection, as well as the  *n*th-order equal-time correlation functions (ETCFs)
in driven-dissipative quantum systems. The handbook about the package and example notebooks can be found in the current directory.
## Applicable conditions
Here, the system needs to meet two conditions. One is that the system Hamiltonian that doesn't includes the coherent driving part must satisfy U(1) symmetry, namely the 
total excitation number conservation. The other is that the driving strength must be small enough. Based on the two points, we could reduce the computation complexity of 
ETCFs from exponential to polynomial when the system contains multiple optical cavities, emitters, and both.

## Computable physical quantities and quantum effects in quantum optics
* Transmission and reflectance spectrums, ETCFs, cross-correlation function, and 2nd-order unequal-time correlation function.
* Photon blockade, antibunching, and bunching effects.
* Dynamical photon blockade effects.
## The commonest systems
* Cavity QED system.
* Waveguide QED system.


## Installation
We've uploaded the open source package to [PyPI](https://pypi.org/project/qcs-phy), and you can add this package to your Python with the command,
```
pip install qcs_phy
```
Before installing, please ensure you've added the Python packages: ``` numpy, scipy ```. Besides, please use the latest version by updating your package with the command,
```
pip install qcs_phy -U
```
If you have any problems installing the tool, please open an issue or write to us. 

## Use

```python
from qcs_phy import qcs
# create the effective Hamiltonian
...Heff
# create the input and output channels
...Input
...Output
# calculate the physical quantity
system = qcs(Heff, Input, Output)
result = system.calculate_quantity(Quantity)
```

For more details and examples on the use of *QCS* see the [handbook](https://github.com/ZhiGuangLu/qcs/tree/main/handbook) and [example](https://github.com/ZhiGuangLu/qcs/tree/main/examples) folder. Besides, 
we've provided the numerical comparison results in [tests](https://github.com/ZhiGuangLu/qcs/tree/main/tests) folder.

<img src="https://github.com/ZhiGuangLu/Load-Figures/blob/main/Qcs.png" width="965px">

## Citation
There is an official article related to this package, and the article is available at: http://arxiv.org/abs/2305.08923.
You can cite the article, if this package is greatly helpful for your research.
## Supplementary instruction
Due to the limited level of my programming ability, I've optimized the code to the best of my ability, and I guarantee its correctness (see the [tests](https://github.com/ZhiGuangLu/qcs/tree/main/tests) folder). Finally, If there are some bugs running the code, please immediately contact me.

## License
QCS is licensed under the terms of the BSD license.

## Update log
`1.0.0` first release

`1.0.1` fixed functions: `print_Dim`,`print_basis`,`print_InOutput`,`print_Heff`

`1.0.2` fixed the problem of the denominator about calculating the single-photon transmission and reflection when the collective spins are coherently driven 

`1.0.3` fixed the formula of the single-photon transmission and reflection

`1.0.4` updated the function: `calculate_2nd_uETCF`

`1.0.5` deleted the function: `print_Dim`, and updated the function: `print_basis`

`1.0.7` fixed the private function: `__prestore_HeffList`
