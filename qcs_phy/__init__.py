"""Quantum Correlation Solver (QCS)
This module calculates the nth-order equal-time correlation functions, single-photon transmission and reflection in open quantum systems.
First, the effective Hamiltonian must satisfy U(1) symmetry, namely the total excitation number conservation.
Second, the incoming coherent state amplitude must be small enough, namely the weak driving approximation.
Finally, this module allows multiple incoming coherent states no matter their frequencies are identical or not, namely the multi-drive case,
and could be also used to calculate the cross-correlation function and the 2nd-order unequal-time correlation function.
"""

# Authors: ZhiGuang Lu
# Contact: youngq@hust.edu.cn


from scipy.sparse import csr_matrix, lil_matrix
from scipy.special import comb
import numpy.linalg as nlg
import scipy.linalg as slg
from numpy import emath
import itertools as it
import numpy as np

# from numpy import exp, abs
# from numpy.linalg import inv, eig
# from scipy.linalg import expm


# all possible number types
number_type = [complex, int, float, np.int64, np.float64, np.complex128, np.complex64]


def dagger(ope: str) -> str:
    """
    :param ope: an operator: o
    :return: o^{\\\dagger}
    """
    if ope in ["a", "sm"]:
        if ope == "a":
            return "ad"
        else:
            return "sp"
    elif ope in ["ad", "sp"]:
        if ope == "ad":
            return "a"
        else:
            return "sm"
    elif "Sm" in ope:
        return ope.replace('m', 'p')
    elif "Sp" in ope:
        return ope.replace('p', 'm')
    elif "Sz" in ope:
        return ope
    else:
        print("Please check your symbols again!!!")


def create_basis(n_exc: int, max_dims: list, k: list) -> list:
    """
    :param n_exc: the excitation number
    :param max_dims: the maximum dimension of each mode
    :param k: the corresponding coefficient of each mode in the total excitation number operator
    :return: all the possible basis vectors
    """
    Basis = []
    mod_num = len(k)

    def dfs(idx, sum_, x):
        if idx == mod_num:
            if sum_ == n_exc:
                Basis.append(x[:])
            return
        for i in range(max_dims[idx] + 1):
            x[idx] = i
            if sum_ + i * k[idx] <= n_exc:
                dfs(idx + 1, sum_ + i * k[idx], x)

    dfs(0, 0, [0] * mod_num)
    Basis.reverse()
    return Basis


def basis_dot(m1: list, m2: np.ndarray, coff: np.ndarray) -> csr_matrix:
    """
    :param m1: fixed basis vectors
    :param m2: updated basis vectors after acted by all possible modes
    :param coff: the coefficient created by all possible modes acted on fixed basis vectors
    :return: a "dot" product between fixed basis vectors and updated basis vectors
    """
    line_1 = len(m1)
    line_2 = len(m2)
    m2_list = m2.tolist()
    m3 = lil_matrix((line_1, line_2))
    for i, m1_i in enumerate(m1):
        if m1_i in m2_list:
            j = m2_list.index(m1_i)
            m3[i, j] = np.real(coff[j])
    return csr_matrix(m3)


def sum_sparse(m: list) -> np.ndarray:
    """
    :param m: a list including multiple sparse matrices
    :return: the summation of all sparse matrices in the list
    """
    m_sum = np.zeros(m[0].shape, dtype=complex)
    for mi in m:
        ri = np.repeat(np.arange(mi.shape[0]), np.diff(mi.indptr))
        m_sum[ri, mi.indices] += mi.data
    return m_sum


def left_prod(C: list) -> np.ndarray:
    """
    :param C: a list including multiple matrices, corresponding to the output operators in different excitation numbers
    :return: the matrices product of all matrices in the list
    """
    C_0 = C[0]
    for C_i in C[1:]:
        C_0 = C_0 @ C_i
    return C_0


def right_prod(H: list, B: list, I: list, ome: float, zp: float or int) -> np.ndarray:
    """
    :param H: a list including multiple matrices, corresponding to the effective Hamiltonian in different excitation numbers
    :param B: a list including multiple matrices, corresponding to the input operators in different excitation numbers
    :param I: a list including multiple matrices, corresponding to the identity operators in different excitation numbers
    :param ome: the frequency of the incoming photon
    :param zp: a coefficient for ensuring the invertibility of effective Hamiltonian
    :return: a matrix acquired by a series of matrices multiplicatopm above
    """
    inv = nlg.inv
    n_exc = len(B)
    K = [1j * inv(H[n] - ((n + 1) * ome + zp) * I[n]) for n in range(n_exc)]
    B_n = K[-1] @ B[-1]
    for i in range(n_exc - 2, -1, -1):
        B_n = B_n @ (K[i] @ B[i])
    return B_n


def sum_frequencies(omega: list or np.ndarray, n: int) -> list:
    """
    :param omega: the frequencies of incoming photons, the corresponding type: list or np.ndarray
    :param n: the total excitation number
    :return: all possible frequencies combination in excitation number n
    """
    sum_f = [[0]]
    for k in range(n):
        sum_f.append([ome_1 + ome_2 for ome_1 in sum_f[k] for ome_2 in omega])
    return sum_f[1:]


def n_m_ary(n: int, m: int) -> list:
    """
    :param n: the total excitation number
    :param m: the number of the input modes
    :return: list
    """
    res = []
    for i in range(m ** n):
        num = i
        digits = []
        for _ in range(n):
            digits.append(num % m)
            num //= m
        res.append(digits[::-1])
    return res


def covert_to_decimals(num_list: list, m: int) -> list:
    """
    :param num_list: list
    :param m: the number of input modes
    :return: list
    """
    decimals = []
    n = len(num_list)
    for k in range(1, n + 1):
        num = num_list[:k]
        decimal = 0
        for i in range(k):
            decimal += num[i] * (m ** (k - 1 - i))
        decimals.append(decimal)
    return decimals


def compare_dicts(dict_a: dict, dict_b: dict) -> bool:
    """
    In order to compare two dicts
    :param dict_a: dict a
    :param dict_b: dict b
    :return: True or False
    """
    for key in dict_a.keys():
        if key not in dict_b.keys() or [da[1:] for da in dict_a[key]] != [da[1:] for da in dict_b[key]]:
            return False
    return True


def update_H(H: list) -> list:
    """
    :param H: the effective Hamiltonian, list
    :return: the nonredundant Hamiltonian terms
    """
    H_c = [Hi[0] for Hi in H]
    H_v = [Hi[1:] for Hi in H]
    H_c_new = []
    H_v_new = []
    for k, Hi in enumerate(H_v):
        if Hi in H_v_new:
            loc = H_v_new.index(Hi)
            H_c_new[loc] += H_c[k]
        else:
            H_v_new.append(Hi)
            H_c_new.append(H_c[k])
    return H_c_new, H_v_new


class qcs:
    __Dim = {}
    __BasisList = dict()
    __HeffList = dict()
    __InOutList = dict()
    __Judge_Heff = []
    __Judge_InOut = dict()

    def __init__(self, Heff: list, Input: list, Output: list, ratio=None):
        """
        Here, for each term in the effective Hamiltonian, such as Heff = E*a_1^\\\dagger*a_1, we use a list to represent it, i.e.,
                                            Heff = [E, ("ad", 1), ("a", 1)],
        where the first element represents the corresponding coefficient, and the last two elements represent the operator a_1^\\\dagger and a_1, respectively.

        For example
        \-----------------------------------------------------------------------------------------------\
        Heff = E*a_1^\\\dagger*a_1 + U*a_1^\\\dagger*a_1^\\\dagger*a_1a_1
        Heff = [[E, ("ad", 1), ("a", 1)], [U, ("ad", 1), ("ad", 1), ("a", 1), ("a", 1)]]
        \-----------------------------------------------------------------------------------------------\
        Note that the first and second elements in ("ad", 1) represent the operator and the corresponding subscript, respectively.

        More importantly, we only give three symbols to represent the corresponding system's operators:
        \-----------------------------------------------------------------------------------------------\
        "ad" ==> bosonic creation operator, such as cavity field mode

        "a"  ==> bosonic annilihlation operator, such as cavity field mode

        "sp" ==> raising operator: |e><g\|, such as two-level spin

        "sm" ==> lowering operator: |g><e\|, such as two-level spin

        "Sp_N" ==> Sp_N = \sum_{i=1}^{N}{sp_i}, N collective two-level spins

        "Sm_N" ==> Sm_N = \sum_{i=1}^{N}{sm_i}, N collective two-level spins

        "Sz_N" ==> Sz_N = \sum_{i=1}^{N}{sp_i*sm_i}, N collective two-level spins
        \-----------------------------------------------------------------------------------------------\

        Meanwhile, the Input and Output variables must be acquired by the two functions Input_channel and Output_channel, respectively.

        And the ratio variable represents that each input channel has a corresponding coherent amplitude,
        e.g., b1==>\\beta_1, b2==>\\beta_2, b3==>\\beta_3, and we have ratio = [\\eta_1, \\eta_2, \\eta_3], where \\eta_k = \\beta_k / \\beta_1.
        Of course, if there was only one input channel, we can ignore this variable, and the default value is [1,1,...,1].

        :param Heff: The effective Hamiltonian in the form of List[list].
        :param Input: The input form is acquired by Input_channel function.
        :param Output: The output form is acquired by Output_channel function.
        :param ratio: The ratio between all input coherent amplitudes.
        """
        self.__Heff_c, self.__Heff_v = update_H(Heff)
        self.__Input = {}
        self.__ratio = {}
        self.__Frequency = {}
        loc, temp_in = 0, self.__combine_channel(Input)
        for key, values in temp_in.items():
            self.__Input[key] = values[0]
            self.__Frequency[key] = values[1]
            if ratio == None:
                self.__ratio[key] = 1
            else:
                self.__ratio[key] = ratio[loc]
            loc += 1
        self.__Output = self.__combine_channel(Output)

    def __combine_channel(self, channels: list or tuple) -> dict:
        """
        Let multiple channels fused into a dict
        :param channels: a list including one or multiple (Input/Output) channel
        :return: a dict
        """
        if type(channels) == list or type(channels) == tuple:
            channel_list = {}
            for channel in channels:
                channel_list.update(channel)
            return channel_list
        else:
            return channels

    def Input_channel(channel_name: str, mode: list, frequency) -> dict:
        """
        Assuming channel_name = "b1", the corresponding input-output formalism is
                    b1_{out}(t) = b1_{in}(t) - i * mode.
        For example, when mode is equal to \\sqrt{\\kappa} * a_1, it corresponds to
                    mode = [\\sqrt{\\kappa}, ("a", 1)].
        Obviously, if the mode could consist of multiple system's operators, such as
                mode = \\sqrt{\\kappa_1} * a_1 + \sqrt{\\kappa_2} * a_2,
        which corresponds to
                mode = [[\\sqrt{\\kappa_1}, ("a", 1)], [\\sqrt{\\kappa_2}, ("a", 2)]]
        :param channel_name: The input channel name, such as 'b1', 'b2', and etc.
        :param mode: it consists of system's annihilation operator
        :param frequency: driving frequency or incoming photon frequency
        :return:
        """
        Input = dict()
        if type(mode[0]) in number_type:
            mode = [mode]
        if type(frequency) != list and type(frequency) not in number_type:
            Input[channel_name] = [mode, list(frequency)]
        else:
            Input[channel_name] = [mode, frequency]
        return Input

    def Output_channel(channel_name: str, mode: list) -> dict:
        """
        Assuming channel_name = "c1", the corresponding input-output formalism is
                    c1_{out}(t) = c1_{in}(t) - i * mode.
        For example, when mode is equal to \\sqrt{\\kappa} * a_1, it corresponds to
                    mode = [\\sqrt{\\kappa}, ("a", 1)].
        Obviously, if the mode could consist of multiple system's operators, such as
                mode = \\sqrt{\\kappa_1} * a_1 + \\sqrt{\\kappa_2} * a_2,
        which corresponds to
                mode = [[\\sqrt{\\kappa_1}, ("a", 1)], [\\sqrt{\\kappa_2}, ("a", 2)]]
        :param channel_name: The output channel name, such as 'c1', 'c2', and etc.
        :param mode: It consists of system's annihilation operator
        :return:
        """
        Output = dict()
        if type(mode[0]) in number_type:
            mode = [mode]
        Output[channel_name] = mode
        return Output

    def __excitation_number(self):
        """
        Calculating the total excitation number
        """
        Ope_a = sorted(list(set([item for sublist in self.__Heff_v for item in sublist if item[0] == 'a'])), key=lambda x: x[1])
        Ope_sm = sorted(list(set([item for sublist in self.__Heff_v for item in sublist if item[0] == 'sm'])), key=lambda x: x[1])
        Ope_ad = sorted(list(set([item for sublist in self.__Heff_v for item in sublist if item[0] == 'ad'])), key=lambda x: x[1])
        Ope_sp = sorted(list(set([item for sublist in self.__Heff_v for item in sublist if item[0] == 'sp'])), key=lambda x: x[1])
        Ope_Sz = sorted(list(set([item for sublist in self.__Heff_v for item in sublist if 'Sz' in item[0]])), key=lambda x: x[1])
        Ope_Sp = sorted(list(set([item for sublist in self.__Heff_v for item in sublist if 'Sp' in item[0]])), key=lambda x: x[1])
        Ope_Sm = sorted(list(set([item for sublist in self.__Heff_v for item in sublist if 'Sm' in item[0]])), key=lambda x: x[1])

        Ope_Szs = [0] * (len(Ope_a) + len(Ope_sm)) + Ope_Sz
        Ope_ani = Ope_a + Ope_sm + Ope_Sm
        Ope_cre = Ope_ad + Ope_sp + Ope_Sp
        k = [1] * len(Ope_ani)

        H_mp = [sublist for sublist in self.__Heff_v if len(sublist) > 2]
        for Hi in H_mp:
            ani_ope = [x for x in Hi if x[0] in ["a", "sm"] or "Sm" in x[0]]
            cre_ope = [x for x in Hi if x[0] in ["ad", "sp"] or "Sp" in x[0]]
            if len(ani_ope) < len(cre_ope):
                mod = len(cre_ope) % len(ani_ope)
                if mod != 0:
                    for ani_i in ani_ope:
                        k[Ope_ani.index(ani_i)] = len(cre_ope)
                    for cre_i in cre_ope:
                        k[Ope_cre.index(cre_i)] = len(ani_ope)
                else:
                    for ani_i in ani_ope:
                        k[Ope_ani.index(ani_i)] = int(len(cre_ope) / len(ani_ope))
        if len(Ope_Sz) != 0:
            self.__Ope_Szs = Ope_Szs
        self.__Ope_ani = Ope_ani
        self.__Ope_cre = Ope_cre
        self.__k = k

    def __basis(self, n_exc: int):
        """
        Calculating all the basis vectors in certain excitation number.
        :param n_exc: the excitation number
        """
        max_dims = []
        for O in self.__Ope_ani:
            if O[0] == "a":
                max_dims.append(n_exc)
            elif O[0] == "sm":
                max_dims.append(1)
            elif "Sm" in O[0]:
                N_spin = int("".join(list(filter(str.isdigit, O[0]))))
                if N_spin <= n_exc:
                    max_dims.append(N_spin)
                else:
                    max_dims.append(n_exc)
            else:
                pass

        qcs.__BasisList[n_exc] = create_basis(n_exc, max_dims, self.__k)
        qcs.__Dim[n_exc] = len(qcs.__BasisList[n_exc])

    def __prestore_HeffList(self, n_exc: int):
        """
        In order to decrease the runtime when user calculates a series of parameters, we
        prestore the effective Hamiltonian matrix in the form of sparse matrices.
        :param n_exc: the excitation number
        """
        Heff_nexc = dict()
        bas_fix = qcs.__BasisList[n_exc]
        for Hi in self.__Heff_v:
            H_ind = [cof[-1] for cof in Hi]
            if H_ind.count(H_ind[0]) != len(H_ind):
                Hi_per = list(map(list, set(it.permutations(Hi))))
                rep = False
                for H in Hi_per:
                    H_d = [(dagger(Hk[0]), Hk[1]) for Hk in H]
                    if tuple(H_d) in Heff_nexc:
                        rep = True
                        break
                if rep:
                    Heff_nexc[tuple(Hi)] = csr_matrix(Heff_nexc[tuple(H_d)].toarray().T)
                    continue

            coff = np.ones((len(bas_fix),))
            bas_cor = np.array(bas_fix)
            for H in Hi:
                if H[0] in ["a", "sm"]:
                    loc = self.__Ope_ani.index(H)
                    bas_cor[:, loc] += 1
                    coff = coff * emath.sqrt(bas_cor[:, loc])
                elif H[0] in ["ad", "sp"]:
                    loc = self.__Ope_cre.index(H)
                    coff = coff * emath.sqrt(bas_cor[:, loc])
                    bas_cor[:, loc] -= 1
                elif "Sm" in H[0]:
                    N_spin = int("".join(list(filter(str.isdigit, H[0]))))
                    loc = self.__Ope_ani.index(H)
                    bas_cor[:, loc] += 1
                    coff = coff * emath.sqrt((N_spin - bas_cor[:, loc] + 1) * bas_cor[:, loc])
                elif "Sp" in H[0]:
                    N_spin = int("".join(list(filter(str.isdigit, H[0]))))
                    loc = self.__Ope_cre.index(H)
                    coff = coff * emath.sqrt((N_spin - bas_cor[:, loc] + 1) * bas_cor[:, loc])
                    bas_cor[:, loc] -= 1
                elif "Sz" in H[0]:
                    N_spin = int("".join(list(filter(str.isdigit, H[0]))))
                    loc = self.__Ope_Szs.index(H)
                    coff = coff * bas_cor[:, loc]
                else:
                    pass
                Heff_nexc[tuple(Hi)] = basis_dot(bas_fix, bas_cor, coff)
        qcs.__HeffList[n_exc] = Heff_nexc

    def __prestore_InOutList(self, n_exc: int):
        """
        In order to decrease the runtime when user calculates a series of parameters, we
        prestore the Input-Output mode's matrix in the form of sparse matrices.
        :param n_exc: the excitation number
        """
        InOut = dict()
        In_Opes = [x[1] for InOpe in self.__Input.values() for x in InOpe]
        Out_Opes = [x[1] for OutOpe in self.__Output.values() for x in OutOpe]
        bas_1 = qcs.__BasisList[n_exc - 1]
        bas_2 = qcs.__BasisList[n_exc]
        for In in In_Opes:
            coff = np.ones((len(bas_2),))
            bas_cor = np.array(bas_2)
            loc = self.__Ope_ani.index(In)
            if "Sm" in In[0]:
                N_spin = int("".join(list(filter(str.isdigit, In[0]))))
                coff = coff * emath.sqrt((N_spin - bas_cor[:, loc] + 1) * bas_cor[:, loc])
            else:
                coff = coff * emath.sqrt(bas_cor[:, loc])
            bas_cor[:, loc] -= 1
            InOut[In] = basis_dot(bas_1, bas_cor, coff)
        for Out in Out_Opes:
            coff = np.ones((len(bas_2),))
            bas_cor = np.array(bas_2)
            if Out not in InOut:
                loc = self.__Ope_ani.index(Out)
                if "Sm" in In[0]:
                    N_spin = int("".join(list(filter(str.isdigit, In[0]))))
                    coff = coff * emath.sqrt((N_spin - bas_cor[:, loc] + 1) * bas_cor[:, loc])
                else:
                    coff = coff * emath.sqrt(bas_cor[:, loc])
                bas_cor[:, loc] -= 1
                InOut[Out] = basis_dot(bas_1, bas_cor, coff)
        qcs.__InOutList[n_exc] = InOut

    def __Heff_Matrix(self, n_exc: int):
        """
        By using an appropriate basis vectors in excitation subspace, we can acquire the matrix, Heff^{(n_exc)}.
        :param n_exc: the excitation number
        :return: the effective Hamiltonian in the form of matrix at the certain excitation subspace
        """
        Heff_List = [qcs.__HeffList[n_exc][tuple(Hv)] * self.__Heff_c[k] for k, Hv in enumerate(self.__Heff_v)]
        Heff_m = sum_sparse(Heff_List)
        return Heff_m

    def __Input_Matrix(self, n_exc: int):
        """
        We assume that the input channel is b, and the input-output formalism can be written as
                                b_out(t) = b_in(t) - i * o_b(t).
        Thus, the function will return the matrix, O^{b}_{n_exc-1, n_exc}.
        :param n_exc: the excitation number
        :return: the modes of input channel in the form of matrix
        """
        Int_m = {}
        for key, value in self.__Input.items():
            Int_m[key] = sum_sparse([qcs.__InOutList[n_exc][x[1]] * x[0] for x in value])
        return Int_m

    def __Output_Matrix(self, n_exc: int):
        """
        We assume that the input channel is c, and the input-output formalism can be written as
                                c_out(t) = c_in(t) - i * o_c(t).
        Thus, the function will return the matrix, O^{c}_{n_exc-1, n_exc}.
        :param n_exc: the excitation number
        :return: the modes of output channel in the form of matrix
        """
        Out_m = {}
        for key, value in self.__Output.items():
            Out_m[key] = sum_sparse([qcs.__InOutList[n_exc][x[1]] * x[0] for x in value])
        return Out_m

    def print_Dim(self, n_exc: int, p=1):
        """
        This function is used to print the dimension, which corresponds to excitation number n_exc.
        :param n_exc: the excitation number
        :param p: print (p=1) or return (p=0), the default is printing
        """
        self.__excitation_number()
        if n_exc not in qcs.__Dim.keys():
            self.__basis(n_exc)
        if p == 1:
            print(self.__Dim[n_exc])
        else:
            return self.__Dim[n_exc]

    def print_basis(self, n_exc: int, p=1):
        """
        This function is used to print the basis vectors, which corresponds to excitation number (n_exc).
        :param n_exc: the excitation number
        :param p: print (p=1) or return (p=0), the default is printing
        """
        self.__excitation_number()
        if n_exc not in qcs.__BasisList.keys():
            self.__basis(n_exc)
        if p == 1:
            print(qcs.__BasisList[n_exc])
        else:
            return qcs.__BasisList[n_exc]

    def print_InOutput(self, n_exc: int, channel_name: str, p=1):
        """
        This function is used to print the matrix, which corresponds to the projections of the input/output mode
        onto the direct sum of the (n_exc-1)-th abd (n_exc)-th excitation subspace.
        :param n_exc: the excitation number
        :param channel_name: the name of channel
        :param p: print (p=1) or return (p=0), the default is printing
        """
        self.__excitation_number()
        if n_exc not in qcs.__InOutList.keys():
            self.__basis(n_exc - 1)
            self.__basis(n_exc)
            self.__prestore_InOutList(n_exc)
        if channel_name in self.__Input:
            if p == 1:
                print(self.__Input_Matrix(n_exc)[channel_name])
            else:
                return self.__Input_Matrix(n_exc)[channel_name]
        elif channel_name in self.__Output:
            if p == 1:
                print(self.__Output_Matrix(n_exc)[channel_name])
            else:
                return self.__Output_Matrix(n_exc)[channel_name]
        else:
            print("Sorry, the channel name %s does not exist." % channel_name)

    def print_Heff(self, n_exc: int, p=1):
        """
        This function is used to print the correspondint effective Hamiltonian in the excitation subspace (n_exc).
        :param n_exc: the excitation number
        :param p: print (p=1), return (p=0), the default is printing
        """
        self.__excitation_number()
        if n_exc not in qcs.__HeffList.keys():
            self.__basis(n_exc)
            self.__prestore_HeffList(n_exc)
        if p == 1:
            print(self.__Heff_Matrix(n_exc))
        else:
            return self.__Heff_Matrix(n_exc)

    def __classification(self, photons: list):
        """
        :param photons: the output modes
        :return: the classification label
        """
        In_list, Out_list = set(self.__Input.keys()), set(photons)
        cover_part = In_list & Out_list
        if len(In_list) == 1 and len(Out_list) == 1:
            if cover_part:
                return 1  # one-to-one-same
            else:
                return -1  # one-to-one-differnt
        elif len(In_list) > 1 and len(Out_list) == 1:
            if cover_part:
                return 2  # many-to-one-same
            else:
                return -2  # many-to-one-differnt
        elif len(In_list) == 1 and len(Out_list) > 1:
            if cover_part:
                return 3  # one-to-many-same
            else:
                return -3  # one-to-many-differnt
        else:
            if cover_part:
                return 4  # many-to-many-same
            else:
                return -4  # many-to-many-differnt

    def calculate_quantity(self, Quantity: str, tlist=0, zp=0):
        """Calculating a series of physical quantities. For example,
        \-----------------------------------------------------------------------------------------------\
        Quantity = "c1" ==> single-photon transmission

        Quantity = "c1c1" ==> 2nd-order equal-time correlation function

        Quantity = "c1c2" ==> 2nd-order equal-time cross-correlation function

        Quantity = "c1c1c1" ==> 3rd-order equal-time correlation function

        ...
        \-----------------------------------------------------------------------------------------------\
        Note that these physical quantities describe the statistical properties of output light in the output channel.

        When input channel is different from the output channel, e.g., input channel "b1" and output channel "c1", the
        physical quantity can represent the correlation function about system's modes based on the input-output formalism.
        For example, the input-output relation about output channel "c1" is given by c1_{out} = c1_{in} - i * mode,
        and we assume that mode = \\\sqrt{\\\kappa} * a_1 and Quantity = "c1c1". The 2nd-order equal-time correlation function
        is equivalent to the correlation function of mode a_1.

        Here, we consider the presence of tlist only when the frequencies of incoming coherent states are not identical, and consider
        zp only when the effective Hamiltonian is irreversible, i.e.,

            inv(Heff^{(n)} - ome - i0^{+}) ≠ 0 ==> zp = i0^{+}
        :param Quantity: physical quantity
        :param tlist: a time lsit
        :param zp: an infinitely small quantity
        :return: the corresponding physical quantity
        """
        inv, exp, abs = nlg.inv, np.exp, np.abs
        photons = {}
        for mode in self.__Output.keys():
            photons[mode] = Quantity.count(mode)
        key_out = [key for key in photons for _ in range(photons[key])]
        label = self.__classification(key_out)
        if len(set(key_out)) == 1:
            key_out = key_out[0]
        n_exc = sum(photons.values())
        self.__excitation_number()
        if self.__Heff_v != qcs.__Judge_Heff:
            qcs.__HeffList.clear()
            qcs.__Judge_Heff = self.__Heff_v
            qcs.__Judge_InOut = dict(self.__Input, **self.__Output)
            for n in range(0, n_exc + 1):
                self.__basis(n)
                if n != 0:
                    self.__prestore_HeffList(n)
                    self.__prestore_InOutList(n)
        elif compare_dicts(self.__Input, qcs.__Judge_InOut) and compare_dicts(self.__Output, qcs.__Judge_InOut):
            pass
        else:
            qcs.__InOutList.clear()
            qcs.__Judge_InOut = dict(self.__Input, **self.__Output)
            for n in range(1, n_exc + 1):
                self.__prestore_InOutList(n)
        for n in range(1, n_exc + 1):
            if n not in list(qcs.__HeffList.keys()):
                self.__basis(n)
                self.__prestore_HeffList(n)
                self.__prestore_InOutList(n)
            elif n not in list(qcs.__InOutList.keys()):
                self.__basis(n)
                self.__prestore_InOutList(n)
            else:
                pass

        if label == -1:  # one-to-one-differnt
            key_in = list(self.__Input.keys())[0]
            ome_list = self.__Frequency[key_in]
            if n_exc == 1:
                B, C, H0 = self.__Input_Matrix(1)[key_in].conj().T, self.__Output_Matrix(1)[key_out], self.__Heff_Matrix(1)
                I = np.eye(qcs.__Dim[n_exc])
                if type(ome_list) not in number_type:
                    return [(abs(C @ inv(H0 - (ome + zp) * I) @ B) ** 2)[0, 0] for ome in ome_list]
                else:
                    return (abs(C @ inv(H0 - (ome_list + zp) * I) @ B) ** 2)[0, 0]
            else:
                B = [self.__Input_Matrix(n)[key_in].conj().T for n in range(1, n_exc + 1)]
                C = [self.__Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
                I = [np.eye(qcs.__Dim[n]) for n in range(1, n_exc + 1)]
                C_tot = left_prod(C)
                if type(ome_list) not in number_type:
                    return [(abs(C_tot @ right_prod(H0, B, I, ome, zp)) ** 2 / abs(C[0] @ inv(H0[0] - (ome + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0] for ome in ome_list]
                else:
                    return (abs(C_tot @ right_prod(H0, B, I, ome_list, zp)) ** 2 / abs(C[0] @ inv(H0[0] - (ome_list + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0]

        elif label == 1:  # one-to-one-same
            key_in = list(self.__Input.keys())[0]
            ome_list = self.__Frequency[key_in]
            if n_exc == 1:
                B, C, H0 = self.__Input_Matrix(1)[key_in].conj().T, self.__Output_Matrix(1)[key_out], self.__Heff_Matrix(1)
                I = np.eye(qcs.__Dim[n_exc])
                if type(ome_list) not in number_type:
                    return [(abs(1 + 1j * C @ inv(H0 - (ome + zp) * I) @ B) ** 2)[0, 0] for ome in ome_list]
                else:
                    return (abs(1 + 1j * C @ inv(H0 - (ome_list + zp) * I) @ B) ** 2)[0, 0]
            else:
                B = [self.__Input_Matrix(n)[key_in].conj().T for n in range(1, n_exc + 1)]
                C = [self.__Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
                I = [np.eye(qcs.__Dim[n]) for n in range(1, n_exc + 1)]
                C_tot = [comb(n_exc, n) * left_prod(C[:n]) for n in range(1, n_exc + 1)]
                if type(ome_list) not in number_type:
                    return [(abs(1 + sum([C_tot[n - 1] @ right_prod(H0[:n], B[:n], I[:n], ome, zp) for n in range(1, n_exc + 1)])) ** 2 / \
                             abs(1 + 1j * C[0] @ inv(H0[0] - (ome + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0] for ome in ome_list]
                else:
                    return (abs(1 + sum([C_tot[n - 1] @ right_prod(H0[:n], B[:n], I[:n], ome_list, zp) for n in range(1, n_exc + 1)])) ** 2 / \
                            abs(1 + 1j * C[0] @ inv(H0[0] - (ome_list + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0]

        elif label == -2:  # many-to-one-differnt
            key_in = list(self.__Input.keys())
            ome_list = list(self.__Frequency.values())
            ty_n = ome_list.count(ome_list[0]) == len(ome_list)
            if ty_n == True:  # the identical incoming photon frequencies
                ome_list = ome_list[0]
                if n_exc == 1:
                    B, C, H0 = sum([self.__ratio[key] * self.__Input_Matrix(1)[key].conj().T for key in key_in]), self.__Output_Matrix(1)[key_out], self.__Heff_Matrix(1)
                    k_sum = sum(self.__ratio.values())
                    I = np.eye(qcs.__Dim[n_exc])
                    if type(ome_list) not in number_type:
                        return [(abs(C @ inv(H0 - ome * I) @ B / k_sum) ** 2)[0, 0] for ome in ome_list]
                    else:
                        return (abs(C @ inv(H0 - ome_list * I) @ B / k_sum) ** 2)[0, 0]
                else:
                    B = [sum([self.__ratio[key] * self.__Input_Matrix(n)[key].conj().T for key in key_in]) for n in range(1, n_exc + 1)]
                    C = [self.__Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                    H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
                    I = [np.eye(qcs.__Dim[n]) for n in range(1, n_exc + 1)]
                    C_tot = left_prod(C)
                    if type(ome_list) not in number_type:
                        return [(abs(C_tot @ right_prod(H0, B, I, ome, zp)) ** 2 / abs(C[0] @ inv(H0[0] - (ome + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0] for ome
                                in ome_list]
                    else:
                        return (abs(C_tot @ right_prod(H0, B, I, ome_list, zp)) ** 2 / abs(C[0] @ inv(H0[0] - (ome_list + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0]
            else:  # the different incoming photon frequencies
                sum_f = sum_frequencies(ome_list, n_exc)
                H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
                I = [np.eye(qcs.__Dim[n]) for n in range(1, n_exc + 1)]
                Klist = [[inv(H0[n - 1] - (ome + zp) * I[n - 1]) for ome in sum_f[n - 1]] for n in range(1, n_exc + 1)]
                C = [self.__Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                ary = n_m_ary(n_exc, len(key_in))
                comb_s = [covert_to_decimals(ary_i, len(key_in)) for ary_i in ary]
                if n_exc == 1:
                    k_sum = sum(self.__ratio.values())
                    B = [self.__Input_Matrix(1)[key].conj().T for k, key in enumerate(key_in)]
                    if type(tlist) not in number_type:
                        T_t = []
                        append = T_t.append
                        for t in tlist:
                            N = [self.__ratio[key] * exp(-1j * self.__Frequency[key] * t) for key in key_in]
                            Blist = [N[k] * B[k] for k in range(len(B))]
                            append((abs(C[0] @ sum([Klist[0][i] @ Blist[i] for i in range(len(key_in))])) ** 2 / abs(k_sum) ** 2)[0, 0])
                        return T_t
                    else:
                        N = [self.__ratio[key] * exp(-1j * self.__Frequency[key] * tlist) for key in key_in]
                        Blist = [N[k] * B[k] for k in range(len(B))]
                        return (abs(C[0] @ sum([Klist[0][i] @ Blist[i] for i in range(len(key_in))])) ** 2 / abs(k_sum) ** 2)[0, 0]
                else:
                    C_tot = left_prod(C)
                    B = [[self.__Input_Matrix(n)[key].conj().T for key in key_in] for n in range(1, n_exc + 1)]
                    if type(tlist) not in number_type:
                        gn_t = []
                        append = gn_t.append
                        for t in tlist:
                            N = [self.__ratio[key] * exp(-1j * self.__Frequency[key] * t) for key in key_in]
                            Blist = [[Bi[k] * N[k] for k in range(len(key_in))] for Bi in B]
                            Gn_0 = 0
                            for j, c in enumerate(comb_s):
                                C_tot_c = C_tot.copy()
                                for k, ci in enumerate(c[::-1]):
                                    C_tot_c = C_tot_c @ Klist[n_exc - k - 1][ci] @ Blist[n_exc - k - 1][ary[j][n_exc - k - 1]]
                                Gn_0 += C_tot_c
                            append((abs(Gn_0) ** 2 / abs(C[0] @ sum([Klist[0][i] @ Blist[0][i] for i in range(len(key_in))])) ** (2 * n_exc))[0, 0])
                        return gn_t
                    else:
                        N = [self.__ratio[key] * exp(-1j * self.__Frequency[key] * tlist) for key in key_in]
                        Blist = [[Bi[k] * N[k] for k in range(len(key_in))] for Bi in B]
                        Gn_0 = 0
                        for c in comb_s:
                            C_tot_c = C_tot.copy()
                            for k, ci in enumerate(c[::-1]):
                                C_tot_c = C_tot_c @ Klist[n_exc - k - 1][ci] @ Blist[n_exc - k - 1][ci]
                            Gn_0 += C_tot_c
                        return (abs(Gn_0) ** 2 / abs(C[0] @ sum([Klist[0][i] @ Blist[0][i] for i in range(len(key_in))])) ** (2 * n_exc))[0, 0]

        elif label == 2:  # many-to-one-same
            key_in = list(self.__Input.keys())
            ome_list = list(self.__Frequency.values())
            ty_n = ome_list.count(ome_list[0]) == len(ome_list)
            if ty_n == True:  # the identical incoming photon frequencies
                ome_list = ome_list[0]
                if n_exc == 1:
                    B, C, H0 = sum([self.__ratio[key] * self.__Input_Matrix(1)[key].conj().T for key in key_in]), self.__Output_Matrix(1)[key_out], self.__Heff_Matrix(1)
                    k_sum = sum(self.__ratio.values())
                    I = np.eye(qcs.__Dim[n_exc])
                    if type(ome_list) not in number_type:
                        return [(abs((self.__ratio[key_out] + 1j * C @ inv(H0 - ome * I) @ B) / k_sum) ** 2)[0, 0] for ome in ome_list]
                    else:
                        return (abs((self.__ratio[key_out] + 1j * C @ inv(H0 - ome_list * I) @ B) / k_sum) ** 2)[0, 0]
                else:
                    B = [sum([self.__ratio[key] * self.__Input_Matrix(n)[key].conj().T for key in key_in]) for n in range(1, n_exc + 1)]
                    C = [self.__Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                    H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
                    I = [np.eye(qcs.__Dim[n]) for n in range(1, n_exc + 1)]
                    C_tot = [self.__ratio[key_out] ** (n_exc - n) * comb(n_exc, n) * left_prod(C[:n]) for n in range(1, n_exc + 1)]
                    coff_free = self.__ratio[key_out] ** n_exc
                    if type(ome_list) not in number_type:
                        return [(abs(coff_free + sum([C_tot[n - 1] @ right_prod(H0[:n], B[:n], I[:n], ome, zp) for n in range(1, n_exc + 1)])) ** 2 / \
                                 abs(self.__ratio[key_out] + 1j * C[0] @ inv(H0[0] - (ome + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0] for ome in ome_list]
                    else:
                        return (abs(coff_free + sum([C_tot[n - 1] @ right_prod(H0[:n], B[:n], I[:n], ome_list, zp) for n in range(1, n_exc + 1)])) ** 2 / \
                                abs(self.__ratio[key_out] + 1j * C[0] @ inv(H0[0] - (ome_list + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0]
            else:  # the identical incoming photon frequencies
                sum_f = sum_frequencies(ome_list, n_exc)
                H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
                I = [np.eye(qcs.__Dim[n]) for n in range(1, n_exc + 1)]
                Klist = [[inv(H0[n - 1] - (ome + zp) * I[n - 1]) for ome in sum_f[n - 1]] for n in range(1, n_exc + 1)]
                C = [self.__Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                loc = key_in.index(key_out)
                if n_exc == 1:
                    k_sum = sum(self.__ratio.values())
                    B = [self.__Input_Matrix(1)[key].conj().T for k, key in enumerate(key_in)]
                    if type(tlist) not in number_type:
                        T_t = []
                        append = T_t.append
                        for t in tlist:
                            N = [self.__ratio[key] * exp(-1j * self.__Frequency[key] * t) for key in key_in]
                            Blist = [N[k] * B[k] for k in range(len(B))]
                            append((abs(N[loc] + 1j * C[0] @ sum([Klist[0][i] @ Blist[i] for i in range(len(key_in))])) ** 2 / abs(k_sum) ** 2)[0, 0])
                        return T_t
                    else:
                        N = [self.__ratio[key] * exp(-1j * self.__Frequency[key] * tlist) for key in key_in]
                        Blist = [N[k] * B[k] for k in range(len(B))]
                        return (abs(N[loc] + 1j * C[0] @ sum([Klist[0][i] @ Blist[i] for i in range(len(key_in))])) ** 2 / abs(k_sum) ** 2)[0, 0]
                else:
                    ary_s = [n_m_ary(n, len(key_in)) for n in range(1, n_exc + 1)]
                    comb_s = [[covert_to_decimals(ary_i, len(key_in)) for ary_i in ary] for ary in ary_s]
                    C_tot = [left_prod(C[:n]) for n in range(1, n_exc + 1)]
                    B = [[self.__Input_Matrix(n)[key].conj().T for key in key_in] for n in range(1, n_exc + 1)]
                    if type(tlist) not in number_type:
                        gn_t = []
                        append = gn_t.append
                        for t in tlist:
                            N = [self.__ratio[key] * exp(-1j * self.__Frequency[key] * t) for key in key_in]
                            coff_free = [comb(n_exc, n) * N[loc] ** (n_exc - n) for n in range(n_exc + 1)]
                            Blist = [[Bi[k] * N[k] for k in range(len(key_in))] for Bi in B]
                            Gn_0 = coff_free[0]
                            for n in range(1, n_exc + 1):
                                G_tot = 0
                                for j, c in enumerate(comb_s[n - 1]):
                                    C_tot_c = C_tot[n - 1].copy()
                                    for k, ci in enumerate(c[::-1]):
                                        C_tot_c = C_tot_c @ Klist[n - k - 1][ci] @ Blist[n - k - 1][ary_s[n - 1][j][n - k - 1]]
                                    G_tot += C_tot_c
                                Gn_0 += (1j) ** n * coff_free[n] * G_tot
                            append((abs(Gn_0) ** 2 / abs(N[loc] + 1j * C[0] @ sum([Klist[0][i] @ Blist[0][i] for i in range(len(key_in))])) ** (2 * n_exc))[0, 0])
                        return gn_t
                    else:
                        N = [self.__ratio[key] * exp(-1j * self.__Frequency[key] * tlist) for key in key_in]
                        coff_free = [comb(n_exc, n) * N[loc] ** (n_exc - n) for n in range(n_exc + 1)]
                        Blist = [[Bi[k] * N[k] for k in range(len(key_in))] for Bi in B]
                        Gn_0 = coff_free[0]
                        for n in range(1, n_exc + 1):
                            G_tot = 0
                            for j, c in enumerate(comb_s[n - 1]):
                                C_tot_c = C_tot[n - 1].copy()
                                for k, ci in enumerate(c[::-1]):
                                    C_tot_c = C_tot_c @ Klist[n - k - 1][ci] @ Blist[n - k - 1][ary_s[n - 1][j][n - k - 1]]
                                G_tot += C_tot_c
                            Gn_0 += (1j) ** n * coff_free[n] * G_tot
                        return (abs(Gn_0) ** 2 / abs(N[loc] + 1j * C[0] @ sum([Klist[0][i] @ Blist[0][i] for i in range(len(key_in))])) ** (2 * n_exc))[0, 0]

        elif label == -3:  # one-to-many-differnt
            key_in = list(self.__Input.keys())[0]
            ome_list = self.__Frequency[key_in]
            B = [self.__Input_Matrix(n)[key_in].conj().T for n in range(1, n_exc + 1)]
            C = [self.__Output_Matrix(n)[key_out[n - 1]] for n in range(1, n_exc + 1)]
            H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
            I = [np.eye(qcs.__Dim[n]) for n in range(1, n_exc + 1)]
            C_tot = left_prod(C)
            if type(ome_list) not in number_type:
                correlation = []
                append = correlation.append
                for ome in ome_list:
                    G = abs(C_tot @ right_prod(H0, B, I, ome, zp)) ** 2
                    K1 = inv(H0[0] - (ome + zp) * I[0]) @ B[0]
                    for key in photons:
                        G /= abs((self.__Output_Matrix(1)[key] @ K1) ** (photons[key])) ** 2
                    append(G[0, 0])
                return correlation
            else:
                correlation = abs(C_tot @ right_prod(H0, B, I, ome_list, zp)) ** 2
                K1 = inv(H0[0] - (ome_list + zp) * I[0]) @ B[0]
                for key in photons:
                    correlation /= abs((self.__Output_Matrix(1)[key] @ K1) ** (photons[key])) ** 2
                return correlation[0, 0]

        elif label == 3:  # one-to-many-same
            key_in = list(self.__Input.keys())[0]
            loc = key_out.index(key_in)
            key_out_new = key_out[: loc] + key_out[loc + photons[key_in]:] + key_out[loc: loc + photons[key_in]]
            ome_list = self.__Frequency[key_in]
            B = [self.__Input_Matrix(n)[key_in].conj().T for n in range(1, n_exc + 1)]
            C = [self.__Output_Matrix(n)[key_out_new[n - 1]] for n in range(1, n_exc + 1)]
            H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
            I = [np.eye(qcs.__Dim[n]) for n in range(1, n_exc + 1)]
            C_tot = [comb(photons[key_in], n) * left_prod(C[:n_exc - n]) for n in range(photons[key_in] + 1)]
            C1_out = {}
            for key in key_out_new:
                C1_out[key] = self.__Output_Matrix(1)[key]
            if type(ome_list) not in number_type:
                correlation = []
                append = correlation.append
                for ome in ome_list:
                    G = abs(sum([C_tot[n] @ right_prod(H0[:n_exc - n], B[:n_exc - n], I[:n_exc - n], ome, zp) for n in range(photons[key_in] + 1)])) ** 2
                    K1 = inv(H0[0] - (ome + zp) * I[0]) @ B[0]
                    for key in photons:
                        if key == key_in:
                            G /= abs((1 + 1j * C1_out[key] @ K1) ** (photons[key])) ** 2
                        else:
                            G /= abs((C1_out[key] @ K1) ** (photons[key])) ** 2
                    append(G[0, 0])
                return correlation
            else:
                G = abs(sum([C_tot[n] @ right_prod(H0[:n_exc - n], B[:n_exc - n], I[:n_exc - n], ome_list, zp) for n in range(photons[key_in] + 1)])) ** 2
                K1 = inv(H0[0] - (ome_list + zp) * I[0]) @ B[0]
                for key in photons:
                    if key == key_in:
                        G /= abs((1 + 1j * C1_out[key] @ K1) ** (photons[key])) ** 2
                    else:
                        G /= abs((C1_out[key] @ K1) ** (photons[key])) ** 2
                return G[0, 0]

        elif label == -4:  # many-to-many-different
            print("This function is not yet available")
        else:  # many-to-many-same
            print("This function is not yet available")

    def calculate_2nd_uETCF(self, channel_name: str, tau=0, zp=0):
        """Calculating the 2nd-order unequal-time coreelation function . For example,
                \-----------------------------------------------------------------------------------------------\
                channel_name = "c1" ==>

                g^{(2)}(\\\ tau) = <c1^{\\\dagger}(0)c1^{\\\dagger}(\\\ tau)c1(\\\ tau) c1(0)> / <c1^{\\\dagger} c1>^2

                \-----------------------------------------------------------------------------------------------\
                :param channel_name: the name of output channel
                :param tau: the delay time
                :param zp: an infinitely small quantity
                :return: the 2nd-order unequal-time coreelation function (uETCF)
        """
        inv, abs, expm = nlg.inv, np.abs, slg.expm  # Assign to a local variable
        key_out = [channel_name for _ in range(2)]
        label = self.__classification(key_out)
        self.__excitation_number()
        if self.__Heff_v != qcs.__Judge_Heff:
            qcs.__BasisList.clear()
            qcs.__Judge_Heff = self.__Heff_v
            qcs.__Judge_InOut = dict(self.__Input, **self.__Output)
            for n in range(0, 3):
                self.__basis(n)
                if n != 0:
                    self.__prestore_HeffList(n)
                    self.__prestore_InOutList(n)
        elif compare_dicts(self.__Input, qcs.__Judge_InOut) and compare_dicts(self.__Output, qcs.__Judge_InOut):
            pass
        else:
            qcs.__Judge_InOut = dict(self.__Input, **self.__Output)
            for n in range(1, 3):
                self.__prestore_InOutList(n)
        for n in range(1, 3):
            if n not in list(qcs.__BasisList.keys()):
                self.__basis(n)
                self.__prestore_HeffList(n)
                self.__prestore_InOutList(n)
            else:
                pass
        if label == -1:  # one-to-one-differnt
            key_in = list(self.__Input.keys())[0]
            ome = self.__Frequency[key_in]
            B = [self.__Input_Matrix(n)[key_in].conj().T for n in range(1, 3)]
            C = [self.__Output_Matrix(n)[channel_name] for n in range(1, 3)]
            I = [np.eye(qcs.__Dim[n]) for n in range(1, 3)]
            Heff = [self.__Heff_Matrix(n) - (n * ome + zp) * I[n - 1] for n in range(1, 3)]
            P1 = C[0] @ inv(Heff[0]) @ B[0]
            P1_R = inv(Heff[0]) @ B[0]
            P2_R = C[1] @ inv(Heff[1]) @ B[1] @ P1_R
            if type(tau) not in number_type:
                g2_t = []
                append = g2_t.append
                for t in tau:
                    P1_t = expm(-1j * Heff[0] * abs(t))
                    append((abs(P1 ** 2 + C[0] @ P1_t @ P2_R - C[0] @ P1_t @ P1_R * P1) ** 2 / abs(P1) ** 4)[0, 0])
                return g2_t
            else:
                P1_t = expm(-1j * Heff[0] * abs(tau))
                return (abs(P1 ** 2 + C[0] @ P1_t @ P2_R - C[0] @ P1_t @ P1_R * P1) ** 2 / abs(P1) ** 4)[0, 0]

        elif label == 1:  # one-to-one-same
            key_in = list(self.__Input.keys())[0]
            ome = self.__Frequency[key_in]
            B = [self.__Input_Matrix(n)[key_in].conj().T for n in range(1, 3)]
            C = [self.__Output_Matrix(n)[channel_name] for n in range(1, 3)]
            I = [np.eye(qcs.__Dim[n]) for n in range(1, 3)]
            Heff = [self.__Heff_Matrix(n) - (n * ome + zp) * I[n - 1] for n in range(1, 3)]
            P1 = C[0] @ inv(Heff[0]) @ B[0]
            P1_R = inv(Heff[0]) @ B[0]
            P2_R = C[1] @ inv(Heff[1]) @ B[1] @ P1_R
            if type(tau) not in number_type:
                g2_t = []
                append = g2_t.append
                for t in tau:
                    P1_t = expm(-1j * Heff[0] * abs(t))
                    append((abs((1 + 1j * P1) ** 2 - C[0] @ P1_t @ P2_R + C[0] @ P1_t @ P1_R * P1) ** 2 / abs(1 + 1j * P1) ** 4)[0, 0])
                return g2_t
            else:
                P1_t = expm(-1j * Heff[0] * abs(tau))
                return (abs((1 + 1j * P1) ** 2 - C[0] @ P1_t @ P2_R + C[0] @ P1_t @ P1_R * P1) ** 2 / abs(1 + 1j * P1) ** 4)[0, 0]

        elif label == -2:  # many-to-one-differnt
            key_in = list(self.__Input.keys())
            ome = list(self.__Frequency.values())[0]
            B = [sum([self.__ratio[key] * self.__Input_Matrix(n)[key].conj().T for key in key_in]) for n in range(1, 3)]
            C = [self.__Output_Matrix(n)[channel_name] for n in range(1, 3)]
            I = [np.eye(qcs.__Dim[n]) for n in range(1, 3)]
            Heff = [self.__Heff_Matrix(n) - (n * ome + zp) * I[n - 1] for n in range(1, 3)]
            P1 = C[0] @ inv(Heff[0]) @ B[0]
            P1_R = inv(Heff[0]) @ B[0]
            P2_R = C[1] @ inv(Heff[1]) @ B[1] @ P1_R
            if type(tau) not in number_type:
                g2_t = []
                append = g2_t.append
                for t in tau:
                    P1_t = expm(-1j * Heff[0] * abs(t))
                    append((abs(P1 ** 2 + C[0] @ P1_t @ P2_R - C[0] @ P1_t @ P1_R * P1) ** 2 / abs(P1) ** 4)[0, 0])
                return g2_t
            else:
                P1_t = expm(-1j * Heff[0] * abs(tau))
                return (abs(P1 ** 2 + C[0] @ P1_t @ P2_R - C[0] @ P1_t @ P1_R * P1) ** 2 / abs(P1) ** 4)[0, 0]

        elif label == 2:  # many-to-one-same
            key_in = list(self.__Input.keys())
            ome = list(self.__Frequency.values())[0]
            B = [sum([self.__ratio[key] * self.__Input_Matrix(n)[key].conj().T for key in key_in]) for n in range(1, 3)]
            C = [self.__Output_Matrix(n)[channel_name] for n in range(1, 3)]
            I = [np.eye(qcs.__Dim[n]) for n in range(1, 3)]
            Heff = [self.__Heff_Matrix(n) - (n * ome + zp) * I[n - 1] for n in range(1, 3)]
            P1 = C[0] @ inv(Heff[0]) @ B[0]
            P1_R = inv(Heff[0]) @ B[0]
            P2_R = C[1] @ inv(Heff[1]) @ B[1] @ P1_R
            ratio_c = self.__ratio[channel_name]
            if type(tau) not in number_type:
                g2_t = []
                append = g2_t.append
                for t in tau:
                    P1_t = expm(-1j * Heff[0] * abs(t))
                    append((abs((ratio_c + 1j * P1) ** 2 - C[0] @ P1_t @ P2_R + C[0] @ P1_t @ P1_R * P1) ** 2 / abs(ratio_c + 1j * P1) ** 4)[0, 0])
                return g2_t
            else:
                P1_t = expm(-1j * Heff[0] * abs(tau))
                return (abs((ratio_c + 1j * P1) ** 2 - C[0] @ P1_t @ P2_R + C[0] @ P1_t @ P1_R * P1) ** 2 / abs(ratio_c + 1j * P1) ** 4)[0, 0]

        else:
            print("This function is not yet available")
