from scipy.sparse import csr_matrix, lil_matrix
from scipy.special import comb
from numpy.linalg import inv, eig
from numpy import exp, abs, sqrt, emath
import itertools as it
import numpy as np

number_type = [complex, int, float, np.int64, np.float64, np.complex128, np.complex64]


def dagger(ope):
    if ope in ["a", "sm"]:
        if ope == "a":
            return "ad"
        else:
            return "sp"
    else:
        if ope == "ad":
            return "a"
        else:
            return "sm"


def create_basis(n_exc, max_dims, k):
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


def basis_dot(m1, m2, coff):
    line_1 = len(m1)
    line_2 = len(m2)
    m2_list = m2.tolist()
    m3 = lil_matrix((line_1, line_2))
    for i, m1_i in enumerate(m1):
        if m1_i in m2_list:
            j = m2_list.index(m1_i)
            m3[i, j] = np.real(coff[j])
    return csr_matrix(m3)


def sum_sparse(m):
    m_sum = np.zeros(m[0].shape, dtype=complex)
    for mi in m:
        ri = np.repeat(np.arange(mi.shape[0]), np.diff(mi.indptr))
        m_sum[ri, mi.indices] += mi.data
    return m_sum


def left_prod(C):
    C_0 = C[0]
    for C_i in C[1:]:
        C_0 = C_0 @ C_i
    return C_0


def right_prod(H, B, I, ome, zp):
    n_exc = len(B)
    K = [1j * inv(H[n] - ((n + 1) * ome + zp) * I[n]) for n in range(n_exc)]
    B_n = K[-1] @ B[-1]
    for i in range(n_exc - 2, -1, -1):
        B_n = B_n @ (K[i] @ B[i])
    return B_n


def sum_frequencies(omega, n):
    sum_f = [[0]]
    for k in range(n):
        sum_f.append([ome_1 + ome_2 for ome_1 in sum_f[k] for ome_2 in omega])
    return sum_f[1:]


def n_m_ary(n, m):
    res = []
    for i in range(m ** n):
        num = i
        digits = []
        for _ in range(n):
            digits.append(num % m)
            num //= m
        res.append(digits[::-1])
    return res


def covert_to_decimals(num_list, m):
    decimals = []
    n = len(num_list)
    for k in range(1, n + 1):
        num = num_list[:k]
        decimal = 0
        for i in range(k):
            decimal += num[i] * (m ** (k - 1 - i))
        decimals.append(decimal)
    return decimals


class Qcs:
    __Dim = {}
    __BasisList = dict()
    __HeffList = dict()
    __InOutList = dict()
    __Judge_Heff = []
    __Judge_InOut = set()

    def __init__(self, Heff, Input, Output, ratio=None):
        self.Heff_c = [Hi[0] for Hi in Heff]
        self.Heff_v = [Hi[1:] for Hi in Heff]
        self.Input = {}
        self.ratio = {}
        self.Frequency = {}
        loc, temp_in = 0, self.__combine_channel(Input)
        for key, values in temp_in.items():
            self.Input[key] = values[0]
            self.Frequency[key] = values[1]
            if ratio == None:
                self.ratio[key] = 1
            else:
                self.ratio[key] = ratio[loc]
            loc += 1
        self.Output = self.__combine_channel(Output)

    def __combine_channel(self, channels):
        if type(channels) == list or type(channels) == tuple:
            channel_list = {}
            for channel in channels:
                channel_list.update(channel)
            return channel_list
        else:
            return channels

    def Input_channel(channel_name, mode, frequency):
        Input = dict()
        if type(mode[0]) in number_type:
            mode = [mode]
        if type(frequency) != list and type(frequency) not in number_type:
            Input[channel_name] = [mode, list(frequency)]
        else:
            Input[channel_name] = [mode, frequency]
        return Input

    def Output_channel(channel_name, mode):
        Output = dict()
        if type(mode[0]) in number_type:
            mode = [mode]
        Output[channel_name] = mode
        return Output

    def __excitation_number(self):
        Ope_a = sorted(list(set([item for sublist in self.Heff_v for item in sublist if item[0] == 'a'])), key=lambda x: x[1])
        Ope_sm = sorted(list(set([item for sublist in self.Heff_v for item in sublist if item[0] == 'sm'])), key=lambda x: x[1])
        Ope_ad = sorted(list(set([item for sublist in self.Heff_v for item in sublist if item[0] == 'ad'])), key=lambda x: x[1])
        Ope_sp = sorted(list(set([item for sublist in self.Heff_v for item in sublist if item[0] == 'sp'])), key=lambda x: x[1])

        Ope_ani = Ope_a + Ope_sm
        Ope_cre = Ope_ad + Ope_sp
        k = [1] * len(Ope_ani)

        H_mp = [sublist for sublist in self.Heff_v if len(sublist) > 2]
        for Hi in H_mp:
            ani_ope = [x for x in Hi if x[0] in ["a", "sm"]]
            cre_ope = [x for x in Hi if x[0] in ["ad", "sp"]]
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

        self.Ope_ani = Ope_ani
        self.Ope_cre = Ope_cre
        self.k = k

    def __basis(self, n_exc):
        max_dims = [n_exc if O[0] == "a" else 1 for O in self.Ope_ani]
        Qcs.__BasisList[n_exc] = create_basis(n_exc, max_dims, self.k)
        Qcs.__Dim[n_exc] = len(Qcs.__BasisList[n_exc])

    def __prestore_HeffList(self, n_exc):
        Heff_nexc = dict()
        bas_fix = Qcs.__BasisList[n_exc]
        for Hi in self.Heff_v:
            H_cof = [cof[-1] for cof in Hi]
            if H_cof.count(H_cof[0]) != len(H_cof):
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
                    loc = self.Ope_ani.index(H)
                    bas_cor[:, loc] += 1
                    coff = coff * emath.sqrt(bas_cor[:, loc])
                else:
                    loc = self.Ope_cre.index(H)
                    coff = coff * emath.sqrt(bas_cor[:, loc])
                    bas_cor[:, loc] -= 1
                Heff_nexc[tuple(Hi)] = basis_dot(bas_fix, bas_cor, coff)
        Qcs.__HeffList[n_exc] = Heff_nexc

    def __prestore_InOutList(self, n_exc):
        InOut = dict()
        In_Opes = [x[1] for InOpe in self.Input.values() for x in InOpe]
        Out_Opes = [x[1] for OutOpe in self.Output.values() for x in OutOpe]
        bas_1 = Qcs.__BasisList[n_exc - 1]
        bas_2 = Qcs.__BasisList[n_exc]
        for In in In_Opes:
            coff = np.ones((len(bas_2),))
            bas_cor = np.array(bas_2)
            loc = self.Ope_ani.index(In)
            coff = coff * emath.sqrt(bas_cor[:, loc])
            bas_cor[:, loc] -= 1
            InOut[In] = basis_dot(bas_1, bas_cor, coff)
        for Out in Out_Opes:
            coff = np.ones((len(bas_2),))
            bas_cor = np.array(bas_2)
            if Out not in InOut:
                loc = self.Ope_ani.index(Out)
                coff = coff * emath.sqrt(bas_cor[:, loc])
                bas_cor[:, loc] -= 1
                InOut[Out] = basis_dot(bas_1, bas_cor, coff)
        Qcs.__InOutList[n_exc] = InOut

    def __Heff_Matrix(self, n_exc):
        Heff_List = np.array(list(Qcs.__HeffList[n_exc].values())) * self.Heff_c
        Heff_m = sum_sparse(list(Heff_List))
        return Heff_m

    def __Input_Matrix(self, n_exc):
        Int_m = {}
        for key, value in self.Input.items():
            Int_m[key] = sum_sparse([Qcs.__InOutList[n_exc][x[1]] * x[0] for x in value])
        return Int_m

    def __Output_Matrix(self, n_exc):
        Out_m = {}
        for key, value in self.Output.items():
            Out_m[key] = sum_sparse([Qcs.__InOutList[n_exc][x[1]] * x[0] for x in value])
        return Out_m

    def __classification(self, photons):
        In_list, Out_list = set(self.Input.keys()), set(photons.keys())
        cover_part = In_list & Out_list
        if len(In_list) == 1 and len(Out_list) == 1:
            if cover_part:
                return 1  # one-to-one-same
            else:
                return -1  # one-to-one-differnt
        elif len(In_list) > 1 and len(Out_list) == 1:
            if cover_part:
                return 2  # multiple-to-one-same
            else:
                return -2  # multiple-to-one-differnt
        elif len(In_list) == 1 and len(Out_list) > 1:
            if cover_part:
                return 3  # one-to-multiple-same
            else:
                return -3  # one-to-multiple-differnt
        else:
            if cover_part:
                return 4  # multiple-to-multiple-same
            else:
                return -4  # multiple-to-multiple-differnt

    def calculate_quantity(self, Quantity, tlist=0, zp=0):
        photons = {}
        for mode in self.Output.keys():
            photons[mode] = Quantity.count(mode)
        key_out = [key for key in photons for i in range(photons[key])]
        if len(set(key_out)) == 1:
            key_out = key_out[0]
        self.__excitation_number()
        n_exc = sum(photons.values())
        if self.Heff_v != Qcs.__Judge_Heff:
            Qcs.__Judge_Heff = self.Heff_v
            Qcs.__Judge_InOut = set(self.Input.keys()).union(set(self.Output.keys()))
            for n in range(0, n_exc + 1):
                self.__basis(n)
                if n != 0:
                    self.__prestore_HeffList(n)
                    self.__prestore_InOutList(n)
        elif set(self.Input.keys()) <= Qcs.__Judge_InOut and set(self.Output.keys()) <= Qcs.__Judge_InOut:
            pass
        else:
            Qcs.__Judge_InOut = set(self.Input.keys()).union(set(self.Output.keys()))
            for n in range(1, n_exc + 1):
                self.__prestore_InOutList(n)
        for n in range(1, n_exc + 1):
            if n not in list(Qcs.__BasisList.keys()):
                self.__basis(n)
                self.__prestore_HeffList(n)
                self.__prestore_InOutList(n)
            else:
                pass
        label = self.__classification(photons)
        if label == -1:  # one-to-one-differnt
            key_in = list(self.Input.keys())[0]
            # key_out = list(self.Output.keys())[0]
            ome_list = self.Frequency[key_in]
            if n_exc == 1:
                B, C, H0 = self.__Input_Matrix(1)[key_in].conj().T, self.__Output_Matrix(1)[key_out], self.__Heff_Matrix(1)
                I = np.eye(Qcs.__Dim[n_exc])
                if type(ome_list) not in number_type:
                    return [(abs(C @ inv(H0 - (ome + zp) * I) @ B) ** 2)[0, 0] for ome in ome_list]
                else:
                    return (abs(C @ inv(H0 - (ome_list + zp) * I) @ B) ** 2)[0, 0]
            else:
                B = [self.__Input_Matrix(n)[key_in].conj().T for n in range(1, n_exc + 1)]
                C = [self.__Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
                I = [np.eye(Qcs.__Dim[n]) for n in range(1, n_exc + 1)]
                C_tot = left_prod(C)
                if type(ome_list) not in number_type:
                    return [(abs(C_tot @ right_prod(H0, B, I, ome, zp)) ** 2 / abs(C[0] @ inv(H0[0] - (ome + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0] for ome in ome_list]
                else:
                    return (abs(C_tot @ right_prod(H0, B, I, ome_list, zp)) ** 2 / abs(C[0] @ inv(H0[0] - (ome_list + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0]

        elif label == 1:  # one-to-one-same
            key_in = list(self.Input.keys())[0]
            # key_out = list(self.Output.keys())[0]
            ome_list = self.Frequency[key_in]
            if n_exc == 1:
                B, C, H0 = self.__Input_Matrix(1)[key_in].conj().T, self.__Output_Matrix(1)[key_out], self.__Heff_Matrix(1)
                dim = H0.shape[0]
                I = np.eye(Qcs.__Dim[n_exc])
                if type(ome_list) not in number_type:
                    return [(abs(1 + 1j * C @ inv(H0 - (ome + zp) * I) @ B) ** 2)[0, 0] for ome in ome_list]
                else:
                    return (abs(1 + 1j * C @ inv(H0 - (ome_list + zp) * I) @ B) ** 2)[0, 0]
            else:
                B = [self.__Input_Matrix(n)[key_in].conj().T for n in range(1, n_exc + 1)]
                C = [self.__Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
                I = [np.eye(Qcs.__Dim[n]) for n in range(1, n_exc + 1)]
                if type(ome_list) not in number_type:
                    return [(abs(1 + sum([comb(n_exc, n) * left_prod(C[:n]) @ right_prod(H0[:n], B[:n], I[:n], ome, zp) for n in range(1, n_exc + 1)])) ** 2 / \
                             abs(1 + 1j * C[0] @ inv(H0[0] - (ome + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0] for ome in ome_list]
                else:
                    return (abs(1 + sum([comb(n_exc, n) * left_prod(C[:n]) @ right_prod(H0[:n], B[:n], I[:n], ome_list, zp) for n in range(1, n_exc + 1)])) ** 2 / \
                            abs(1 + 1j * C[0] @ inv(H0[0] - (ome_list + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0]

        elif label == -2:  # multiple-to-one-differnt
            key_in = list(self.Input.keys())
            # key_out = list(self.Output.keys())[0]
            ome_list = list(self.Frequency.values())
            ty_n = ome_list.count(ome_list[0]) == len(ome_list)
            if ty_n == True:
                ome_list = ome_list[0]
                if n_exc == 1:
                    B, C, H0 = sum([self.ratio[key] * self.__Input_Matrix(1)[key].conj().T for key in key_in]), self.__Output_Matrix(1)[key_out], self.__Heff_Matrix(1)
                    k_sum = sum(self.ratio.values())
                    I = np.eye(Qcs.__Dim[n_exc])
                    if type(ome_list) not in number_type:
                        return [(abs(C @ inv(H0 - ome * I) @ B / k_sum) ** 2)[0, 0] for ome in ome_list]
                    else:
                        return (abs(C @ inv(H0 - ome_list * I) @ B / k_sum) ** 2)[0, 0]
                else:
                    B = [sum([self.ratio[key] * self.__Input_Matrix(n)[key].conj().T for key in key_in]) for n in range(1, n_exc + 1)]
                    C = [self.__Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                    H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
                    I = [np.eye(Qcs.__Dim[n]) for n in range(1, n_exc + 1)]
                    C_tot = left_prod(C)
                    if type(ome_list) not in number_type:
                        return [(abs(C_tot @ right_prod(H0, B, I, ome, zp)) ** 2 / abs(C[0] @ inv(H0[0] - (ome + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0] for ome
                                in ome_list]
                    else:
                        return (abs(C_tot @ right_prod(H0, B, I, ome_list, zp)) ** 2 / abs(C[0] @ inv(H0[0] - (ome_list + zp) * I[0]) @ B[0]) ** (2 * n_exc))[0, 0]
            else:
                sum_f = sum_frequencies(ome_list, n_exc)
                H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
                I = [np.eye(Qcs.__Dim[n]) for n in range(1, n_exc + 1)]
                Klist = [[inv(H0[n - 1] - (ome + zp) * I[n - 1]) for ome in sum_f[n - 1]] for n in range(1, n_exc + 1)]
                C = [self.__Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                ary = n_m_ary(n_exc, len(key_in))
                comb_s = [covert_to_decimals(ary_i, len(key_in)) for ary_i in ary]
                if n_exc == 1:
                    B = [self.__Input_Matrix(1)[key].conj().T for k, key in enumerate(key_in)]
                    if type(tlist) not in number_type:
                        T_t = []
                        for t in tlist:
                            N = [self.ratio[key] * exp(-1j * self.Frequency[key] * t) for key in key_in]
                            Blist = [N[k] * B[k] for k in range(len(B))]
                            T_t.append((abs(C[0] @ sum([Klist[0][i] @ Blist[i] for i in range(len(key_in))])) ** 2 / abs(sum(N)) ** 2)[0, 0])
                        return T_t
                    else:
                        N = [self.ratio[key] * exp(-1j * self.Frequency[key] * tlist) for key in key_in]
                        Blist = [N[k] * B[k] for k in range(len(B))]
                        return (abs(C[0] @ sum([Klist[0][i] @ Blist[i] for i in range(len(key_in))])) ** 2 / abs(sum(N)) ** 2)[0, 0]
                else:
                    C_tot = left_prod(C)
                    B = [[self.__Input_Matrix(n)[key].conj().T for key in key_in] for n in range(1, n_exc + 1)]
                    if type(tlist) not in number_type:
                        gn_t = []
                        for t in tlist:
                            N = [self.ratio[key] * exp(-1j * self.Frequency[key] * t) for key in key_in]
                            Blist = [[Bi[k] * N[k] for k in range(len(key_in))] for Bi in B]
                            Gn_0 = 0
                            for j, c in enumerate(comb_s):
                                C_tot_c = C_tot.copy()
                                for k, ci in enumerate(c[::-1]):
                                    C_tot_c = C_tot_c @ Klist[n_exc - k - 1][ci] @ Blist[n_exc - k - 1][ary[j][n_exc - k - 1]]
                                Gn_0 += C_tot_c
                            gn_t.append((abs(Gn_0) ** 2 / abs(C[0] @ sum([Klist[0][i] @ Blist[0][i] for i in range(len(key_in))])) ** (2 * n_exc))[0, 0])
                        return gn_t
                    else:
                        N = [self.ratio[key] * exp(-1j * self.Frequency[key] * tlist) for key in key_in]
                        Blist = [[Bi[k] * N[k] for k in range(len(key_in))] for Bi in B]
                        Gn_0 = 0
                        for c in comb_s:
                            C_tot_c = C_tot.copy()
                            for k, ci in enumerate(c[::-1]):
                                C_tot_c = C_tot_c @ Klist[n_exc - k - 1][ci] @ Blist[n_exc - k - 1][ci]
                            Gn_0 += C_tot_c
                        return (abs(Gn_0) ** 2 / abs(C[0] @ sum([Klist[0][i] @ Blist[0][i] for i in range(len(key_in))])) ** (2 * n_exc))[0, 0]

        # elif label == 2: # multiple-to-one-same
        #
        elif label == -3:  # one-to-multiple-differnt
            key_in = list(self.Input.keys())[0]
            # key_out = [key for key in photons for i in range(photons[key])]
            ome_list = self.Frequency[key_in]
            B = [self.__Input_Matrix(n)[key_in].conj().T for n in range(1, n_exc + 1)]
            C = [self.__Output_Matrix(n)[key_out[n - 1]] for n in range(1, n_exc + 1)]
            H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
            I = [np.eye(Qcs.__Dim[n]) for n in range(1, n_exc + 1)]
            C_tot = left_prod(C)
            if type(ome_list) not in number_type:
                correlation = []
                for ome in ome_list:
                    G = abs(C_tot @ right_prod(H0, B, I, ome, zp)) ** 2
                    K1 = inv(H0[0] - (ome + zp) * I[0]) @ B[0]
                    for key in photons:
                        G /= abs((self.__Output_Matrix(1)[key] @ K1) ** (photons[key])) ** 2
                    correlation.append(G[0, 0])
                return correlation
            else:
                correlation = abs(C_tot @ right_prod(H0, B, I, ome_list, zp)) ** 2
                K1 = inv(H0[0] - (ome_list + zp) * I[0]) @ B[0]
                for key in photons:
                    correlation /= abs((self.__Output_Matrix(1)[key] @ K1) ** (photons[key])) ** 2
                return correlation[0, 0]
        elif label == 3:
            key_in = list(self.Input.keys())[0]
            # key_out = [key for key in photons for i in range(photons[key])]
            ome_list = self.Frequency[key_in]
            B = [self.__Input_Matrix(n)[key_in].conj().T for n in range(1, n_exc + 1)]
            C = [self.__Output_Matrix(n)[key_out[n - 1]] for n in range(1, n_exc + 1)]
            H0 = [self.__Heff_Matrix(n) for n in range(1, n_exc + 1)]
            I = [np.eye(Qcs.__Dim[n]) for n in range(1, n_exc + 1)]
            C_tot = left_prod(C)
            if type(ome_list) not in number_type:
                correlation = []
                for ome in ome_list:
                    # 1 + sum([comb(photons[key_in], n) * left_prod(C[:n]) @ right_prod(H0[:n], B[:n], I[:n], ome, zp) for n in range(1, photons[key_in] + 1)])
                    G = abs(C_tot @ right_prod(H0, B, I, ome, zp)) ** 2
                    K1 = inv(H0[0] - (ome + zp) * I[0]) @ B[0]
                    for key in photons:
                        if key == key_in:
                            G /= abs((1 + 1j * self.__Output_Matrix(1)[key] @ K1) ** (photons[key])) ** 2
                        else:
                            G /= abs((self.__Output_Matrix(1)[key] @ K1) ** (photons[key])) ** 2
                    correlation.append(G[0, 0])
                return correlation
            else:
                correlation = abs(C_tot @ right_prod(H0, B, I, ome_list, zp)) ** 2
                K1 = inv(H0[0] - (ome_list + zp) * I[0]) @ B[0]
                for key in photons:
                    correlation /= abs((self.__Output_Matrix(1)[key] @ K1) ** (photons[key])) ** 2
                return correlation[0, 0]
        # elif label == -4:
        #
        # else:

# N = 16
# theta = np.pi
# delta = 0.4
# d = 1
# phi = 1
# ome0 = 0
#
# gamma = 1
#
# z = [d * (i + delta * np.cos(np.pi * i + theta)) for i in range(1, N + 1)]
#
# para = {'gamma': 2 * gamma, 'z': z, 'k': phi / d}
#
# Elist = gamma / np.linspace(-5.5, -4, 50000) + ome0
#
# Hfree = []
# Hint = []
# for i in range(N):
#     Hfree.append([ome0 - 1j * gamma, ('sp', i), ('sm', i)])
# for i in range(N):
#     for j in range(i + 1, N):
#         Hint.append([-1j * np.exp(1j * phi / d * (z[j] - z[i])) * gamma, ('sp', i), ('sm', j)])
#         Hint.append([-1j * np.exp(1j * phi / d * (z[j] - z[i])) * gamma, ('sp', j), ('sm', i)])
# Heff = Hfree + Hint
#
# B1 = []
# B2 = []
# for i in range(N):
#     B1.append([np.sqrt(gamma) * np.exp(-1j * phi / d * z[i]), ('sm', i)])
#     B2.append([np.sqrt(gamma) * np.exp(1j * phi / d * z[i]), ('sm', i)])
#
# Input = Qcs.Input_channel('b1', B1, Elist)
#
# Output1 = Qcs.Output_channel('b1', B1)
#
# Result = Qcs(Heff, Input, Output)
#
# ti = time.time()
#
# R = Result.calculate_quantity('b1')
# g2 = Result.calculate_quantity('b1b1')
#
# tf = time.time()
# print(tf - ti)
#
# Delta = np.linspace(-5.5, -4, 50000)

# plt.plot(Delta, R)
# plt.plot(Delta, g2)
# plt.ylim([0, 3])
# plt.show()

# N = 20
#
# a = 0.25
# z = [i + j for i in range(int(N / 2)) for j in [0, a]]
# phi = 2 * np.pi
#
# gamma = 1
#
# Hcoh = []
# for i in range(N):
#     for j in range(i + 1, N):
#         Hcoh.append([np.sin(phi * (z[j] - z[i])) * gamma, ('sp', i), ('sm', j)])
#         Hcoh.append([np.sin(phi * (z[j] - z[i])) * gamma, ('sp', j), ('sm', i)])
#
# Result = Qcs(Hcoh, [], [])
#
# Result.excitation_number()
# Result.basis(2)
# Result.prestore_HeffList(2)
#
# H = Result.Heff_Matrix(2)
# lm, _ = np.linalg.eigh(H)

# N = 50
# J = 1
# kap = 1
#
# s = 200
#
# Hfree = []
# Hint = []
#
# for i in range(N):
#     if i == 0:
#         Hfree.append([ome[i] - 1j * kap / 2, ("ad", i), ("a", i)])
#     elif i == N - 1:
#         Hfree.append([ome[i] - 1j * kap / 2, ("ad", i), ("a", i)])
#     else:
#         Hfree.append([ome[i], ("ad", i), ("a", i)])
#
# for i in range(N - 1):
#     Hint.append([J, ("ad", i), ("a", i + 1)])
#     Hint.append([J, ("a", i), ("ad", i + 1)])
#
# Heff = Hint + Hfree
#
# Elist = np.linspace(-4, 4, 10000)
#
# B1 = [[np.sqrt(kap), ("a", 0)]]
# BN = [[np.sqrt(kap), ("a", N - 1)]]
#
# Input = Qcs.Input_channel('b1', B1, Elist)
#
# Output1 = Qcs.Output_channel('bN', BN)
# Output2 = Qcs.Output_channel('b1', B1)
#
# Result1 = Qcs(Heff, Input, Output1)
# # Result2 = Qcs(Heff, Input, Output2)
#
# T = Result1.calculate_quantity("bN")
# # R = Result2.calculate_quantity("b1")
# # Tlist.append(G2)
# # R = Result.calculate_quantity("b1")
# # print(t)
# plt.plot(Elist, T)
# # plt.plot(Elist, R)
# plt.show()

# Tlist = np.array(Tlist)
#
# TT = np.sum(Tlist, 0) / 200
#
# plt.plot(Elist, TT)
# plt.show()


# N = 8
# theta = np.pi * 0
# delta = 0.4
# d = 1
# phi = 1
# ome0 = 100
# gamma = 0.01
# kappa = 1
# g = 1
# z = [d * (i + delta * np.cos(np.pi * i + theta)) for i in range(1, N + 1)]
#
# para = {'gamma': 2 * gamma, 'z': z, 'k': phi / d}
#
# Elist = gamma / np.linspace(-5.5, -4, 10000) + ome0
#
# Hfree = []
# Hint = []
# Hfree.append([ome0 - 1j * kappa, ('ad', 0), ('a', 0)])
# for i in range(N):
#     Hfree.append([ome0 - 1j * gamma, ('sp', i), ('sm', i)])
# for i in range(N):
#     Hint.append([g * np.exp(-1j * phi / d * z[i]), ('ad', 0), ('sm', i)])
#     Hint.append([g * np.exp(1j * phi / d * z[i]), ('a', 0), ('sp', i)])
#     for j in range(i + 1, N):
#         Hint.append([-1j * np.exp(1j * phi / d * (z[j] - z[i])) * gamma, ('sp', i), ('sm', j)])
#         Hint.append([-1j * np.exp(1j * phi / d * (z[j] - z[i])) * gamma, ('sp', j), ('sm', i)])
# Heff = Hfree + Hint
#
# # B1 = [[np.sqrt(gamma) * np.exp(-1j * phi / d * z[i]), ("sm", i)] for i in range(N)]
# # BN = [[np.sqrt(gamma) * np.exp(1j * phi / d * z[i]), ("sm", i)] for i in range(N)]
# B1 = [[np.sqrt(kappa), ("a", 0)]]
#
# Input = Qcs.Input_channel('br', B1, Elist)
# Output = Qcs.Output_channel('bl', B1)
#
# Result = Qcs(Heff, Input, Output)
#
# T = Result.calculate_quantity("bl")
#
# Delta = np.linspace(-5.5, -4, 10000)
# plt.plot(Delta, T)
# plt.show()


# N = 8
#
# k = 2 * np.pi
# d = 1
# z = [d * i for i in range(N)]
#
# ome0 = 0
# gamma = 1
# kap_wg = 100
# kap_sc = 300
# kappa = kap_wg + kap_sc
# g = 40
#
# Emax = 1 / (2 * np.tan(np.pi / N))
#
# Elist = np.linspace(-0.6, 0.6, 5000)
#
# Hfree = []
# Hint = []
# Hfree.append([ome0 - 1j * kappa / 2, ('ad', 0), ('a', 0)])
# for i in range(N):
#     Hfree.append([ome0 - 1j * gamma / 2, ('sp', i), ('sm', i)])
# for i in range(N):
#     Hint.append([g * np.exp(-1j * k * z[i]), ('ad', 0), ('sm', i)])
#     Hint.append([g * np.exp(1j * k * z[i]), ('a', 0), ('sp', i)])
#     for j in range(i + 1, N):
#         Hint.append([-1j * np.exp(1j * k * (z[j] - z[i])) * gamma, ('sp', i), ('sm', j)])
#         # Hint.append([-1j * np.exp(1j * k * (z[j] - z[i])) * (gamma - 0.4) / 2, ('sp', j), ('sm', i)])
# Heff = Hfree + Hint
#
# # B1 = [[np.sqrt(gamma) * np.exp(-1j * phi / d * z[i]), ("sm", i)] for i in range(N)]
# # BN = [[np.sqrt(gamma) * np.exp(1j * phi / d * z[i]), ("sm", i)] for i in range(N)]
# B1 = [np.sqrt(kap_wg), ("a", 0)]
#
# Input = Qcs.Input_channel('br', B1, Elist)
# Output = Qcs.Output_channel('br', B1)
#
# Result = Qcs(Heff, Input, Output)
#
# T = Result.calculate_quantity("br")
# g2 = Result.calculate_quantity("brbr")
# #
# plt.semilogy(Elist, g2)
# plt.plot(Elist, T)
# plt.show()
