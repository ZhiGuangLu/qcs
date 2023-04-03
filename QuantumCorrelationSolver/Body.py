from scipy.sparse import csr_matrix, lil_matrix
from scipy.special import comb
from numpy.linalg import inv, eig
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
            m3[i, j] = coff[j]
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


def right_prod(H, B, I, ome):
    n_exc = len(B)
    K = [1j * inv(H[n] - (n + 1) * ome * I[n]) for n in range(n_exc)]
    B_n = K[-1] @ B[-1]
    for i in range(n_exc - 2, -1, -1):
        B_n = B_n @ (K[i] @ B[i])
    return B_n


class Qcs:
    BasisList = dict()
    HeffList = dict()
    InOutList = dict()

    def __init__(self, Heff, Input, Output, ratio=None, tlist=0):
        self.Heff_c = [Hi[0] for Hi in Heff]
        self.Heff_v = [Hi[1:] for Hi in Heff]
        self.Input = {}
        self.Frequency = {}
        temp_in = self.combine_channel(Input)
        for key, values in temp_in.items():
            self.Input[key] = values[0]
            self.Frequency[key] = values[1]
        if len(temp_in) > 1 and ratio == None:
            self.ratio = [1] * len(temp_in)
        else:
            self.ratio = ratio
        self.Output = self.combine_channel(Output)

    def combine_channel(self, channels):
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
            frequency_c = list(frequency)
        Input[channel_name] = [mode, frequency_c]
        return Input

    def Output_channel(channel_name, mode):
        Output = dict()
        if type(mode[0]) in number_type:
            mode = [mode]
        Output[channel_name] = mode
        return Output

    def excitation_number(self):
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

    def basis(self, n_exc):
        max_dims = [n_exc if O[0] == "a" else 1 for O in self.Ope_ani]
        Qcs.BasisList[n_exc] = create_basis(n_exc, max_dims, self.k)

    def prestore_HeffList(self, n_exc):
        Heff_nexc = dict()
        bas_fix = Qcs.BasisList[n_exc]
        for Hi in self.Heff_v:
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
                    coff *= np.sqrt(bas_cor[:, loc])
                else:
                    loc = self.Ope_cre.index(H)
                    coff *= np.sqrt(bas_cor[:, loc])
                    bas_cor[:, loc] -= 1
                Heff_nexc[tuple(Hi)] = basis_dot(bas_fix, bas_cor, coff)
        Qcs.HeffList[n_exc] = Heff_nexc

    def prestore_InOutList(self, n_exc):
        InOut = dict()
        In_Opes = [x[1] for InOpe in self.Input.values() for x in InOpe]
        Out_Opes = [x[1] for OutOpe in self.Output.values() for x in OutOpe]
        bas_1 = Qcs.BasisList[n_exc - 1]
        bas_2 = Qcs.BasisList[n_exc]
        for In in In_Opes:
            coff = np.ones((len(bas_2),))
            bas_cor = np.array(bas_2)
            loc = self.Ope_ani.index(In)
            coff *= np.sqrt(bas_cor[:, loc])
            bas_cor[:, loc] -= 1
            InOut[In] = basis_dot(bas_1, bas_cor, coff)
        for Out in Out_Opes:
            coff = np.ones((len(bas_2),))
            bas_cor = np.array(bas_2)
            if Out not in InOut:
                loc = self.Ope_ani.index(Out)
                coff *= np.sqrt(bas_cor[:, loc])
                bas_cor[:, loc] -= 1
                InOut[Out] = basis_dot(bas_1, bas_cor, coff)
        Qcs.InOutList[n_exc] = InOut

    def Heff_Matrix(self, n_exc):
        Heff_List = np.array(list(Qcs.HeffList[n_exc].values())) * self.Heff_c
        Heff_m = sum_sparse(list(Heff_List))
        return Heff_m

    def Input_Matrix(self, n_exc):
        Int_m = {}
        for key, value in self.Input.items():
            Int_m[key] = sum_sparse([Qcs.InOutList[n_exc][x[1]] * x[0] for x in value])
        return Int_m

    def Output_Matrix(self, n_exc):
        Out_m = {}
        for key, value in self.Output.items():
            Out_m[key] = sum_sparse([Qcs.InOutList[n_exc][x[1]] * x[0] for x in value])
        return Out_m

    def classification(self, photons):
        In_list, Out_list = set(self.Input.keys()), set(photons.keys())
        cover_part = In_list & Out_list
        if len(In_list) == 1 and len(Out_list) == 1:
            if cover_part:
                return 1  # "one-to-one-same"
            else:
                return -1  # "one-to-one-differnt"
        elif len(In_list) > 1 and len(Out_list) == 1:
            if cover_part:
                return 2  # "multiple-to-one-same"
            else:
                return -2  # "multiple-to-one-differnt"
        elif len(In_list) == 1 and len(Out_list) > 1:
            if cover_part:
                return 3  # "one-to-multiple-same"
            else:
                return -3  # "one-to-multiple-differnt"
        else:
            if cover_part:
                return 4  # "multiple-to-multiple-same"
            else:
                return -4  # "multiple-to-multiple-differnt"

    def calculate_quantity(self, Quantity, time=None):
        photons = {}
        for mode in self.Output.keys():
            photons[mode] = Quantity.count(mode)

        self.excitation_number()
        n_exc = sum(photons.values())
        for n in range(0, n_exc + 1):
            self.basis(n)
            if n != 0:
                self.prestore_InOutList(n)
                self.prestore_HeffList(n)
        label = self.classification(photons)
        if label == -1:
            key_in = list(self.Input.keys())[0]
            key_out = list(self.Output.keys())[0]
            ome_list = self.Frequency[key_in]
            if n_exc == 1:
                B, C, H0 = self.Input_Matrix(1)[key_in].conj().T, self.Output_Matrix(1)[key_out], self.Heff_Matrix(1)
                dim = H0.shape[0]
                I = np.eye(dim)
                if type(ome_list) not in number_type:
                    return [(np.abs(C @ inv(H0 - ome * I) @ B) ** 2)[0, 0] for ome in ome_list]
                else:
                    return (np.abs(C @ inv(H0 - ome_list * I) @ B) ** 2)[0, 0]
            else:
                B = [self.Input_Matrix(n)[key_in].conj().T for n in range(1, n_exc + 1)]
                C = [self.Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                H0 = [self.Heff_Matrix(n) for n in range(1, n_exc + 1)]
                I = [np.eye(H0[i].shape[0]) for i in range(len(H0))]
                C_tot = left_prod(C)
                if type(ome_list) not in number_type:
                    return [(np.abs(C_tot @ right_prod(H0, B, I, ome)) ** 2 / np.abs(C[0] @ inv(H0[0] - ome * I[0]) @ B[0]) ** (2 * n_exc))[0, 0] for ome in ome_list]
                else:
                    return (np.abs(C_tot @ right_prod(H0, B, I, ome_list)) ** 2 / np.abs(C[0] @ inv(H0[0] - ome_list * I[0]) @ B[0]) ** (2 * n_exc))[0, 0]

        elif label == 1:
            key_in = list(self.Input.keys())[0]
            key_out = list(self.Output.keys())[0]
            ome_list = self.Frequency[key_in]
            if n_exc == 1:
                B, C, H0 = self.Input_Matrix(1)[key_in].conj().T, self.Output_Matrix(1)[key_out], self.Heff_Matrix(1)
                dim = H0.shape[0]
                I = np.eye(dim)
                if type(ome_list) not in number_type:
                    return [(np.abs(1 + 1j * C @ inv(H0 - ome * I) @ B) ** 2)[0, 0] for ome in ome_list]
                else:
                    return (np.abs(1 + 1j * C @ inv(H0 - ome_list * I) @ B) ** 2)[0, 0]
            else:
                B = [self.Input_Matrix(n)[key_in].conj().T for n in range(1, n_exc + 1)]
                C = [self.Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                H0 = [self.Heff_Matrix(n) for n in range(1, n_exc + 1)]
                I = [np.eye(H0[i].shape[0]) for i in range(len(H0))]
                if type(ome_list) not in number_type:
                    return [(np.abs(1 + sum([comb(n_exc, n) * left_prod(C[:n]) @ right_prod(H0[:n], B[:n], I[:n], ome) for n in range(1, n_exc + 1)])) ** 2 / \
                             np.abs(1 + 1j * C[0] @ inv(H0[0] - ome * I[0]) @ B[0]) ** (2 * n_exc))[0, 0] for ome in ome_list]
                else:
                    return (np.abs(1 + sum([comb(n_exc, n) * left_prod(C[:n]) @ right_prod(H0[:n], B[:n], I[:n], ome_list) for n in range(1, n_exc + 1)])) ** 2 / \
                            np.abs(1 + 1j * C[0] @ inv(H0[0] - ome_list * I[0]) @ B[0]) ** (2 * n_exc))[0, 0]
        elif label == -2:
            key_in = list(self.Input.keys())
            key_out = list(self.Output.keys())[0]
            ome_list = list(self.Frequency.values())
            ty_n = ome_list.count(ome_list[0]) == len(ome_list)
            if ty_n == True:
                ome_list = ome_list[0]
                if n_exc == 1:
                    Blist, C, H0 = [self.Input_Matrix(1)[key].conj().T for key in key_in], self.Output_Matrix(1)[key_out], self.Heff_Matrix(1)
                    B = sum(map(lambda k, Bi: k * Bi, self.ratio, Blist))
                    k_sum = sum(self.ratio)
                    dim = H0.shape[0]
                    I = np.eye(dim)
                    if type(ome_list) not in number_type:
                        return [(np.abs(C @ inv(H0 - ome * I) @ B / k_sum) ** 2)[0, 0] for ome in ome_list]
                    else:
                        return (np.abs(C @ inv(H0 - ome_list * I) @ B / k_sum) ** 2)[0, 0]
                else:
                    Blist = [[self.Input_Matrix(n)[key].conj().T for key in key_in] for n in range(1, n_exc + 1)]
                    B = [sum(map(lambda k, Bi: k * Bi, self.ratio, Blist_i)) for Blist_i in Blist]
                    C = [self.Output_Matrix(n)[key_out] for n in range(1, n_exc + 1)]
                    H0 = [self.Heff_Matrix(n) for n in range(1, n_exc + 1)]
                    I = [np.eye(H0[i].shape[0]) for i in range(len(H0))]
                    C_tot = left_prod(C)
                    if type(ome_list) not in number_type:
                        return [(np.abs(C_tot @ right_prod(H0, B, I, ome)) ** 2 / np.abs(C[0] @ inv(H0[0] - ome * I[0]) @ B[0]) ** (2 * n_exc))[0, 0] for ome in ome_list]
                    else:
                        return (np.abs(C_tot @ right_prod(H0, B, I, ome_list)) ** 2 / np.abs(C[0] @ inv(H0[0] - ome_list * I[0]) @ B[0]) ** (2 * n_exc))[0, 0]
            # else:
            # if n_exc == 1:
            #
            # else:
        # elif label == 2:
        #
        elif label == -3:
            key_in = list(self.Input.keys())[0]
            key_out = [key for key in photons for i in range(photons[key])]
            ome_list = self.Frequency[key_in]
            B = [self.Input_Matrix(n)[key_in].conj().T for n in range(1, n_exc + 1)]
            C = [self.Output_Matrix(n)[key_out[n - 1]] for n in range(1, n_exc + 1)]
            H0 = [self.Heff_Matrix(n) for n in range(1, n_exc + 1)]
            I = [np.eye(H0[i].shape[0]) for i in range(len(H0))]
            C_tot = left_prod(C)
            if type(ome_list) not in number_type:
                correlation = []
                for ome in ome_list:
                    G = np.abs(C_tot @ right_prod(H0, B, I, ome)) ** 2
                    K1 = inv(H0[0] - ome * I[0]) @ B[0]
                    for key in photons:
                        G /= np.abs((self.Output_Matrix(1)[key] @ K1) ** (photons[key])) ** 2
                    correlation.append(G[0, 0])
                return correlation
            else:
                correlation = np.abs(C_tot @ right_prod(H0, B, I, ome_list)) ** 2
                K1 = inv(H0[0] - ome_list * I[0]) @ B[0]
                for key in photons:
                    correlation /= np.abs((self.Output_Matrix(1)[key] @ K1) ** (photons[key])) ** 2
                return correlation[0, 0]
        elif label == 3:
            key_in = list(self.Input.keys())[0]
            key_out = [key for key in photons for i in range(photons[key])]
            ome_list = self.Frequency[key_in]
            B = [self.Input_Matrix(n)[key_in].conj().T for n in range(1, n_exc + 1)]
            C = [self.Output_Matrix(n)[key_out[n - 1]] for n in range(1, n_exc + 1)]
            H0 = [self.Heff_Matrix(n) for n in range(1, n_exc + 1)]
            I = [np.eye(H0[i].shape[0]) for i in range(len(H0))]
            C_tot = left_prod(C)
            if type(ome_list) not in number_type:
                correlation = []
                for ome in ome_list:

                    # 1 + sum([comb(photons[key_in], n) * left_prod(C[:n]) @ right_prod(H0[:n], B[:n], I[:n], ome) for n in range(1, photons[key_in] + 1)])

                    G = np.abs(C_tot @ right_prod(H0, B, I, ome)) ** 2
                    K1 = inv(H0[0] - ome * I[0]) @ B[0]
                    for key in photons:
                        if key == key_in:
                            G /= np.abs((1 + 1j * self.Output_Matrix(1)[key] @ K1) ** (photons[key])) ** 2
                        else:
                            G /= np.abs((self.Output_Matrix(1)[key] @ K1) ** (photons[key])) ** 2
                    correlation.append(G[0, 0])
                return correlation
            else:
                correlation = np.abs(C_tot @ right_prod(H0, B, I, ome_list)) ** 2
                K1 = inv(H0[0] - ome_list * I[0]) @ B[0]
                for key in photons:
                    correlation /= np.abs((self.Output_Matrix(1)[key] @ K1) ** (photons[key])) ** 2
                return correlation[0, 0]
        # elif label == -4:
        #
        # else:


