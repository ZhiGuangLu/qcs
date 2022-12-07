import numpy as np
import itertools as it
import sympy as sp
from sympy import Matrix, sqrt, symbols
from numpy.linalg import inv


def decompose(n, m, s, l):
    if n == 0:
        l.append(s)
    else:
        if m > 1:
            decompose(n, m - 1, s, l)
        if m <= n:
            decompose(n - m, m, [m] + s, l)
    return l[::-1]


def findPermutations(basis, element, element0=[]):
    l = len(element) - element.count(0)
    words = element.copy()
    if l == 1:
        basis.append(element0 + words.copy())
        for j in range(len(element) - 1):
            words = [0] * (j + 1) + words[j: -1]
            basis.append(element0 + words.copy())
    else:
        for j in range(len(words) - l + 1):
            findPermutations(basis, element[j + 1:], element0 + element[:j + 1])
            element = [0] * (j + 1) + element[j: -1]


class Heff_Matrix:
    """
    ！！！重要成员属性！！！
    :param n 激发子数, 如 n=2 表示激发子数为2
    :param inf 系统元素的信息, 如 inf = [('cavity', 10), ('s=1/2', 5), ('J=4/2', 0)] 代表10个腔 5个二能级原子 或 0个集体算符J=4/2
    :param H_string 系统的字符串哈密顿量形式
    ！！！重要成员函数！！！


    注：本代码只提供以下哈密顿量的字符输入
    1、符号约束要求：腔 ==> ‘ada’=a^dagger a, 自旋部分 ==> 'sds'=sigma_p sigma_m, 集体算符 ==> 'JdJ'=J_p J_m
    2、输入的H_string形式应为：[H1, H2, Hi, ...]==> H=H1+H2+Hi+...
        例如：H = g * (a_1 * ad_2 + ad_1 * a_2) 对应的字符串形式应为：
             H1 = [g, ('ad', 2), ('ad', 1)], H2 = [g, ('ad', 1), ('a', 2)] ====> H_str = [H1, H2]
             其中 ('ad', 1),('a', 2),('ad', 3),('a',4)表示 a_1*ad_2*a_3*ad_4
    3、输入的信息Inf位置顺序应为：腔，自旋1/2，集体算符；
    4、H_string里不应该出现产生-产生或者湮灭-湮灭项，如:[x, ('ad',i), ('ad',j)]， 如果需要添加Kerr项的话应该将其拆解为:
        ad*ad*a*a=ad*a*ad*a-ad*a
    5、H_string中的每一项里的产生算符都应该放在湮灭算符的前面，如 H1=[g, ('ad',1), ('a', 2)]而不能写成[g, ('a',2), ('ad', 1)]
    """

    def __init__(self, inf, H_string, sol=0, n=1):
        inf_new = self.check_inf(inf)
        self.n = n  # 激发子数为n
        self.inf = inf_new  # 系统元素的信息，如：inf = [('cavity', 10), ('s=1/2', 5), ('J=4/2', 0)] 代表10个腔 5个二能级原子 或 0个集体算符J=4/2
        self.L = [inf_new[i][1] for i in range(3)]
        self.jud = sol  # 是否给出解析解
        # print('Your system order = ', self.targs())
        self.H_str = H_string
        self.basis = [self.creat_basis(i) for i in range(n - 1, n + 1)]
        self.H_mat = self.get_Heff()

    def check_inf(self, inf):
        if len(inf) == 1:
            inf.append(('s=', 0))
            inf.append(('J=', 0))
        if len(inf) == 2:
            if 's' in inf[-1][0]:
                inf.append(('J=', 0))
            else:
                inf.insert(1, ('s=', 0))
        return inf

    def targs(self):
        tag = dict()
        tag[self.inf[0][0]] = list(range(0, self.L[0]))
        tag[self.inf[1][0]] = list(range(sum(self.L[:1]), sum(self.L[:2])))
        tag[self.inf[2][0]] = list(range(sum(self.L[:2]), sum(self.L)))
        return tag

    def creat_basis(self, n_exc):
        basis = []
        if n_exc == 0:
            basis.append([0] * sum(self.L))
        else:
            de = decompose(n_exc, n_exc, [], [])
            for dei in de[::-1]:
                if len(dei) > sum(self.L):
                    de.remove(dei)
            n_list = []
            for ni in de:
                n_list += map(list, set(it.permutations(ni)))
            for ni in n_list:
                findPermutations(basis, ni + [0] * (sum(self.L) - len(ni)))
            del_ind = []
            for id, b in enumerate(basis):
                next = False
                if self.L[1] == 0 and self.L[2] == 0:
                    break
                if self.L[1] != 0:
                    if max(b[self.L[0]:sum(self.L[:2])]) > 1:
                        next = True
                        del_ind.append(id)
                if self.L[2] != 0 and next == False:
                    if max(b[sum(self.L[:2]):sum(self.L)]) > int(self.inf[2][0][2:-2]):
                        del_ind.append(id)
            for ind in del_ind[::-1]:
                basis.pop(ind)
        return basis

    def a(self, ind):
        B0 = self.basis[0]
        B1 = self.basis[1]
        a_matrix = np.zeros((len(B0), len(B1)))
        a_symbol = sp.zeros(len(B0), len(B1))
        for id, Bi in enumerate(B1):
            Bm = Bi.copy()
            num = Bm[ind]
            Bm[ind] += -1
            if num != 0:
                a_matrix[B0.index(Bm), id] = np.sqrt(num)
                a_symbol[B0.index(Bm), id] = sqrt(num)
        return [a_matrix, a_symbol][self.jud]

    def ad(self, ind):
        return self.a(ind).T

    def s(self, ind):
        B0 = self.basis[0]
        B1 = self.basis[1]
        s_matrix = np.zeros((len(B0), len(B1)))
        s_symbol = sp.zeros(len(B0), len(B1))
        for id, Bi in enumerate(B1):
            Bm = Bi.copy()
            num = Bm[ind]
            Bm[ind] += -1
            if num != 0:
                s_matrix[B0.index(Bm), id] = num
                s_symbol[B0.index(Bm), id] = num
        return [s_matrix, s_symbol][self.jud]

    def sd(self, ind):
        return self.s(ind).T

    def J(self, ind):
        Dim = int(self.inf[2][0][2:-2]) + 1
        B0 = self.basis[0]
        B1 = self.basis[1]
        J_matrix = np.zeros((len(B0), len(B1)))
        J_symnol = sp.zeros(len(B0), len(B1))
        for id, Bi in enumerate(B1):
            Bm = Bi.copy()
            num = Bm[ind]
            Bm[ind] += -1
            if num != 0:
                J_matrix[B0.index(Bm), id] = np.sqrt((Dim - num) * num)
                J_symnol[B0.index(Bm), id] = sqrt((Dim - num) * num)
        return [J_matrix, J_symnol][self.jud]

    def Jd(self, ind):
        return self.J(ind).T

    def Jz(self, ind):
        B1 = self.basis[1]
        Jz_matrix = np.zeros((len(B1), len(B1)))
        Jz_symbol = np.zeros(len(B1), len(B1))
        for id, Bi in enumerate(B1):
            Bm = Bi.copy()
            num = Bm[ind]
            if num != 0:
                Jz_matrix[id, id] = num
                Jz_symbol[id, id] = num
        return [Jz_matrix, Jz_symbol][self.jud]

    # def I(self, ind):
    #     B1 = self.basis[1]
    #     return np.eye(len(B1))

    def get_mat(self, H):
        H_list = ['self.' + Hi[0] + '(%s)' % Hi[1] for Hi in H[1:]]
        H_string = str(H_list).replace(',', '@')
        return eval(H_string[1:-1].replace('\'', '')) * H[0]

    def get_Heff(self):
        Heff = self.get_mat(self.H_str[0])
        for Hi in self.H_str[1:]:
            Heff += self.get_mat(Hi)
        return Heff

    def generate_Heff_n(self, n_new):
        self.n = n_new
        self.basis = [self.creat_basis(i) for i in range(self.n - 1, self.n + 1)]
        self.H_mat = self.get_Heff()

    def calculate_g_n(self, n, drive_ath, target_bth, omega_L=[0]):
        """
        :param n: n-阶等时关联函数
        :param drive_ath: 对第a个元素进行弱驱动
        :param target_bth: 计算第b个元素的关联
        :param omega_L: 弱驱动频率，默认为零(相当于做了旋转框架)
        :return: n-阶关联函数值
        """
        Heffn, A_a, A_b = [], [], []
        A_a.append(self.a(drive_ath))
        # A_b.append(self.a(target_bth))
        A_b.append(self.a(target_bth[0]))
        A_b2 = self.a(target_bth[1])
        Heffn.append(self.H_mat)
        for ni in range(2, n + 1):
            self.generate_Heff_n(ni)
            A_a.append(self.a(drive_ath))
            A_b.append(self.a(target_bth[1]))
            Heffn.append(self.H_mat)
        gn_0 = []
        # theta = []
        T = []
        if self.jud == 0:
            for ome in omega_L:
                Left = A_b[0]
                for Ab in A_b[1:]:
                    Left = Left @ Ab
                for i in range(n):
                    Ki = Heffn[-1 - i] - (n - i) * ome * np.eye(Heffn[-1 - i].shape[0])
                    Left = Left @ (np.linalg.inv(Ki) @ A_a[-1 - i].T)
                K0 = Heffn[0] - ome * np.eye(Heffn[0].shape[0])
                # theta.append(np.angle(Left))
                # gn_0.append((np.abs(Left) ** 2 / (np.abs(A_b[0] @ np.linalg.inv(K0) @ A_a[0].T) ** (2 * n)))[0, 0])
                gn_0.append((np.abs(Left) ** 2 / (np.abs(A_b[0] @ np.linalg.inv(K0) @ A_a[0].T * A_b2 @ np.linalg.inv(K0) @ A_a[0].T) ** 2))[0, 0])
                T.append((np.abs(A_b[0] @ np.linalg.inv(K0) @ A_a[0].T) ** 2)[0, 0])
        else:
            Left = A_b[0]
            for Ab in A_b[1:]:
                Left = Left @ Ab
            for i in range(n):
                Ki = sp.simplify(Heffn[-1 - i] - (n - i) * ome * sp.eye(Heffn[-1 - i].shape[0]))
                Left = sp.simplify(Left @ (Ki.inv() @ A_a[-1 - i].T))
            K0 = sp.simplify(Heffn[0] - ome * sp.eye(Heffn[0].shape[0]))
            B = sp.simplify(A_b[0] @ K0.inv() @ A_a[0].T)
            # (Left / (A_b[0] @ K0.inv() @ A_a[0].T) ** n)
            return Left[0], B[0]
            # T.append((np.abs(A_b[0] @ np.linalg.inv(K0) @ A_a[0].T) ** 2 / 4)[0, 0])
        return gn_0, T
        # return gn_0, theta

    def calculate_g_2(self, omega_L=[0]):
        """
        :param omega_L: 弱驱动频率，默认为零(相当于做了旋转框架)
        :return: n-阶关联函数值
        """
        Heffn, A = [], []
        # 激发子数为1
        A.append(np.sqrt(1 / 2) * sum([self.a(ni) for ni in range(self.L[0])]))
        Heffn.append(self.H_mat)
        # 激发子数为2
        self.generate_Heff_n(2)
        A.append(np.sqrt(1 / 2) * sum([self.a(ni) for ni in range(self.L[0])]))
        Heffn.append(self.H_mat)

        gn_0 = []
        for ome in omega_L:
            L1 = A[0]
            L2 = A[1]
            K2 = Heffn[1] - 2 * ome * np.eye(Heffn[1].shape[0])
            K1 = Heffn[0] - ome * np.eye(Heffn[0].shape[0])
            P2 = 1 + 2 * 1j * (L1 @ inv(K1) @ L1.T) + 1j ** 2 * (L1 @ L2 @ inv(K2) @ L2.T @ inv(K1) @ L1.T)
            P1 = 1 + 1j * (L1 @ inv(K1) @ L1.T)
            gn_0.append(np.abs(P2) ** 2 / (np.abs(P1) ** 4))
        return gn_0
