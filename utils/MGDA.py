import torch


class MinNormSolver:
    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        if v1v2 >= v1v1:
            gamma = 0.999
            cost = v1v1
            return gamma, cost

        if v1v2 >= v2v2:
            gamma = 0.001
            cost = v2v2
            return gamma, cost

        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)

        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs):
        n = len(vecs)

        dps = {}
        sol = None

        d_min = 1e8
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in dps:
                    dps[(i, j)] = sum(torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu() for k in range(len(vecs[i])))
                    dps[(j, i)] = dps[(i, j)]

                if (i, i) not in dps:
                    dps[(i, i)] = sum(torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu() for k in range(len(vecs[i])))

                if (j, j) not in dps:
                    dps[(j, j)] = sum(torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu() for k in range(len(vecs[i])))

                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < d_min:
                    d_min = d
                    sol = [(i, j), c, d]

        return sol

    @staticmethod
    def find_min_norm_element(vecs):
        n = len(vecs)  # always 2

        sol = MinNormSolver._min_norm_2d(vecs)

        sol_vec = torch.zeros(n)
        sol_vec[sol[0][0]] = sol[1]
        sol_vec[sol[0][1]] = 1 - sol[1]

        return sol_vec
