import numpy as np
import geatpy as ea
from pickle import load


class moea_NSGA2_DE_templet(ea.MoeaAlgorithm):

    def __init__(self,
                problem,
                population,
                MAXGEN=None,
                MAXTIME=None,
                MAXEVALS=None,
                MAXSIZE=None,
                logTras=None,
                verbose=None,
                outFunc=None,
                drawing=None,
                dirName=None,
                **kwargs):
# 调用父类构造
        super().__init__(problem,
                         population,
                         MAXGEN,
                         MAXTIME,
                         MAXEVALS,
                         MAXSIZE,
                         logTras,
                         verbose,
                         outFunc,
                         drawing,
                         dirName)
        if population.ChromNum !=1:
            raise RuntimeError('传入种群对象必须是单染色体')
        self.name = 'NSGA2_DE'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS #采用ENS_ss进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS #高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS快
        self.selFunc = 'tour' #选择方式，采用锦标赛选择
        if population.Encoding =='RI':
            self.mutOper = ea.Mutde(F=0.5) #生成差分变异算子对象
            self.rec0per = ea.Xovbd(XOVR=0.5, Half_N=True) #生成二项式分布交叉算子对象，这里的XOVR即为DE中的Cr
        else:
            raise RuntimeError('编码方式必须为‘RI')

    def crowdis(self,Popobj,FrontNo=None):
        N, M =np.shape(Popobj)
        A = np.array([1200, 25])
        alpha = 0.9


        if FrontNo is None:
            FrontNo = np.ones(N)

        CrowDis = np.zeros(N)
        Fronts = np.setdiff1d(np.unique(FrontNo), np.inf)
        for f in Fronts:
            Front = np.where(FrontNo==f)[0]
            Fmax = np.max(Popobj[Front, :], axis=0)
            Fmin = np.min(Popobj[Front, :], axis=0)

            for i in range(M):
                rank = np.argsort(Popobj[Front,i])
                CrowDis[Front[rank[0]]]=np.inf
                CrowDis[Front[rank[-1]]]=np.inf

                for j in range(1, len(Front)-1):
                    #b = Popobj[Front[rank[j]]]
                    B = np.array(Popobj[Front[rank[j]]])
                    #print(b)
                    #CrowDis[Front[rank[j]]] += (Popobj[Front[rank[j+1]],i]-Popobj[Front[rank[j-1]], i])/(Fmax[i]-Fmin[i])
                    CrowDis[Front[rank[j]]] += alpha*(np.dot(A,B))/(np.linalg.norm(A) * np.linalg.norm(B))\
                                               +(Popobj[Front[rank[j + 1]], i] - Popobj[Front[rank[j - 1]], i]) / (Fmax[i] - Fmin[i])

        return CrowDis



    def reinsertion(self, population, offspring, NUM):

        #父子两代合并
        population = population + offspring
        #选择个体保留到下一代
        [levels, criLevel] = self.ndSort(population.ObjV, NUM, None, population.CV,
                                         self.problem.maxormins) #对NUM个个体进行非支配分层
        dis = self.crowdis(population.ObjV, levels) #计算拥挤距离
        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort') #计算适应度值
        chooseFlag = ea.selecting('dup', population.FitnV, NUM) #调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
        return population[chooseFlag]

    def run(self, prophetPop=None): #prophetPop为先知种群
        #初始化
        population = self.population
        NIND = population.sizes
        self.initialization() #初始化算法类的一些动态参数
        #准备进化
        population.initChrom() #初始化种群染色体矩阵
        #插入先验知识
        if prophetPop is not None:
            population = (prophetPop+population)[:NIND] #插入先知种群
        self.call_aimFunc(population) #计算种群的目标函数值
        [levels, criLevel] = self.ndSort(population.ObjV, NIND, None, population.CV,
                                         self.problem.maxormins) #对NIND个个体进行非支配分层
        population.FitnV = (1/levels).reshape(-1, 1) #直接根据levels来计算初代个体的适应度
        #开始进化
        while not self.terminated(population):
            #进行差分进化操作
            r0 = ea.selecting(self.selFunc, population.FitnV, NIND) #得到基向量索引
            offspring = population.copy() #存储子代种群
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field, [r0]) #变异
            tempPop = population + offspring #当前种群个体与变异个体进行合并(为的是后面用于重组)
            offspring.Chrom = self.rec0per.do(tempPop.Chrom) #重组
            self.call_aimFunc(offspring) #求进化后个体的目标函数值
            population = self.reinsertion(population, offspring,NIND) #重插入生成新一代种群
        return self.finishing(population) #调用finishing完成后续工作并返回结果



class MyProblem(ea.Problem):

    def __init__(self):
        name = 'MyProblem'
        M = 2
        maxormins = [-1, -1]
        Dim = 25
        varTypes = [0] * Dim
        lb = [100,90,10,210,69,56,2506,1950,581,6,0.02,0.7,273,511495,0.41, 602,825,25,10,1552,0.01,0.6,0.01,0.01,602]
        ub = [121,108,12,228,432,199,15824,16176,3289,32,1.9,3.3,3623,	2346954,5.5,3774,1000,700,180,2709,0.66,3.9,0.24,0.24,3774]


        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub)


    def aimFunc(self, pop):
        Vars = pop.Phen
        Var1_indices = [1, 19, 20, 5, 12, 21, 22, 23, 24, 17, 18]
        Var1 = Vars[:, Var1_indices]  # 获取前11个变量
        Var2 = Vars[:, :19]  # 获取后19个变量

        model1 = load(open('model_pickle/GBR_ts_descriptor_selected.pkl', 'rb'))
        model2 = load(open('model_pickle/GBR_el_descriptor_selected.pkl', 'rb'))

        f1 = model1.predict(Var1).reshape(-1, 1)
        f2 = model2.predict(Var2).reshape(-1, 1)

        pop.ObjV = np.hstack([f1, f2])  # .reshape(400,2)
        pop.ObjV = np.array(pop.ObjV)
        # print("ObjV", pop.ObjV, "type------------  ", type(pop.ObjV))
        # print(pop.ObjV.ndim)
        # ea.moeaplot(ObjV=pop.ObjV) 每一代的散点图都输出



if __name__ == '__main__':
    problem = MyProblem()
    algorithm = moea_NSGA2_DE_templet(
        problem,
        ea.Population(Encoding='RI', NIND=200),
        MAXGEN=500,
        logTras=0)
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=False,
                      drawLog=True,
                      saveFlag=True,
                      dirName='des_pf_803')
    print(res)

