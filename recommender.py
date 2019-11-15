#/software/anaconda/5.1.0/bin/python

#!/opt/local/bin/python


import argparse
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv
from sklearn.neighbors import NearestNeighbors
import scipy.linalg
import bottleneck as bn
import os



###python recommender.py --help
parser = argparse.ArgumentParser(description='Recommender of custormer product system',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--work_dir', default='.',
                    help='The work directory contains the input file and store output file')
parser.add_argument('--train', default='train_debug.npz',
                    help='The matrix of training data(*.npz)')

parser.add_argument('--test', default='test_debug.npz',
                    help='The matrix of test data(*.npz)')

parser.add_argument('--k', default=5,
                    help='The k nearest neighor')

parser.add_argument('--ouput', default='Prediction.txt',
                    help='The output file name')


args = parser.parse_args()



class Recommender(object):
    """ 
    Read the matrix
    Calculate the similarity matrix 
    """
    def __init__(self, args):
        
        os.chdir(args.work_dir)
        self.name=args.train
        self.Bmat=sparse.load_npz(self.name)
        self.custn=self.Bmat.shape[0]
        self.prodn=self.Bmat.shape[1]
        Tbmat=self.Bmat.transpose()
        self.mat=sparse.bmat([[None,self.Bmat],[Tbmat, None]])
        np.set_printoptions(threshold=np.nan)

############################################################################## 
#### Soring function 
##############################################################################          

########MFA method, return a similarity matrix
####turn similarity matrix
        
    def MFA(self):
        d_cust=self.Bmat.sum(axis=1)
        d_prod=self.Bmat.sum(axis=0)
        
        d_cust=np.squeeze(np.array(d_cust))
        print(d_cust.shape)
        d_prod=np.squeeze(np.asarray(d_prod))
        print(d_prod.shape)
        diag=np.concatenate((d_cust,d_prod),axis=0)
        print(diag.shape)
        Diag=sparse.diags(diag)
        L=Diag-self.mat
        I=sparse.eye(L.shape[0])
        M=L+I
        np.set_printoptions(threshold=np.nan)
        self.T=inv(M)
        Direct=self.T[:self.custn,self.custn:]
#         sparse.save_npz("mfa.npz", self.T)
#         print(self.T.toarray())
        
        return [self.T,Direct]
    
########MFA method, return a similarity matrix      
    def Lplus(self):

        mat = self.mat
        tmat = mat.transpose()
        adj = scipy.sparse.bmat([[None, mat],[tmat, None]]).toarray()
#         np.save('adj.npy', adj)

        d_cust = mat.sum(axis=1)
        d_prod = mat.sum(axis=0)
        d_cust = np.squeeze(np.asarray(d_cust))
        d_prod = np.squeeze(np.asarray(d_prod))
        diag = np.concatenate((d_cust, d_prod), axis=0)
        diag = np.diag(diag)
#         np.save(outprefix+'diag.npy', diag)

        L = diag - adj
#         np.save(outprefix+'L.npy', L)
        #Lplus = scipy.linalg.pinv(L)
        Lplus = sparse.csr_matrix(scipy.linalg.pinv2(L))
        Diret=Lplus[:self.custn,self.custn:]
        #print(np.allclose(L, np.dot(L, np.dot(Lplus, L))))
        #print(np.allclose(Lplus, np.dot(Lplus, np.dot(L, Lplus))))
#         np.save(outprefix+'lplus.npy',Lplus)
        return [Lplus,Diret]

  ####knn method################


    def KNN(self):
        k=args.k
        neigh = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='minkowski')
        neigh.fit(self.mat)
        dist,index=neigh.kneighbors() 
        neigh.fit(self.mat.transpose())
        dist2,index2=neigh.kneighbors() 
        
        return [[dist,index],[dist2,index2]]
    
    
    
    ###version==0 stand for k nearest for customer 
    ###return [customer_customer_sim, customer_customer_index,prod_prod_sim, prod_prod_ind ]
        
    def Knearest(self, smat):
        k=args.k
#         smat=sparse.csr_matrix(smat)
        print(type(smat))
        res=[]
#         if smatn.find(".npy")!=-1:
#             print("npy")
#             smat=np.load(smatn)
#             smat=sparse.csr_matrix(smat)
#         else:      
#             smat=sparse.load_npz(smatn)
#         print(smat.toarray())
        print(self.custn)
        
        for j in range(2):
            if j==0:
#                 outp_sim=sim_outp+"_cc.npy"
#                 outp_index=index_outp+"_cc.npy"
                sim_mat=smat[:self.custn,:self.custn]
            else:
#                 outp_sim=sim_outp+"_pp.npy"
#                 outp_index=index_outp+"_pp.npy"
                sim_mat=smat[self.custn:,self.custn:]
                    
            print(sim_mat.shape)
    #         print(cust_cust.toarray())
            res_sim=[]
            res_index=[]

            for i in range(sim_mat.shape[0]):
                x=sim_mat[i].toarray()           
                size=x.shape[1]
#                 print(x[0])
                index=bn.argpartition(x[0], size-k)[-k:]
                res_index.append(np.array(index))
    #             print(np.array(index))
                data=x[0][index]
                res_sim.append(np.array(data))
#                 print(data)
            print(len(res_sim))
            print(len(res_sim[0]))
            res.append([res_sim,res_index])
#             np.save(outp_sim, res_sim)
#             np.save(outp_index, res_index)
        return res
  
############################################################################## 
#### Evaluation methods 
##############################################################################  
        
    def Direct(self, smat,output):
        smat=sparse.load_npz(smat)
        direct=smat[:self.custn,self.custn:]
        sparse.save_npz(output, direct)
        
    
    def Userbase(self, k_sim, k_index):
              
#         sim_arry=np.load(k_sim)
        sim_arry=k_sim
#         print(sim_arry)
        index_arry=k_index
        size=len(sim_arry)
#         print(index_arry)
        print(size)
        res=sparse.csr_matrix(np.zeros((1,self.prodn)))
    
              
        for i in range(size):
            sim=sim_arry[i]  
            index=index_arry[i]
            pmat=self.Bmat[index]
#             print(pmat.toarray())            
            numerator=sim*pmat
#             print(numerator.shape)
            denumer=np.sum(sim)
#             print(denumer)
            pred=numerator/denumer
            res=sparse.vstack((res,sparse.csr_matrix(pred)))
#             print(pred)
        res=res[1:]
#         print(res.toarray())
#         sparse.save_npz(output, res)
        return res
    
    
    def Prodbase(self, k_sim, k_index):
        sim_arry=k_sim
        index_arry=k_index
        size=len(sim_arry)
        print(size) 
        res=sparse.csr_matrix(np.zeros((1,self.custn)))
        print(res.shape)
        
        for i in range(size):
            sim=sim_arry[i]  
            index=index_arry[i]
            pmat=self.Bmat[:,index]
#             print(pmat.toarray())
#             print(sim.shape)
#             print(pmat.shape)
            numerator=pmat*sim.T
#             print(sim.T.shape)
#             print(numerator.shape)
            denumer=np.sum(sim)
            pred=numerator/denumer
#             print(pred.shape)
            res=sparse.vstack((res,sparse.csr_matrix(pred)))
#             print(res.shape)
        
        res=sparse.csr_matrix(res)
        
        res=res[1:]
#         print(res.T.toarray())
#         sparse.save_npz(output, res.T)      
        return res

    
############################################################################## 
#### prediction method  
############################################################################## 
    
    # Degree of agreement: 50% -> random ranking; 100 -> ideal ranking
    def degree_of_agreement(test, train, rank):
        num_person = test.get_shape()[0]
        agree = 0
        for k in range(num_person):
            test_k = test.getrow(k).toarray()[0]
            train_k = train.getrow(k).toarray()[0]
            rank_k = rank[k].argsort()
            M1 = test_k.nonzero()[0] # product in test set
            M2 = ((test_k.astype(int) | train_k.astype(int)) == 0).nonzero()[0] # product not in test or training set
            if(M2.size == 0):
                agree += 1 # if all product has been purchased by the person, inidividual degree of agrement = 1
            else:
                correctpairs = 0
                allpairs = 0
                for i in M1:
                    for j in M2:
                        if(numpy.where(rank_k == i)[0][0] > numpy.where(rank_k == j)[0][0]):
                            correctpairs += 1
                        allpairs += 1
                agree += correctpairs / allpairs # individual degree of agreement - pairs ranked in the correct order/total number of pairs
        agree /= num_person # global degree of agreement - average individual degree of agreements over number of person

        return agree

    # Percentile: ~ 0% -> ideal ranking
    def percentile(test, rank):
        num_person = test.get_shape()[0]
        num_product = rank.shape[1]
        percentile = 0
        for k in range(num_person):
            test_k = test.getrow(k).toarray()[0].nonzero()[0]
            rank_k = rank[k].argsort()
            test_product_rank = []
            for product in test_k:
                test_product_rank.append(numpy.where(rank_k == product)[0][0])
            percentile += (num_product - numpy.median(test_product_rank))/num_product
        percentile /= num_person

        return percentile

    # Recall: 100% -> ideal ranking
    def recall(test, rank, n):
        num_person = test.get_shape()[0]
        limit = rank.shape[1] - 1 - n
        recall = 0
        for k in range(num_person):
            test_k = test.getrow(k).toarray()[0].nonzero()[0]
                    #indices = numpy.nonzero(rank[k])[0]
            #rank_k = indices[numpy.argsort(rank[k][indices])]
            rank_k = rank[k].argsort()
            correct = 0
            for product in test_k:
                if( numpy.where(rank_k == product)[0][0] > limit):
                    correct += 1
            recall += correct/test_k.size
        recall /= num_person

        return recall
    

        
           
             

if __name__=="__main__":
    
    #dir="/Users/sisi/Google_Drive/computer_science/class/network_8480/project"
#     for i in range(5):
#     for i in range(5,10):
#         trainname="train_all_"+str(i+1)+".npz"
#         output="mfa_all_"+str(i+1)+".npz"
#         mfa=MFA_Pred(dir, trainname)
#         mfa.Sim(output)
#         print("done with "+str(i+1))

#     mfa=Recommender(dir, "train_all.npz")
#     mfa.Knearest("mfa_all.npz", 100, "k_sim", "k_index")

#     mfa.Userbase("./result/dist1.npy", "./result/ind1.npy", "userbase_knn1.npz" )
#     mfa.Moviebase("k_sim_pp.npy", "k_index_pp.npy", "moviebase.npz" )
    
#     for i in range(10):
#         user_sim="./knn_mm/dist"+str(i+1)+".npy"
#         user_ind="./knn_mm/ind"+str(i+1)+".npy"
# #         user_output="./result/userbase_knn"+str(i+1)+".npz"
#         movie_output="./knn_mm/moviebase_knn"+str(i+1)+".npz"
# #         mfa.Userbase(user_sim, user_ind, user_output)
#         mfa.Moviebase(user_sim, user_ind, movie_output)
#         print("done with "+str(i+1))

#     movie=np.load("/scratch1/lisiw/network/lplus/8-lplus.npy")
#     print(movie.shape)

#     for i in range(9,10):
# #         iput="./mfa/mfa_all_"
#         iput="./lplus/"+str(i+1)+"-lplus"
# #         knn=iput+str(i+1)+".npz"
#         knn=iput+".npy"
#         print(knn)
#         sim=iput+"sim_"+str(i+1)
#         ind=iput+"ind_"+str(i+1)
#         mfa.Knearest(knn, 100, sim, ind)
#         user_sim=sim+"_cc.npy"
#         user_ind=ind+"_cc.npy"        
#         user_output=iput+"user_"+str(i+1)+".npz"
#         mfa.Userbase(user_sim, user_ind, user_output)
#         movie_sim=sim+"_pp.npy"
#         movie_ind=ind+"_pp.npy"        
#         movie_output=iput+"movie_"+str(i+1)+".npz"
        
#         mfa.Moviebase(movie_sim, movie_ind, movie_output)
# #         direct=iput+"direct"+str(i+1)+".npz"
# #         mfa.Direct(knn,direct)
        
#         print("done with "+str(i+1))
        

    

############debug matrix############
#     bmat=sparse.load_npz("train_debug.npz")
#     bmat_1=bmat[:4, :6]
#     print(bmat_1.toarray())
#     sparse.save_npz("poster.npz", bmat_1) 
#     mfa=Recommender(dir, "poster.npz")
#     mfa.Sim("mfa_poster.npz")
#     mfa.Knearest("mfa_poster.npz", 3, "sim_poster", "ind_poster")
#     mfa.Userbase("sim_poster_cc.npy", "ind_poster_cc.npy", "user_poster.npz")
#     mfa.Moviebase("sim_poster_pp.npy", "ind_poster_pp.npy", "movie_poster.npz")
    
    
    
############debug matrix############    
    
    recomm=Recommender(args)
    mfa=recomm.MFA()
    lplus=recomm.Lplus()
    knn=recomm.KNN()
    mfa_knn=recomm.Knearest(mfa[0])
    lplus_knn=recomm.Knearest(lplus[0])
    print(mfa_knn)
    print(lplus_knn)
    mfa_cc=recomm.Userbase(mfa_knn[0][0],mfa_knn[0][1])
    mfa_pp=recomm.Prodbase(mfa_knn[1][0],mfa_knn[1][1])
    mfa_dir=mfa[1]
    
    lplus_cc=recomm.Userbase(lplus_knn[0][0],lplus_knn[0][1])
    lplus_pp=recomm.Prodbase(lplus_knn[1][0],lplus_knn[1][1])
    lplus_dir=lplus[1]
    print(lplus_knn.toarray())


#     mfa.Knearest("mfa_debug.npz", 5, "sim_debug", "ind_debug")
    
#     print(bmat.toarray())
#     test=sparse.load_npz("test_debug.npz")
#     print(test.toarray())
#     train=sparse.load_npz("train_debug.npz")
#     print(train.toarray())
# #     mfa.Userbase("sim_debug_cc.npy", "ind_debug_cc.npy", "user_debug.npz")
#     mfa.Moviebase("sim_debug_pp.npy", "ind_debug_pp.npy", "movie_debug.npz")

############debug matrix############
                   
   
    print("good")
    
    
    
    