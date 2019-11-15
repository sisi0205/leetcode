#!/opt/local/bin/python

import numpy as np
import os

class Product(object):
    """"The struct for product,
    Numcst: the number of customers who rated the product
    listcst:List of customer
    """
    def __init__(self, id):
        self.asin=0
        self.id=id
        self.Numcst=0
        self.listcst=set()
    
    def asin(self,asin):
        self.asin=asin
    
    def addrat(self, cstid):
        self.listcst.add(cstid)
        self.Numcst+=1

    def printprod(self):
        for i in self.listcst:
            print(str(i)+"   "+str(self.id))

class Customer(object):
    """"The struct for product
    numrat: number of ratings
    listprod: list of product he/she rated"""
    def __init__(self, id):
        self.id=id
        self.Numrat=0
        self.listprod=set()
        self.idnum=0       
    
    def addrat(self, prodid):
        self.listprod.add(prodid)
        self.Numrat+=1

    def chidnum(self,num):
        self.idnum=num
        
    def printcust(self):
        for i in self.listprod:
            print(str(i)+"   "+str(self.idnum))


    
class FileProcess(object):
    """"Processing the data"""
    def __init__(self,dir,name):
        os.chdir(dir)
        self.name=name
        self.custdict=dict()
        self.lproduct=[]

    """"Read the file and 
    return the dict() of customer
    list of product"""

    def ReadFile(self):
        with open(self.name,"r") as file:
            idn=0
            for line in file:
                if line.find("Id")!=-1:
                    self.lproduct.append(Product(idn))
                    idn+=1               
                    # print(line)
                    
                if line.find("cutomer")!=-1:
                    # print(line)
                    custID=line.split()[2]
                    # print(custID)
                    self.lproduct[-1].addrat(custID)

                    if self.custdict.get(custID)==None:
                        self.custdict[custID]=Customer(custID)
                        self.custdict[custID].addrat(idn)
                    else:
                        self.custdict[custID].addrat(idn)
            num=0
            for i in self.custdict:
                self.custdict[i].chidnum(num)
                num+=1
        return [self.custdict, self.lproduct]
#          
    def printproduct(self):

        for i in self.lproduct:
            i.printprod()
            print("\n")

    def printcustomer(self):

        for i in self.custdict:
            print(i)
            self.custdict[i].printcust()
            print("\n")

        

class Bipartite(object):
    """"generate the bipartite graph"""
    def __init__(self, custdict, lprod):
        self.custdict=custdict
        self.lprod=lprod


    def Genmat(self):
        lenc=len(self.custdict)
        lenp=len(self.lprod)
        TBmat=np.zeros((lenp,lenc))
        for prod in self.lprod:
            # print(prod.id)
            # print(prod.listcst)
            for cst in prod.listcst:
                print(self.custdict[cst].idnum)
                TBmat[prod.id][self.custdict[cst].idnum]=1
                # print(custdict[cst].listprod)
        print(TBmat[2])
        Bmat=TBmat.transpose()
        Zeros=np.zeros((lenc, lenc))
        Tzeros=np.zeros((lenp,lenp))
        stack1=np.hstack((Zeros,Bmat))
        print(stack1.shape)
        stack2=np.hstack((TBmat,Tzeros))
        print(stack2.shape)
        stack=np.vstack((stack1,stack2))
        print(stack[2])

        return stack

                      

if __name__=="__main__":
    # dir="/scratch1/lisiw/network"
    dir="/Users/sisi/Google_Drive/computer_science/class/network_8480/project"
    File=FileProcess(dir, "test.txt")
    lcustprod=File.ReadFile()
    graph=Bipartite(lcustprod[0],lcustprod[1])
    mat=graph.Genmat()


    # File.printproduct()
    # File.printcustomer()

#     graph.ReadFile("test.txt")
    
    