from numpy import exp, sin,cos, linspace
import matplotlib.pyplot as plt
import os, time, glob
import math
import numpy as np
import pickle

def get_labels(idx,Mol):
    LBL=["R1","R2","R3","R4","R5","R6"]
    if Mol=="HFCO":
        LBL=["CH","CF","CO","HCO","FCO","Phi"]
    if Mol=="HONO":
        LBL=["N=O","O-N","OH","ONO","HON","Phi"]
    if Mol=="CS2":
        LBL=["CS1","CS2","Phi"]

    return LBL[idx]

def get_ri(ai,A):
    N=len(A)
    rai=0.0
    for i in range(N):
        LA=A[i].split()
        if len(LA)==3 and ai in LA:
            rai=float(LA[2])
    return rai

def read_hono_op_file(op_file):
    
    r=[];wu=[];c=0.0
    path = "./"
    OP_file=open(path+op_file,'r')
    OPF=OP_file.readlines()
    OP_file.close()
    N=len(OPF)
    #print("Number of lines= ",N)
    #AT=""
    #for ia in OPF:
    #    AT=AT+ia
    #print("AT= ",AT)
    #print("AT[0]= ",AT.split(",")[0])
    NN=80 # Number of Neurons
    ND=6 # Number of degrees of freedom
    for i in range(NN):
        wu.append([])
        ri="r"+str(i)
        #print(ri," = ",get_ri(ri,OPF))
        r.append(get_ri(ri,OPF))

    for i in range(NN):
        for j in range(ND):
            #wu.append([])
            ri="w"+str(i)+"u"+str(j)
            #print(ri," = ",get_ri(ri,OPF))
            wu[i].append(get_ri(ri,OPF))
    c=get_ri("c",OPF)    
        
    return r,wu,c


def read_hfco_op_file(op_file):
    
    r=[];wu=[];c=0.0
    path = "./"
    OP_file=open(path+op_file,'r')
    OPF=OP_file.readlines()
    OP_file.close()
    N=len(OPF)
    #print("Number of lines= ",N)
    #AT=""
    #for ia in OPF:
    #    AT=AT+ia
    #print("AT= ",AT)
    #print("AT[0]= ",AT.split(",")[0])
    NN=75 # Number of Neurons
    ND=6 # Number of degrees of freedom
    for i in range(NN):
        wu.append([])
        ri="r"+str(i)
        #print(ri," = ",get_ri(ri,OPF))
        r.append(get_ri(ri,OPF))

    for i in range(NN):
        for j in range(ND):
            #wu.append([])
            ri="w"+str(i)+"u"+str(j)
            #print(ri," = ",get_ri(ri,OPF))
            wu[i].append(get_ri(ri,OPF))
    c=get_ri("c",OPF)    
        
    return r,wu,c


def read_op_file(op_file):
    
    
    path = "./"
    OP_file=open(path+op_file,'r')
    OPF=OP_file.readlines()
    OP_file.close()
    N=len(OPF)
    #print("Number of lines= ",N)
    NN=30 # Number of Neurons
    ND=3 # Number of degrees of freedom
    r=[];wu=[];c=0.0
    for i in range(NN):
        wu.append([])
        ri="r"+str(i)
        #print(ri," = ",get_ri(ri,OPF))
        r.append(get_ri(ri,OPF))

    for i in range(NN):
        for j in range(ND):
            #wu.append([])
            ri="w"+str(i)+"u"+str(j)
            #print(ri," = ",get_ri(ri,OPF))
            wu[i].append(get_ri(ri,OPF))
    c=get_ri("c",OPF)    
    #print("r= ",r)
    #print("wu= ",wu)
    #print("c= ",c)
    #r=[];wu=[]
    return r,wu,c

def hfco_pes(hfco_r,hfco_th,t,t1,ith,ith1):
    

    nofd = 6 #Number of degrees of freedom
    nofn = 75 #Number of Neurons
    H_cm=219474.63 #Hartree to cm-1

    v = []
    #coord = [3.446,2.488,2.4714,0.8619,-0.2519,3.14285] # TRANS HFCO minimum
    #coord = [2.0632,2.534,2.22874,-0.6138,-0.54038,3.14285] #coordinates of EQ HFCO
    coord=[]
    #coord.append(t)
    coord.append(float(hfco_r.split(",")[0]))
    coord.append(float(hfco_r.split(",")[1]))
    coord.append(float(hfco_r.split(",")[2]))
    coord.append(float(hfco_th.split(",")[0]))
    coord.append(float(hfco_th.split(",")[1]))
    coord.append(float(hfco_th.split(",")[2]))
    #coord[ith]=t
    coord= coord[:ith]+[t]+coord[ith+1:]
    coord= coord[:ith1]+[t1]+coord[ith1+1:]
    
    # Now, get r[i]
    # get wu[i][j]
    # Compute qu[i][j] which is exp[wu[i][j],0]
    # Then compute V =sum(all neurons, r[i]*qu[i][j]) + c
    # We have CS2.op file, read and populate ri, wuij
    r,wu,c=read_hfco_op_file("hfco_75N.op")

    for i in range(nofn):
        v.append("0")

    for i in range(nofn):
        v[i] = 1.0
        for j in range(nofd):
            v[i] = v[i]*exp(wu[i][j]*coord[j])
            #v[i] = v[i]*exp(float(wu[i][j])*coord[j])

    V = 0.0
    for i in range(nofn):
        V = V + r[i]*v[i] 
        #V = V + float(LW[i])*v[i]*exp(float(b[i]))*float(rd[i])
    #print("eq-HFCO =",V+float(c)," Hartree")
    #return H_cm*(V+float(c))
    return H_cm*(V+c)

def pes_6D(hono_r,hono_th,t,t1,ith,ith1,ms):

    
    #r = pickle.load(open('r_params.pkl','rb'))
    #wu = pickle.load(open('w_params.pkl','rb'))
    if ms=="HFCO":
        r = pickle.load(open('r_hfco.pkl','rb'))
        wu = pickle.load(open('w_hfco.pkl','rb'))
        c = 0.219898906 # for HFCO Full 6D PES
    if ms=="HONO":
        r = pickle.load(open('r_hono.pkl','rb'))
        wu = pickle.load(open('w_hono.pkl','rb'))
        c = 0.065600224 # for HONO 6D PES        
    if ms=="CS2":
        r = pickle.load(open('r_cs2.pkl','rb'))
        wu = pickle.load(open('w_cs2.pkl','rb'))
        c = 0.272084269200190 # for CS2 3D PES
    #r=np.loadtxt("r_params.txt")
    #wu=np.loadtxt("w_params.txt")
    #c = 0.219898906 # for HFCO Cis
    
    nofd = len(wu[0]) #6 #Number of degrees of freedom
    nofn = len(r) #100 #Number of Neurons
    H_cm=219474.63 #Hartree to cm-1

    v = []
    #coord = [3.446,2.488,2.4714,0.8619,-0.2519,3.14285] # TRANS HFCO minimum
    #coord = [2.0632,2.534,2.22874,-0.6138,-0.54038,3.14285] #coordinates of EQ HFCO
    coord=[]
    RR=hono_r.split(",")
    RT=hono_th.split(",")
    for ia in RR:
        coord.append(float(ia))
    for ib in RT:
        coord.append(float(ib))
        
    #coord[ith]=t
    coord= coord[:ith]+[t]+coord[ith+1:]
    coord= coord[:ith1]+[t1]+coord[ith1+1:]
    
    for i in range(nofn):
        v.append("0")

    for i in range(nofn):
        v[i] = 1.0
        for j in range(nofd):
            v[i] = v[i]*exp(wu[i][j]*coord[j])

    V = 0.0
    for i in range(nofn):
        V = V + r[i]*v[i] 
    return H_cm*(V+c)


def hono_pes(hono_r,hono_th,t,t1,ith,ith1):

    
    r = pickle.load(open('r_params.pkl','rb'))
    wu = pickle.load(open('w_params.pkl','rb'))
    #r=np.loadtxt("r_params.txt")
    #wu=np.loadtxt("w_params.txt")
    c = 0.219898906 # for HFCO Cis
    
    nofd = len(wu[0]) #6 #Number of degrees of freedom
    nofn = len(r) #100 #Number of Neurons
    H_cm=219474.63 #Hartree to cm-1

    v = []
    #coord = [3.446,2.488,2.4714,0.8619,-0.2519,3.14285] # TRANS HFCO minimum
    #coord = [2.0632,2.534,2.22874,-0.6138,-0.54038,3.14285] #coordinates of EQ HFCO
    coord=[]
    coord.append(float(hono_r.split(",")[0]))
    coord.append(float(hono_r.split(",")[1]))
    coord.append(float(hono_r.split(",")[2]))
    coord.append(float(hono_th.split(",")[0]))
    coord.append(float(hono_th.split(",")[1]))
    coord.append(float(hono_th.split(",")[2]))
    #coord[ith]=t
    coord= coord[:ith]+[t]+coord[ith+1:]
    coord= coord[:ith1]+[t1]+coord[ith1+1:]
    
    for i in range(nofn):
        v.append("0")

    for i in range(nofn):
        v[i] = 1.0
        for j in range(nofd):
            v[i] = v[i]*exp(wu[i][j]*coord[j])

    V = 0.0
    for i in range(nofn):
        V = V + r[i]*v[i] 
    return H_cm*(V+c)


def cs2_pes(cs2_r,cs2_th,t,t1,ith,ith1):
    
    nofd = 3 #Number of degrees of freedom
    nofn = 30 #Number of Neurons
    H_cm=219474.63 #Hartree to cm-1

    v = []
    #coord = [3.446,2.488,2.4714,0.8619,-0.2519,3.14285] # TRANS HFCO minimum
    #coord = [2.0632,2.534,2.22874,-0.6138,-0.54038,3.14285] #coordinates of EQ HFCO
    coord=[]
    #coord.append(t)
    #coord.append(float(hfco.split(",")[0]))
    coord.append(float(cs2_r.split(",")[0]))
    coord.append(float(cs2_r.split(",")[1]))
    #coord.append(float(hfco_r.split(",")[2]))
    coord.append(float(cs2_th.split(",")[0]))
    #coord.append(float(hfco_th.split(",")[1]))
    #coord.append(float(hfco_th.split(",")[2]))
    #coord[ith]=t
    coord= coord[:ith]+[t]+coord[ith+1:]
    coord= coord[:ith1]+[t1]+coord[ith1+1:]
    
    # Now, get r[i]
    # get wu[i][j]
    # Compute qu[i][j] which is exp[wu[i][j],0]
    # Then compute V =sum(all neurons, r[i]*qu[i][j]) + c
    # We have CS2.op file, read and populate ri, wuij
    r,wu,c=read_op_file("cs2_30N.op")

    for i in range(nofn):
        v.append("0")

    for i in range(nofn):
        v[i] = 1.0
        for j in range(nofd):
            v[i] = v[i]*exp(wu[i][j]*coord[j])
            #v[i] = v[i]*exp(float(wu[i][j])*coord[j])

    V = 0.0
    for i in range(nofn):
        V = V + r[i]*v[i] 
        #V = V + float(LW[i])*v[i]*exp(float(b[i]))*float(rd[i])
    #print("eq-HFCO =",V+float(c)," Hartree")
    #return H_cm*(V+float(c))
    return H_cm*(V+c)

def compute_ver0(R,Q, bstr0, bstr1,Mo, resolution=20):
    
    b0=float(bstr0.split(",")[0])
    w0=float(bstr0.split(",")[1])
    T0=int(bstr0.split(",")[2])
    Tf=int(bstr0.split(",")[3])
    b1=float(bstr1.split(",")[0])
    w1=float(bstr1.split(",")[1])
    T1=int(bstr1.split(",")[2])
    Tf1=int(bstr1.split(",")[3])
    
    """Return filename of plot of the NN Fitted Molecular PES."""
    t = linspace(b0, w0, T0)
    t1 = linspace(b1, w1, T1)
    if Tf==Tf1:
        rr=get_labels(Tf,Mo)
        plt.figure()
        plt.title('%s 1D-cut PES'%Mo)
        plt.xlabel("%s"%rr)
        plt.ylabel("V(R$_{%s}$), cm$^{-1}$"%rr)
        #plt.plot(t, hfco_pes(R,Q,t,Tf),'-or')
        if Mo=="CS2":
            plt.plot(t, cs2_pes(R,Q,t,t,Tf,Tf),'-or')
        if Mo=="HONO":
            plt.plot(t, hono_pes(R,Q,t,t,Tf,Tf),'-or')
        if Mo=="HFCO":
            plt.plot(t, hono_pes(R,Q,t,t,Tf,Tf),'-or')

    else:
        
        Z=[]
        for i in t:
            Z0=[]
            for j in t1:
                if Mo=="CS2":
                    Z0.append(cs2_pes(R,Q,i,j,Tf,Tf1))
                if Mo=="HONO":
                    Z0.append(hono_pes(R,Q,i,j,Tf,Tf1))
                if Mo=="HFCO":
                    Z0.append(hono_pes(R,Q,i,j,Tf,Tf1))
            Z.append(Z0)
        Z=np.array(Z)

        [X, Y] = np.meshgrid(t, t1)

        rr=get_labels(Tf,Mo)
        rr0=get_labels(Tf1,Mo)
        fig,ax=plt.subplots(1,1)
        ax = plt.axes(projection='3d')
        ax.set_title('%s Contour PES'%Mo)
        ax.set_xlabel('%s'%rr)
        ax.set_ylabel('%s'%rr0)
        plt.contour(X, Y, Z, 100, cmap='RdGy');
        plt.colorbar();

    if not os.path.isdir('static'):
        os.mkdir('static')
    else:
        # Remove old plot files
        for filename in glob.glob(os.path.join('static', '*.png')):
            os.remove(filename)
    # Use time since Jan 1, 1970 in filename in order make
    # a unique filename that the browser has not chached
    plotfile = os.path.join('static', str(time.time()) + '.png')
    plt.savefig(plotfile)
    return plotfile

def compute(R,Q, bstr0, bstr1,Mo, resolution=20):
    
    b0=float(bstr0.split(",")[0])
    w0=float(bstr0.split(",")[1])
    T0=int(bstr0.split(",")[2])
    Tf=int(bstr0.split(",")[3])
    b1=float(bstr1.split(",")[0])
    w1=float(bstr1.split(",")[1])
    T1=int(bstr1.split(",")[2])
    Tf1=int(bstr1.split(",")[3])
    
    """Return filename of plot of the NN Fitted Molecular PES."""
    t = linspace(b0, w0, T0)
    t1 = linspace(b1, w1, T1)
    if Tf==Tf1:
        rr=get_labels(Tf,Mo)
        plt.figure()
        plt.title('%s 1D-cut PES'%Mo)
        plt.xlabel("%s"%rr)
        plt.ylabel("V(R$_{%s}$), cm$^{-1}$"%rr)
        #plt.plot(t, hfco_pes(R,Q,t,Tf),'-or')
        if Mo=="CS2" or Mo=="HONO" or Mo=="HFCO":
            plt.plot(t, pes_6D(R,Q,t,t,Tf,Tf,Mo),'-or')
    else:
        
        Z=[]
        for i in t:
            Z0=[]
            for j in t1:
                if Mo=="CS2" or Mo=="HONO" or Mo=="HFCO":
                    Z0.append(pes_6D(R,Q,i,j,Tf,Tf1,Mo))
            Z.append(Z0)
        Z=np.array(Z)

        [X, Y] = np.meshgrid(t, t1)

        rr=get_labels(Tf,Mo)
        rr0=get_labels(Tf1,Mo)
        fig,ax=plt.subplots(1,1)
        ax = plt.axes(projection='3d')
        ax.set_title('%s Contour PES'%Mo)
        ax.set_xlabel('%s'%rr)
        ax.set_ylabel('%s'%rr0)
        plt.contour(X, Y, Z, 100, cmap='RdGy');
        plt.colorbar();

    if not os.path.isdir('static'):
        os.mkdir('static')
    else:
        # Remove old plot files
        for filename in glob.glob(os.path.join('static', '*.png')):
            os.remove(filename)
    # Use time since Jan 1, 1970 in filename in order make
    # a unique filename that the browser has not chached
    plotfile = os.path.join('static', str(time.time()) + '.png')
    plt.savefig(plotfile)
    return plotfile

