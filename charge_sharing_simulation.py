import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy import optimize
from scipy import special

def charge_sharing():
    print('Now we study charge-sharing')
    fig1, sub1 = plt.subplots()
    fig2, sub2 = plt.subplots()

    pp = 75 #micron
    dp = 0.1 #micron
    nop = 1000
    Pgridx = np.random.uniform(-0.2*pp,+1.2*pp, size=nop) # np.arange(-0.25*pp,+1.25*pp,0.1*pp)
    Pgridy = np.random.uniform(-0.2*pp,+1.2*pp, size=nop) #np.arange(-0.25*pp,+1.25*pp,0.1*pp)
    # Pgridx = [pp/2] 
    # Pgridy = [pp/2]

    print('Particles:', nop)

    Evgridx = np.arange(-0.5*pp,+1.5*pp,0.01*pp)
    Evgridy = np.arange(-0.5*pp,+1.5*pp,0.01*pp)
    X, Y = np.meshgrid(Evgridx, Evgridy)

    sigma = 7.5 # micron
    sigmaeV = 90*3.6 # eV
    energy = 8000
    charges = []

    for x0,y0 in zip(Pgridx,Pgridy):
        # print('chcloud.shape:', np.shape(chcloud))
        result, error = dblquad(gauss2d, 0, pp, 0, pp, args=(x0,y0,sigma,energy))
        charges.extend(np.random.normal(result, sigmaeV,10))            

        if np.random.uniform(0,1) > 0.01: continue
        sub1.cla()
        sub2.cla()
        chcloud = gauss2d(X,Y,x0,y0,sigma,1)
        sub1.contourf(X,Y,chcloud)
        sub1.plot(x0,y0, '.')
        sub1.plot([0,pp,pp,0,0],[0,0,pp,pp,0],'-', color='black')
        n,b,p = sub2.hist(charges, bins=100, range=(0,1.5*energy),histtype='step')
        sub1.set_xlabel('x(um)')
        sub1.set_ylabel('y(um)')

        fig1.show()
        fig2.show()
        plt.pause(0.1)

    sub2.cla()
    sub2.set_xlabel('Energy (eV)')
    sub2.set_ylabel('Counts')
    n,b,p = sub2.hist(charges, bins=100, range=(0.4*energy,1.5*energy),histtype='step')
    param, cov = optimize.curve_fit(gauss_box,b[1:],n,p0=[energy,sigmaeV,nop,nop/2])
    print(param)
    sub2.plot(b[1:], gauss_box(b[1:],param[0],param[1],param[2],param[3]),'-')
    sub2.plot(b[1:], gauss1d(b[1:],param[0],param[1],param[2]),':')
    sub2.plot(b[1:], box(b[1:],param[0],param[1],param[3]),':')
    fig1.show()
    fig2.show()

def gauss2d(x,y,x0,y0,sigma,I):
    return I/(2*np.pi*np.power(sigma,2))*np.exp(-0.5*np.power(x-x0,2)/np.power(sigma,2)-0.5*np.power(y-y0,2)/np.power(sigma,2))

def gauss1d(x,x0,sigma,A):
    return A/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*np.power(x-x0,2)/np.power(sigma,2))

def box(x,x0,sigmac,Ac):
    return Ac*special.erfc(np.sqrt(2)*(x-x0)/sigmac)

def gauss_box(x,x0,sigma,A,Ac,An,x1,Ab):
    return gauss1d(x,x0,sigma,A)+ box(x,x0,sigma,Ac) + gauss1d(x,0,sigma,An)+ gauss1d(x,x0+x1,sigma,Ab)

def Cu_Fluo():
    print('Now we study charge-sharing')
    fig1, sub1 = plt.subplots()
    fig2, sub2 = plt.subplots()

    pp = 75 #micron
    array_s = 5
    nop = 1000
    ev_mult = 1
    Pgridx = np.random.uniform(0.2*pp,(array_s-.2)*pp, size=nop) # np.arange(-0.25*pp,+1.25*pp,0.1*pp)
    Pgridy = np.random.uniform(0.2*pp,(array_s-.2)*pp, size=nop) #np.arange(-0.25*pp,+1.25*pp,0.1*pp)
    # Pgridx = [pp/2] 
    # Pgridy = [pp/2]

    print('Particles:', nop)

    Evgridx = np.arange(-0.5*pp,(array_s+.5)*pp,0.01*pp)
    Evgridy = np.arange(-0.5*pp,(array_s+.5)*pp,0.01*pp)
    X, Y = np.meshgrid(Evgridx, Evgridy)

    sigma = 7.5 # micron
    sigmaeV = np.sqrt(np.power(50*3.6,2) + np.power(50,2)) # eV
    print('Energy resolution:', sigmaeV, 'eV')
    energy = np.array([8046 if np.random.uniform(0,1) > 0.12 else 8904 for i in range(nop)]) #Cu fluo 
    #energy = [6405 if np.random.uniform(0,1) > 0.12 else 7059 for i in range(nop)] #Fe fluo 
    charges = [[] for i in range(array_s*array_s)]
    cluster = []

    print('Ka:',np.count_nonzero(energy == 8046)*ev_mult)
    print('Kb:',np.count_nonzero(energy == 8904)*ev_mult)

    thr0 = 8046/2
    thr1 = (8046 + 8904)/2
    count0 = 0
    count1 = 0
    print('thr0:',thr0,'- thr2:',thr1)
    x_ka_corr = []
    y_ka_corr = []
    x_kb_corr = []
    y_kb_corr = []
    x_kb_as_ka = []
    y_kb_as_ka = []
    x_ka_as_kb = []
    y_ka_as_kb = []

    p_done = 0
    for x0,y0,e in zip(Pgridx,Pgridy,energy):
        # print('chcloud.shape:', np.shape(chcloud))
        #print('coordinates:', x0,y0)
#        print('E:', e )
        for i in range(array_s*array_s):
            ix = i%array_s
            iy = int(i/array_s)
            result, error = dblquad(gauss2d, 
                            iy*pp,(iy+1)*pp,ix*pp,(ix+1)*pp, args=(x0,y0,sigma,e))
            ch = np.random.normal(result, sigmaeV, ev_mult)
            charges[i].extend(ch)
#            print(ix,iy,'charges:',result,ch)
            for ev in ch:
                if ev > thr0 and ev <= thr1:
                    count0+=1
                    if e == 8046:
                        x_ka_corr.append(x0%pp)
                        y_ka_corr.append(y0%pp)
                    if e == 8904:    
                        x_kb_as_ka.append(x0%pp)
                        y_kb_as_ka.append(y0%pp)
                if ev > thr1: 
                    count1+=1
                    if e == 8904:
                        x_kb_corr.append(x0%pp)
                        y_kb_corr.append(y0%pp)
                    if e == 8046:
                        x_ka_as_kb.append(x0%pp)
                        y_ka_as_kb.append(y0%pp)                        

            #print('Pixel',i, ix,iy, 'limits:', ix*pp,(ix+1)*pp,iy*pp,(iy+1)*pp, result)
        p_done +=1
        if (p_done % 10 == 0) : 
            print(f'{p_done/nop*100:.2f}%', end = '\r')            

        if np.random.uniform(0,1) > -0.05: continue
        sub1.cla()
        sub2.cla()
        chcloud = gauss2d(X,Y,x0,y0,sigma,1)
        sub1.contourf(X,Y,chcloud)
        sub1.plot(x0,y0, '.')
        for i in range(array_s*array_s):
            ix = i%array_s
            iy = int(i/array_s)
            sub1.plot([ix*pp,(ix+1)*pp,(ix+1)*pp,ix*pp,ix*pp],
                      [iy*pp,iy*pp,(iy+1)*pp,(iy+1)*pp,iy*pp],'-', label=i)
            n,b,p = sub2.hist(charges[i], bins=200, range=(0,1.5*e),histtype='step')
        sub1.set_xlabel('x(um)')
        sub1.set_ylabel('y(um)')
        #sub1.legend()

        fig2.show()
        fig1.show()
        plt.pause(0.1)

    print('Counter0:', count0)
    print('Counter1:', count1)

    print('Ka corr:', np.shape(x_ka_corr))
    print('Kb corr:', np.shape(x_kb_corr))
    print('Kb as Ka:', np.shape(x_kb_as_ka))
    print('Ka as Kb:', np.shape(x_ka_as_kb))

    bins = 20
    fig4, sub4 = plt.subplots() #ka_corr
    H, xedges, yedges = np.histogram2d(x_ka_corr, y_ka_corr, bins=(bins, bins), range=[[0,pp],[0,pp]])
    Hn, xedgesn, yedgesn = np.histogram2d(Pgridx%pp, Pgridy%pp, bins=(bins, bins), range=[[0,pp],[0,pp]])
    ka_corr_norm = np.zeros_like(H)
    for ix in range(np.shape(H)[0]):
        for iy in range(np.shape(H)[1]):
            ka_corr_norm[ix,iy]=H[ix,iy]/Hn[ix,iy]/ev_mult if Hn[ix,iy]!=0 else 0

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = sub4.imshow(ka_corr_norm, extent=extent, interpolation='none',origin ='lower')
    fig4.colorbar(im, ax=sub4)
    fig4.show()

    sub2.cla()
    sub2.set_xlabel('Energy (eV)')
    sub2.set_ylabel('Counts')
    fig3, sub3 = plt.subplots() #scurve
    #for i in range(array_s*array_s):
        #n,b,p = sub2.hist(charges[i], bins=200, range=(0*np.min(energy),1.5*np.max(energy)),histtype='step', label=i)
        #param, cov = optimize.curve_fit(gauss_box,b[1:],n,p0=[np.mean(energy),sigmaeV,nop,nop/2])
        #print(param)
        # sub2.plot(b[1:], gauss_box(b[1:],param[0],param[1],param[2],param[3]),'-')
        # sub2.plot(b[1:], gauss1d(b[1:],param[0],param[1],param[2]),':')
        # sub2.plot(b[1:], box(b[1:],param[0],param[1],param[3]),':')

    na,ba,pa = sub2.hist(np.array(charges).flatten(), bins=200, range=(0*np.min(energy),1.5*np.max(energy)),histtype='step', label='average')
    scurve = [np.sum(na[i:]) for i in range(len(na))]
    sub3.plot(ba[1:],scurve)
    #cluster = np.sum(charges,axis=0)
    cluster = []
    for i in range(array_s-1):
        for j in range(array_s-1):
            indexes = [[i,j],[i+1,j],[i,j+1],[i+1,j+1]]
            lin_indexes = [ int(x*array_s+y) for x,y in indexes] 
            # print(i,j,indexes, lin_indexes)
            cluster.extend( np.sum([charges[jj] for jj in lin_indexes], axis=0)) 
            
    # print('Charges:', np.shape(charges))
    # print('Cluster:', np.shape(cluster))
    nc,bc,pc = sub2.hist(cluster, bins=200, range=(0*np.min(energy),1.5*np.max(energy)),histtype='step', label='clusters')

    parama, cova = optimize.curve_fit(gauss_box,ba[1:],na,
                        p0=[np.min(energy),sigmaeV,nop,nop/2,nop,1000,nop],
                        bounds=[[0,0,0,0,0,0,0],[9000,1e3,1e6,1e6,1e6,1500,1e6]])
    print(parama)
    sub2.plot(ba[1:], gauss_box(ba[1:],parama[0],parama[1],parama[2],parama[3],parama[4],parama[5],parama[6]),'-')

    paramc, covc = optimize.curve_fit(gauss_box,bc[1:],nc,
                        p0=[np.min(energy),sigmaeV,nop,nop/2,nop,1000,nop],
                        bounds=[[0,0,0,0,0,0,0],[9000,1e3,1e6,1e6,1e6,1500,1e6]])
    print(paramc)
    sub2.plot(bc[1:], gauss_box(bc[1:],paramc[0],paramc[1],paramc[2],paramc[3],paramc[4],paramc[5],paramc[6]),'-')
    sub2.plot([thr0,thr0],[0,np.max(nc)],':')
    sub2.plot([thr1,thr1],[0,np.max(nc)],':')

    sub2.legend()
    fig1.show()
    fig2.show()

    sub3.set_xlabel('Energy (eV)')
    sub3.set_ylabel('Counts')
    fig3.show()

    fig5, sub5 = plt.subplots() #kb_corr
    fig6, sub6 = plt.subplots() #kb_as_ka
