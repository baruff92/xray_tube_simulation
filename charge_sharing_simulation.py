import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.integrate import quad
from scipy import optimize
from scipy import special
from datetime import datetime

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

def gauss2dwr(x,y,x0,y0,sigma,I,pp,array_s):
    return (gauss2d(x,y,x0,y0,sigma,I) + 
            gauss2d(x,y,x0+pp*array_s,y0,sigma,I) + 
            gauss2d(x,y,x0+pp*array_s,y0+pp*array_s,sigma,I) +
            gauss2d(x,y,x0,y0+pp*array_s,sigma,I)+
            gauss2d(x,y,x0-pp*array_s,y0,sigma,I) + 
            gauss2d(x,y,x0-pp*array_s,y0-pp*array_s,sigma,I) +
            gauss2d(x,y,x0,y0-pp*array_s,sigma,I) )

def gauss1d(x,x0,sigma,A):
    return A/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*np.power(x-x0,2)/np.power(sigma,2))

def box(x,x0,sigmac,Ac):
    return Ac*special.erfc((x-x0)/(sigmac*np.sqrt(2)))

def box2side(x,x0,sigmac,Ac,x1,sigma1,mu):
    return Ac*special.erfc((x-x0)/(sigmac*np.sqrt(2)))*special.erfc((x1-x)/(sigma1*np.sqrt(2)))*np.exp(-x*mu) 

def gauss_box(x,x0,sigma,A,Ac,x1,Ab):
    return gauss1d(x,x0,sigma,A)+ box(x,x0,sigma,Ac) + gauss1d(x,x0+x1,sigma,Ab)

def scurve_func(th, flex, noise, amplitude, chargesharing):
    y=0.5*amplitude*(1+special.erf((flex-th)/(noise*np.sqrt(2))))*(1+chargesharing*(flex-th)/noise)
    return y

def Cu_Fluo():
    print('Now we study charge-sharing')
    fig1, sub1 = plt.subplots()
    fig2, sub2 = plt.subplots()

    pp = 25 #micron
    array_s = 5
    nop = 400
    ev_mult = 10
    Pgridx = np.random.uniform(0.*pp,(array_s-0.)*pp, size=nop) # np.arange(-0.25*pp,+1.25*pp,0.1*pp)
    Pgridy = np.random.uniform(0.*pp,(array_s-0.)*pp, size=nop) #np.arange(-0.25*pp,+1.25*pp,0.1*pp)
    # Pgridx = [1.1*pp] 
    # Pgridy = [0.1*pp]

    print('Particles:', nop)

    Evgridx = np.arange(0.*pp,(array_s+0.)*pp,0.01*pp)
    Evgridy = np.arange(0.*pp,(array_s+0.)*pp,0.01*pp)
    X, Y = np.meshgrid(Evgridx, Evgridy)

    sigma = 7.5 # micron
    thre_disp = 0 # eV
    intr_nois = 31*3.6 # eV
    sigmaeV = np.sqrt(np.power(intr_nois,2) + np.power(thre_disp,2)) # eV
    print('Energy resolution:', sigmaeV, 'eV')
    print('Intrinsic noise:', intr_nois, 'eV')
    print('Threshold disperion:', thre_disp, 'eV')
    energy = np.array([8046 if np.random.uniform(0,1) > 0.12 else 8904 for i in range(nop)]) #Cu fluo 
    #energy = [6405 if np.random.uniform(0,1) > 0.12 else 7059 for i in range(nop)] #Fe fluo 
    charges = [[] for i in range(array_s*array_s)]
    cluster = []

    num_ka = np.count_nonzero(energy == 8046)*ev_mult
    num_kb = np.count_nonzero(energy == 8904)*ev_mult
    print('Ka:',num_ka)
    print('Kb:',num_kb)

    thr0 = 7999.4/2
    thr1 = 8500
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

    start = datetime.now()

    p_done = 0
    for x0,y0,e in zip(Pgridx,Pgridy,energy):
        # print('chcloud.shape:', np.shape(chcloud))
        # print('coordinates:', x0,y0)
        ipx = int(x0/pp) 
        ipy = int(y0/pp)
        # print(ipx,ipy)
        itochx = [(ipx-1)%array_s,ipx%array_s,(ipx+1)%array_s]
        itochy = [(ipy-1)%array_s,ipy%array_s,(ipy+1)%array_s]
        # print('search in:', itochx,itochy)
#        print('E:', e )
        for i in range(array_s*array_s):
            ix = i%array_s
            iy = int(i/array_s)
            if ix in itochx and iy in itochy: 
                if ix==array_s-1 or ix==0 or iy==array_s-1 or iy==0:     
                    result, error = dblquad(gauss2dwr,
                            iy*pp,(iy+1)*pp,ix*pp,(ix+1)*pp, args=(x0,y0,sigma,e,pp,array_s), epsabs=0.5)
                else:
                    result, error = dblquad(gauss2d,
                            iy*pp,(iy+1)*pp,ix*pp,(ix+1)*pp, args=(x0,y0,sigma,e), epsabs=0.5)

            else: result = 0
#            print(ix,iy,'charges:',result)
            ch = np.random.normal(result, sigmaeV, ev_mult)
            charges[i].extend(ch)
            for ev in ch:
                if ev > thr0 and ev <= thr1:
                    count0+=1
                    if e == 8046:
                        x_ka_corr.append(x0%pp)
                        y_ka_corr.append(y0%pp)
                    if e == 8904:    
                        x_kb_as_ka.append(x0%pp)
                        y_kb_as_ka.append(y0%pp)
                elif ev > thr1: 
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
        chcloud = (gauss2d(X,Y,x0,y0,sigma,1) + 
                    gauss2d(X,Y,x0+pp*array_s,y0,sigma,1) + 
                    gauss2d(X,Y,x0+pp*array_s,y0+pp*array_s,sigma,1) +
                    gauss2d(X,Y,x0,y0+pp*array_s,sigma,1)+
                    gauss2d(X,Y,x0-pp*array_s,y0,sigma,1) + 
                    gauss2d(X,Y,x0-pp*array_s,y0-pp*array_s,sigma,1) +
                    gauss2d(X,Y,x0,y0-pp*array_s,sigma,1)                    )
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

    now = datetime.now()
    elapsed_time = now-start
    print('Elapsed time:', str(elapsed_time))

    print('Counter0:', count0, '/', num_ka, ' :', count0/num_ka)
    print('Counter1:', count1, '/', num_kb, ' :', count1/num_kb)

    print('Ka corr:', np.shape(x_ka_corr)[0], '/', num_ka, ' :', np.shape(x_ka_corr)[0]/num_ka)
    print('Kb corr:', np.shape(x_kb_corr)[0], '/', num_kb, ' :', np.shape(x_kb_corr)[0]/num_kb)
    print('Kb as Ka:', np.shape(x_kb_as_ka)[0], '/', num_kb, ' :', np.shape(x_kb_as_ka)[0]/num_kb)
    print('Ka as Kb:', np.shape(x_ka_as_kb)[0], '/', num_ka, ' :', np.shape(x_ka_as_kb)[0]/num_ka)
    print('Correct counts (ka_corr/count0):', np.shape(x_ka_corr)[0]/count0)

    bins = 20
    fig4, sub4 = plt.subplots() #ka_corr
    H, xedges, yedges = np.histogram2d(x_ka_corr, y_ka_corr, bins=(bins, bins), range=[[0,pp],[0,pp]])
    Hn, xedgesn, yedgesn = np.histogram2d([x%pp for x,e in zip(Pgridx,energy) if e==8046],
                                            [y%pp for y,e in zip(Pgridy,energy) if e==8046], bins=(bins, bins), range=[[0,pp],[0,pp]])
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

    na,ba,pa = sub2.hist(np.array(charges).flatten(), bins=400, range=(0*np.min(energy),1.5*np.max(energy)),histtype='step', label='Spectrum')
    scurve = [np.sum(na[i:]) for i in range(len(na))]
    sub3.plot(ba[1:],scurve)

    xmin = 100
    paramscurve, cova = optimize.curve_fit(scurve_func,ba[xmin+1:],scurve[xmin:],
                        p0=[8046, sigmaeV, nop, 0.15])
    sub3.plot(ba,scurve_func(ba,paramscurve[0],paramscurve[1],paramscurve[2],paramscurve[3]),':')
    print('flex:', paramscurve[0])
    print('sigma:', paramscurve[1])
    print('Chsharing:', paramscurve[3])
    sub3.set_xlabel('Energy (eV)')
    sub3.set_ylabel('Counts')
    fig3.show()

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

    parama, cova = optimize.curve_fit(gauss_box,ba[xmin+1:],na[xmin:],
                        p0=[8046,sigmaeV,nop,10,1000,nop/5],
                        bounds=[[0,0,0,0,500,0],[9000,1e3,1e6,1e6,1500,1e6]])
    print('X0:', parama[0])
    print('sigma:', parama[1])
    print('X1:', parama[0]+parama[4])
    print('CS:', parama[3])

    #sub2.plot(ba[1:], gauss_box(ba[1:],parama[0],parama[1],parama[2],parama[3],parama[4],parama[5]),'-',label='Fit')
    #sub2.plot(ba[1:], gauss1d(ba[1:],parama[0],parama[1],parama[2]),'--', label='K_a')
    #sub2.plot(ba[1:], gauss1d(ba[1:],parama[0]+parama[4],parama[1],parama[5]),'--', label='K_b')
    #sub2.plot(ba[1:], box(ba[1:],parama[0],parama[1],parama[3]),'--', label='Charge sharing')

    nc,bc,pc = sub2.hist(cluster, bins=400, range=(0*np.min(energy),1.5*np.max(energy)),histtype='step', label='Clusters')
    paramc, covc = optimize.curve_fit(gauss_box,bc[xmin+1:],nc[xmin:],
                         p0=[8046,sigmaeV,nop,10,1000,nop/5],
                         bounds=[[0,0,0,0,500,0],[9000,1e3,1e6,1e6,1500,1e6]])
    #print(paramc)
    sub2.plot(bc[1:], gauss_box(bc[1:],paramc[0],paramc[1],paramc[2],paramc[3],paramc[4],paramc[5]),'-', label='Fit')
    sub2.plot([thr0,thr0],[0,np.max(nc)],':')
    sub2.plot([thr1,thr1],[0,np.max(nc)],':')

    sub2.set_xlim([-100,12000])
    sub2.set_ylim([-2,110])
    sub2.legend()
    fig1.show()
    fig2.show()

    fig5, sub5 = plt.subplots() #kb_corr
    sub5.scatter(x_kb_corr,y_kb_corr,alpha=0.1, marker='.', color='blue', label='Kb correct')
    sub5.scatter(x_kb_corr,pp-np.array(y_kb_corr),alpha=0.1, marker='.', color='blue')
    sub5.scatter(pp-np.array(x_kb_corr),pp-np.array(y_kb_corr),alpha=0.1, marker='.', color='blue')
    sub5.scatter(pp-np.array(x_kb_corr),np.array(y_kb_corr),alpha=0.1, marker='.', color='blue')
    sub5.scatter(x_kb_as_ka,y_kb_as_ka,alpha=0.1, marker='.', color='red', label='Kb as Ka')
    sub5.scatter(x_kb_as_ka,pp-np.array(y_kb_as_ka),alpha=0.1, marker='.', color='red')
    sub5.scatter(pp-np.array(x_kb_as_ka),y_kb_as_ka,alpha=0.1, marker='.', color='red')
    sub5.scatter(pp-np.array(x_kb_as_ka),pp-np.array(y_kb_as_ka),alpha=0.1, marker='.', color='red')
    sub5.plot([0,pp,pp,0,0],[0,0,pp,pp,0],'-')
    sub5.set_xlabel('x(um)')
    sub5.set_ylabel('y(um)')
    sub5.set_xlim([-0.1*pp,1.1*pp])
    sub5.set_ylim([-0.1*pp,1.1*pp])
    leg = sub5.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    fig5.show()

    fig6, sub6 = plt.subplots() #ka_corr
    sub6.scatter(x_ka_corr,y_ka_corr,alpha=0.1, marker='.', color='blue', label='Ka correct')
    sub6.scatter(x_ka_corr,pp-np.array(y_ka_corr),alpha=0.1, marker='.', color='blue')
    sub6.scatter(pp-np.array(x_ka_corr),pp-np.array(y_ka_corr),alpha=0.1, marker='.', color='blue')
    sub6.scatter(pp-np.array(x_ka_corr),np.array(y_ka_corr),alpha=0.1, marker='.', color='blue')
    sub6.plot([0,pp,pp,0,0],[0,0,pp,pp,0],'-')
    sub6.set_xlabel('x(um)')
    sub6.set_ylabel('y(um)')
    sub6.set_xlim([-0.1*pp,1.1*pp])
    sub6.set_ylim([-0.1*pp,1.1*pp])
    leg = sub6.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    fig6.show()


def use_modulus():
    fig5, sub5 = plt.subplots() #kb_corr
    x0 = 10
    s = 10
    A = 1
    x = np.arange(-75,3*75,0.1)
    sub5.plot(x%150,gauss1d(x,x0,s,A))

    result, error = quad(gauss1d, 
                      75,150, args=(x0,s,A))
    result1, error = quad(gauss1d, 
                      0,75, args=(x0,s,A))

    print(result)
    print(result1)
    fig5.show()

def make_plot_lgads():
    print('We make a nice plot')

    fig2, sub2 = plt.subplots()
    sub2.cla()
    sub2.set_xlabel('Energy (eV)')
    sub2.set_ylabel('Counts (Arb. un.)')

    x = np.arange(-200,3000,1)
    x_e = 2000 
    A_e = 1500
    x_h = x_e/3
    A_h = 1500
    sigma_e = 60
    sigma_h = sigma_e/1.2
    ch_sh = 0.5
    mu = 0.0008

    electron_generated = gauss1d(x,x_e,sigma_e,A_e) + box(x,x_e,sigma_e,ch_sh) + gauss1d(x,0,sigma_e,A_e*20)
    hole_generated = gauss1d(x,x_h,sigma_h,A_h) + box(x,x_h,sigma_h,ch_sh+0.1) + gauss1d(x,0,sigma_e,A_e*20)
    part_mult = box2side(x,x_e,sigma_e,ch_sh*4,x_h,sigma_h,mu)
    tot = electron_generated+part_mult+hole_generated
    sub2.plot(x, electron_generated, '--', label='Electron-generated')
    sub2.plot(x, hole_generated, '--', label='Hole-generated')
    sub2.plot(x, part_mult, '--', label='Partial multiplication')
    sub2.plot(x, tot, '-', label='Total')
    sub2.set_xlim([-50,2500])
    sub2.set_ylim([-1,20])    
    sub2.legend()
    fig2.show()

    fig3, sub3 = plt.subplots()
    sub3.cla()
    sub3.set_xlabel('Energy (eV)')
    sub3.set_ylabel('Counts (Arb. un.)')
    SCelectron_generated = [np.sum(electron_generated[i:]) for i in range(len(electron_generated))] 
    SChole_generated = [np.sum(hole_generated[i:]) for i in range(len(hole_generated))] 
    SCpart_mult = [np.sum(part_mult[i:]) for i in range(len(part_mult))] 
    SCtot = [np.sum(tot[i:]) for i in range(len(tot))] 
    sub3.plot(x, SCelectron_generated, '--', label='Electron-generated')
    sub3.plot(x, SChole_generated, '--', label='Hole-generated')
    sub3.plot(x, SCpart_mult, '--', label='Partial multiplication')
    sub3.plot(x, SCtot, '-', label='Total')
    sub3.set_xlim([-50,2500])
    sub3.set_ylim([-500,10000])    
    sub3.legend()
    fig3.show()
