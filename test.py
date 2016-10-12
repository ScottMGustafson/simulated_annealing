import numpy as np
#from test_data import x,y,ye
from simulated_annealing import *
from matplotlib import pyplot as plt

#true param values
a_true=-12.
b_true=43.
c_true=6000.

def generate_test_data(SNR=20.,a=a_true,b=b_true,c=c_true):
    xx=np.random.uniform(-10.,10.,size=100)
    model=test_fn(xx,a,b,c)
    _scale=np.fabs(np.median(model))/SNR,
    noise=np.random.normal(scale=_scale, size=np.shape(xx)[0])
    y=model+noise
    ye=np.fabs(np.random.normal(scale=_scale, size=np.shape(xx)[0]))
    return xx, y, ye

def test_fn(x,a=a_true,b=b_true,c=c_true):
    return a*x**2+b*x+c

if __name__=="__main__":
    x,y,ye=generate_test_data()
    param_samples, out =param_locked_SA(test_fn,x,y,ye,
                                            a=[-15.,-8.], 
                                            b=[40.,50.], 
                                            c=[400.,750.])
    #param_samples, out =simulated_annealing(test_fn,x,y,ye,
    #                                        a=[-15.,-8.], 
    #                                        b=[40.,50.], 
    #                                        c=[400.,750.])


    plt.plot(x,y,'ko')
    xx=np.arange(np.amin(x),np.amax(x), 0.01)
    plt.plot(xx, test_fn(xx,**param_samples[0]), 'r-')
    plt.show()

    x=[it['a'] for it in param_samples.tolist()]
    plt.plot(x,out,'ko')
    plt.xlabel('a')
    plt.ylabel(r'$\chi^2$')
    plt.show()


    out=np.array([it for it in out.tolist() if it<out[0]+4.])
    param_samples=param_samples[0:np.shape(out)[0]-1]

    for param in ['a','b','c']:
        lst=[it[param] for it in param_samples.tolist()]
        if len(lst)<20:
            print('too few good points:')
            break
        _max=max(lst)
        _min=max(lst)
        best=param_samples[0][param]
        print(best, _max-best, best-_min)

   

    
