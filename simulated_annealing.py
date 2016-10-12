import numpy as np

def accept_prob(old,new,T,*args, **kwarg):
    #if new<old, the new solution is better, thus bigger accept prob
    if new<old:
        return 1.
    else:
        return  np.exp((old-new)/T)

def get_random_samples(num_samples, **param_ranges):
    param_samples={}
    for key, val in dict(param_ranges).items():
        param_samples[key]=(val[1]-val[0])*np.random.random_sample(num_samples)+val[0]
    return param_samples

def chi2(model, y, ye):
    return np.sum((y-model)*(y-model)/(ye*ye))

def simulated_annealing(fn, *x, **kwargs):
    """
    input:
    ------
    fn: a fuction we are testing
    *x: data.  in this example, we will use, x,y,y_error, though it may be anything
    **param_ranges: guess ranges for the params of fn

    output:
    -------
    a tuple of:
    --array of dicts of param values in the form np.array([{params},{blah},{blah}])
    --chi2 for the above described params
    """

    x,y,ye=tuple(x)

    Tmin=float(kwargs.pop("Tmin",.0001))
    alpha=float(kwargs.pop("alpha",0.9))
    T=float(kwargs.pop("Tstart",10000.))
    epsilon=float(kwargs.pop("epsilon",10e-8)) #tolerance multiplier to define locked params
    max_steps=int(kwargs.pop("max_steps",1000))
    chi2_lim=int(kwargs.pop("chi2_lim",4))
    param_ranges=kwargs
    nparams=len(dict(param_ranges).keys())
    guess=get_random_samples(1, **param_ranges)

    old_chi2 = chi2( fn(x,**guess), y,ye)

    samples = [ [guess, old_chi2] ]

    try:
        while T>Tmin:
            _step=4.
            for i in range(max_steps):
                new_guess={}
                for key, val in guess.items():
                    #if a param_range shows that the item is locked, don't vary it
                    _rng=np.fabs(param_ranges[key][1]-param_ranges[key][0])
                    if _rng<epsilon:
                        new_guess[key]=guess[key]
                    else:
                        step_size=_step*_rng
                        new_guess[key] = guess[key] + np.random.normal(scale=np.fabs(step_size))
                new_chi2=chi2(fn(x, **new_guess), y,ye)
                if accept_prob(old_chi2, new_chi2,T)>np.random.uniform():
                    old_chi2=new_chi2
                    guess = new_guess
                    samples.append( [guess, new_chi2] )
                    if _step<0.05:
                        break
                    else:
                        _step*=alpha  
                #iteratively narrow the param range when a sucessful model is found
            if i==max_steps:  
                #if no convergence, restart by reinitializing T
                T=old_chi2
                guess=get_random_samples(1, **param_ranges)
                if new_chi2<min_chi2+10.*chi2_lim:
                    #append if chi2 is reasonably small
                    samples.append( [guess, new_chi2] )

            #since we are saving each model tried, get rid of absurdly poor models
            min_chi2=min([it[-1] for it in samples])
            samples=[it for it in samples if it[-1]<min_chi2+10.*chi2_lim]

            T*=alpha
    except KeyboardInterrupt:
        print "user stopped optimization"
        

    samples=sorted(samples, key=lambda x_: x_[-1], reverse=False)
    param_samples=[{key:val[0] 
                    for key, val in samples[i][0].items()} 
                    for i in range(0,len(samples))]
    return np.array(param_samples), np.array([it[1] for it in samples])


def param_locked_SA(fn, *x,  **kwargs):
    """
    function to run simulated annealing with locking parameters of interest
    (see Avni 1976)
    """
    num_steps=int(kwargs.get("num_steps",10)) #num of locked values to test/param
    max_steps=int(kwargs.get("max_steps",400)) #max num iterations for
    chi2_lim=int(kwargs.get("chi2_lim",4))

    param_ranges={k:v for k,v in kwargs.iteritems() if not k in "max_steps chi2_lim".split()}
    param_samples, all_chi2=None,None
    epsilon=1./float(num_steps)
    for param, rng in param_ranges.iteritems():
        #lock each param
        for val in np.linspace(rng[0],rng[1], num_steps):
            #update the parameter and put into kwargs
            new_rng=[val,val]
            kwargs[param]=new_rng
            psamp, chi2=simulated_annealing(fn, *x, **kwargs)
            if all_chi2 is None:
                param_samples, all_chi2=psamp, chi2
            else:
                param_samples=np.concatenate((param_samples, psamp))
                all_chi2=np.concatenate((all_chi2, chi2))
        kwargs[param]=rng
    return param_samples, all_chi2
        

