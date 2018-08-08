#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 17:48:18 2018

@author: haiyan
"""
"""
2 2 "G_surface"
2 3 "surface_1"
2 4 "surface_2"
3 1 "G_volume"
"""


from fenics import *
set_log_level(ERROR)
pwd='/home/osmanabu/Downloads/Assignment_2/'
mesh = Mesh(pwd+'Mesh_1.xml')
mesh.order()
D = mesh.topology().dim()
#xL = 25.0
#yL = 180.0
#zL = 5.0

V = VectorFunctionSpace(mesh, 'P', 1)
T = TensorFunctionSpace(mesh, 'P', 1)

cells = MeshFunction('size_t', mesh , 'Mesh_1_physical_region.xml')
facets = MeshFunction('size_t', mesh ,'Mesh_1_facet_region.xml')
dA = Measure('ds', domain=mesh, subdomain_data=facets)
dV = Measure('dx', domain=mesh, subdomain_data=cells)

left = CompiledSubDomain('near(x[1],0) && on_boundary')
right = CompiledSubDomain('near(x[1],180.0) && on_boundary')

facets.set_all(0)
left.mark(facets, 1)
period = 120.
displacement = Expression(('0.0','0.001*sin(2.*pi*f*time)','0.0'), f=1./period, time=0, degree=2)

#bc1 = [DirichletBC(V, (0.,0.,0.), left)]
#bc2 = [DirichletBC(V, displacement, right)]
bc = [DirichletBC(V , (0.,0.,0.) , left )]
f_gr = Constant((0.,0.,0.))

du = TrialFunction(V)
delu = TestFunction(V)
u00 = Function(V)
u0 = Function(V)
u = Function(V)
S = Function(T)
S0 = Function(T)

print ('initializing, time ')
t = 0.0
tend = period
dt = 2

init = Expression(('0','0','0'), degree=2)
u.interpolate(init)
u0.assign(u)
u00.assign(u0)

print ('initializing, space')
rho0 = 9000.0E-15 #g/mikrometer^3
lambada = 90.0 #mN/mikrometer^2 (GPa)
E1, E2 = 200.0, 200.0 # mN/mikrometer^2 (GPa)
mu = 2.0E5 #mN ms / mikrometer^2 (N s/mm^2)

i,j,k,r = indices(4)
delta = Identity(3)

eps = as_tensor(1.0/2.0*(u[i].dx(k)+u[k].dx(i)), [i,k])
eps0 = as_tensor(1.0/2.0*(u0[i].dx(k)+u0[k].dx(i)), [i,k])

eps_dev= as_tensor(eps[i,j]-eps[k,k]*1.0/3.0*delta[i,j], [i,j])
eps0_dev= as_tensor(eps0[i,j]-eps0[k,k]*1.0/3.0*delta[i,j], [i,j])

devS0= as_tensor(S0[i,j]-S0[k,k]*1.0/3.0*delta[i,j], [i,j])

S = as_tensor(lambada*eps[j,j]*delta[i,k] + mu/(E2*dt+mu)*devS0[i,k] + E1*E2*dt/(E2*dt+mu)*eps_dev[i,k] +mu*(E1+E2)/(E2*dt+mu)*(eps_dev[i,k]-eps0_dev[i,k]), [i,k])

N = FacetNormal(mesh)

Form = (rho0/dt/dt*(u-2.*u0+u00)[i]*delu[i]+S[k,i]*delu[i].dx(k) - rho0*f_gr[i]*delu[i])*dV

nz = as_tensor([0.0,0.0,1.0])
#forceZ = as_tensor(F[i,k]*S[r,k]*nz[r], [i,])

Gain = derivative(Form, u, du)

fie_u = File(pwd+'displacement.pvd')

#file_list = ...
time_values=[0.0]
#forcesZ_values=[0.0]
stresses = []
strains = []

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pylab
pylab.rc('text', usetex=True )
pylab.rc('font', family='serif', serif='cm', size=30 )
pylab.rc('legend', fontsize=30 )
pylab.rc(('xtick.major','ytick.major'), pad=15)

#pylab.ion()
fig = pylab.figure(1, figsize= (12,8) )
fig.clf()
pylab.subplots_adjust(bottom=0.18)
pylab.subplots_adjust(left=0.16)
pylab.xlabel(r'Stain')
pylab.ylabel(r'Stress')
pylab.grid(True)

tic()
while t<tend:
    displacement.time = t
    t += dt
    print ('time: ', t, 'indent: ', 'in ', toc(), 'seconds')
    tic()
    
    solve(Form==0, u, bc, J=Gain, solver_parameters={"newton_solver":
        {"linear_solver":"cg", "preconditioner": "hyper_amg", "relative_tolerance":
            1E-2, "absolute_tolerance": 1E-5, "maximum_iterations": 30} },
        form_compiler_parameters={"cpp_optimize": True, "representation": "quadrature", "quadrature_degree": 2} )
    
    file_u << (u,t)
    time_values.append(t)
 #   fZ = project(forceZ, V)
  #  fZvalue = abs(fZ((0,0,0))[2])
   # forceZ_values.append(fZvalue)
    
    stresses.append(eps)
    strains.append(S)
    S0.assign(project(S,T))
    u00.assign(u0)
    u0.assign(u)
  
    pylab.plot(strains, stresses, 'ro-')
    pylab.savefig(pwd+'Try22.pdf')
