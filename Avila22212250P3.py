# Modulos (consola) y libreria para sistemas de control
#!pip install control
#!pip install slycot
import control as ctrl

# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd


x0,t0,tend,dt,w,h = 0,0,10,1E-3,7,3.5
N = round((tend - t0)/dt) + 1
t = np.linspace (t0,tend,N)
u = np.zeros(N)
u[round(1 / dt):round(2 / dt)] = 1  # Impulso

def musc(Cs,Cp,R,a):
    num=[R*Cs, 1 - a]
    den=[R*(Cp + Cs), 1]
    sys = ctrl.tf(num,den)
    return sys

#Funcion de transferencia: Control
Cs,Cp,R,a =10E-6,100E-6,100,0.25
syscon = musc(Cs,Cp,R,a)  
print(f'Funcion de transferencia del normotenso: {syscon}')

#Funcion de transferencia: Caso
Cs,Cp,R,a =10E-6,100E-6,10E3,0.25
syscaso = musc(Cs,Cp,R,a)  
print(f'Funcion de transferencia del hipotenso: {syscaso}')



_,Fs1 = ctrl.forced_response(syscon,t,u,x0)
_,Fs2 = ctrl.forced_response(syscaso,t,u,x0)

fgl =plt.figure()
plt.plot(t,u,'-',linewidth = 1, color = [0.569,0.392,0.235],label = 'Fs1(t): Impulso')
plt.plot(t,Fs1,'-',linewidth = 1, color = [0.902,0.224,0.274],label = 'Fs2(t): Control')
plt.plot(t,Fs2,'-',linewidth = 1, color = [0.114,0.208,0.341],label = 'Fs3(t): Caso')
plt.grid(False)
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4);plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('t [s]', fontsize = 11)
plt.ylabel('F(t) [V]', fontsize = 11)
plt.legend(bbox_to_anchor = (0.5,-0.3),loc = 'center', ncol = 3,
           fontsize = 9, frameon = True)
plt.show()
fgl.set_size_inches(w,h)
fgl.tight_layout()
fgl.savefig('Sistema Musculoesqueletico.png',dpi=600, bbox_inches ='tight')
fgl.savefig('Sistema Musculoesqueletico.pdf')



def controlador (kP,kI,sys):
    Cr = 1E-6
    Re = 1/(kI*Cr)
    Rr = kP*Re
    numPI = [Rr*Cr,1]
    denPI = [Re*Cr,0]
    PI = ctrl.tf(numPI,denPI)
    X = ctrl.series(PI, sys)
    sysPI = ctrl.feedback(X,1,sign=-1)
    return sysPI

hipoPI = controlador (0.0215736670168563,42064.8477577268,syscaso)


_,Fs3 = ctrl.forced_response(hipoPI,t,Fs1,x0)

fg2 = plt.figure()
plt.plot(t,u,'-',linewidth = 1, color = [0.569,0.392,0.235],label = 'Fs1(t): Impulso')
plt.plot(t,Fs1,'-',linewidth = 1, color = [0.902,0.224,0.274],label = 'Fs2(t): Control')
plt.plot(t,Fs2,'-',linewidth = 1, color = [0.114,0.208,0.341],label = 'Fs3(t): Caso')
plt.plot(t,Fs3,'--',linewidth = 1.5, color = [0.114,0.208,0.341],label = 'Fs4(t): Controlador')
plt.grid(False)
plt.xlim(0,10);plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4);plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('t [s]', fontsize = 11)
plt.ylabel('F(t) [V]', fontsize = 11)
plt.legend(bbox_to_anchor = (0.5,-0.3),loc = 'center', ncol = 3,
           fontsize = 9, frameon = True)
plt.show()
fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('Sistema MusculoesqueleticoPI.png',dpi=600, bbox_inches ='tight')
fg2.savefig('Sistema MusculoesqueleticoPI.pdf')



