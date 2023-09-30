import os, sys
import numpy as np
import pandas as pd
from typing import Union

import matplotlib.pyplot as plt
import seaborn as sb

import ipywidgets as widgets
from IPython.display import display

class FinUniformAtr():
    
    def __init__(
            self,
            htc: float,
            k: float,
            Tinf: float,
            Tbase: float
        ):
        """Initiate fin geometric model."""
        
        self._h = htc
        self._k = k
        self._Tinf = Tinf
        self._Tbase = Tbase
        
        self._thetaB = Tbase - Tinf
       
    def mCoeffFin(self):
        return np.sqrt(
                   self._h*self._perimeter/(self._k*self._area_tr)
               )
        
    def MCoeffFin(self):
        return np.sqrt(
                   self._h*self._perimeter*self._k*self._area_tr
               )*self._thetaB

    def _temperatureFinActiveBoundary(
            self
        )   -> float:
    
        # This function must return an array
        xRange = np.linspace(0.0, self._L, 100)
                        
        thetaB = self._Tbase - self._Tinf
        
        m = self.mCoeffFin()
        M = self.MCoeffFin()
        
        arg   = m*(self._L - xRange)
        coeff = self._h/(m*self._k)
        
        thetaRatio = (
                         np.cosh(arg) + coeff*np.sinh(arg)
                     )/(
                         np.cosh(m*self._L) + coeff*np.sinh(m*self._L)
                     )
        
        return self._Tinf + thetaRatio*thetaB
    
    def temperatureDistribution(self):
        """Compute temperature distribution in a planar fin."""
        
        return self._temperatureFinActiveBoundary()

    def heatTransfer(self):
        
        # By using the equivalent length
        # the heat transfer can be approximated
        # by the equation for the adiabatic tip
        return self._M*np.tanh(self._m*self._Lc)
            
    def efficiency(self):
        
        qMax = self._h*self._area_fin*self._thetaB
        
        return self.heatTransfer()/qMax
    
    def effectiveness(self):
        
        qBaseNoFin = self._h*self._area_tr*self._thetaB
        
        return self.heatTransfer()/qBaseNoFin
    
    def getLength(self):
        """Get fin length."""
        
        return self._L
              
    def getPerimeter(self):
        """Get fin perimeter."""
        
        return self._perimeter
    
    def getSurfaceArea(self):
        """Get fin surface area."""
        
        return self._area_fin
    
    def getTrArea(self):
        """Get fin section area."""
        
        return self._area_tr


class PlanarFin(FinUniformAtr):
    
    def __init__(
            self,
            length: float,
            width: float,
            thickness: float,
            htc: float,
            k: float,
            Tinf: float,
            Tbase: float
        ):
        """Initiate fin geometric model."""
        
        FinUniformAtr.__init__(
            self,
            htc=htc,
            k=k,
            Tbase=Tbase,
            Tinf=Tinf
        )
        
        # Rectangular planar fin 
        self._L = length
        self._w = width
        self._t = thickness
        
        # Equivalent length to compute efficiency
        self._Lc = self._L + 0.5*self._t
        
        self._perimeter = 2*self._w + 2*self._t
        self._area_fin  = self._Lc*self._perimeter
        self._area_tr   = self._w*self._t
        self._area_prof = self._t*self._L
        
        # Their computattion depends on the geometry
        self._m = self.mCoeffFin()
        self._M = self.MCoeffFin()

class PiniformFin(FinUniformAtr):
    
    def __init__(
            self,
            length: float,
            diameter: float,
            htc: float,
            k: float,
            Tinf: float,
            Tbase: float
        ):
        """Initiate fin geometric model."""
        
        FinUniformAtr.__init__(
            self,
            htc=htc,
            k=k,
            Tbase=Tbase,
            Tinf=Tinf
        )
            
        # Rectangular piniform fin test
        self._L = length
        self._D = diameter
        
        # Equivalent length to compute efficiency
        self._Lc = self._L + 0.25*self._D
        
        self._perimeter = np.pi*self._D
        self._area_fin  = self._Lc*self._perimeter
        
        self._area_tr   = 0.25*np.pi*self._D**2
        self._area_prof = self._D*self._L
        
        # Their computattion depends on the geometry
        self._m = self.mCoeffFin()
        self._M = self.MCoeffFin()
