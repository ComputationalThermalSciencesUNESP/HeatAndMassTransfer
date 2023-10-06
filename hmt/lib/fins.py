# Copyright (C) 2023, Iago L. de Oliveira

# HeatAndMassTransfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Collection of tools and classes of fins for heat transfer enhancement.

This module provides an implementation of fins temperature distribution, heat
transfer computation and other properties such as their efficiency and
effectiveness computation.

The function implemented here are based on the analysis and results presented
by Incropera's Fundamentas of Heat and Mass Transfer, 8th ed. All the
computations are based on the hypothesis of an active tip of the fin, i.e. with
convection at the tip.
"""

import numpy as np
from typing import Union

# Import modified Bessel functions
from scipy.special import i0, i1, k0, k1, iv

_SMALL = 1.0e-4

def i2(x):
    """Modified Bessel function of first kind of second order."""
    return iv(2, x)

# def temperatureFinInfLength(
#         x: float,
#         htc: float,
#         k: float,
#         perimeter: float,
#         area_tr: float,
#         Tinf: float,
#         Tbase: float
#     )   -> float:

#     thetaB = diffTemp(Tbase, Tinf)

#     m = mCoeffFin(
#             htc,
#             k,
#             perimeter,
#             area_tr
#         )

#     return Tinf + thetaB*np.exp(-m*x)
# def heatTransferFinActiveBoundary(
#         length: float,
#         htc: float,
#         k: float,
#         perimeter: float,
#         area_tr: float,
#         Tinf: float,
#         Tbase: float
#     )   -> float:

#     m = mCoeffFin(
#             htc,
#             k,
#             perimeter,
#             area_tr
#         )

#     M = MCoeffFin(
#             htc,
#             k,
#             perimeter,
#             area_tr,
#             Tinf,
#             Tbase
#         )

#     coeff = htc/(m*k)

#     return M*(
#                 np.sinh(m*length) + coeff*np.cosh(m*length)
#            )/(
#                 np.cosh(m*length) + coeff*np.sinh(m*length)
#            )

# def heatTransferFinInfLength(
#         htc: float,
#         k: float,
#         perimeter: float,
#         area_tr: float,
#         Tinf: float,
#         Tbase: float
#     )   -> float:


#     M = MCoeffFin(
#             htc,
#             k,
#             perimeter,
#             area_tr,
#             Tinf,
#             Tbase
#         )

#     return M

class Fin():

    def __init__(
            self,
            htc: float,
            k: float,
            Tinf: float,
            Tbase: float
        ):
        """Initiate model of a fin."""

        self._h = htc
        self._k = k
        self._Tinf = Tinf
        self._Tbase = Tbase

        self._thetaB = Tbase - Tinf

    def getmCoeffFin(self):
        return self._m

    def getLength(self):
        """Get fin length."""

        return self._L

    def getOperationalConditions(self):
        """Get the operation conditions of a fin.

        Given the fin, return the heat transfer coefficient, base temperature,
        flow temperature, and thermal conductivity. Return a dictionary.
        """

        return {"htc": self._h,
                "k": self._k,
                "Tinf": self._Tinf,
                "Tbase": self._Tbase}

    def getProfileArea(self):
        """Return fin profile area."""

        return self._area_prof

    def getBaseArea(self):
        """Return fin base area."""

        return self._area_base

    def getSurfaceArea(self):
        """Get fin surface area."""

        return self._area_fin

    def getVolume(self):
        """Return fin volume."""

        return self._volume

    def getMaxHeatTransfer(self):
        return self._h*self._area_fin*self._thetaB

    def getThermalResistance(self):

        return 1.0/(self._h*self._area_fin*self.getEfficiency())

class FinUniformAtr(Fin):

    def __init__(
            self,
            length: float,
            htc: float,
            k: float,
            Tinf: float,
            Tbase: float,
            area_tr: float=None,
            perimeter: float=None
        ):
        """Initiate model of fin with uniform section area."""

        Fin.__init__(
            self,
            htc=htc,
            k=k,
            Tbase=Tbase,
            Tinf=Tinf
        )

        self._L = length

        if area_tr is not None and perimeter is not None:
            self._area_tr   = area_tr
            self._perimeter = perimeter
            self._area_base = self._area_tr
            self._volume    = self._area_base*self._L

            self._area_fin  = self._perimeter*self._L + self._area_base

            # Their computattion depends on the geometry
            self._m = self._mCoeffFin()
            self._M = self._MCoeffFin()

    def _check_x(self, x):

        if x < 0.0 or x > self._L:
            raise ValueError(
                      "Fin longitudinal position, x, must be smaller than "\
                      "its length."
                  )

    def _mCoeffFin(self):
        return np.sqrt(
                   self._h*self._perimeter/(self._k*self._area_tr)
               )

    def _MCoeffFin(self):
        return np.sqrt(
                   self._h*self._perimeter*self._k*self._area_tr
               )*self._thetaB

    def getPerimeter(self):
        """Get fin perimeter."""

        return self._perimeter

    def getProfileArea(self):
        raise NotImplementedError(
                "Profile area not implemented for generic-profile fin."
            )

    def getTrArea(self):
        """Get fin section area."""

        return self._area_tr

    def getTemperature(self, x):
        """Compute temperature distribution in a planar fin.

        The computation here is based on the active boundary of a uniform fin.
        """

        self._check_x(x)

        arg   = self._m*(self._L - x)
        coeff = self._h/(self._m*self._k)

        thetaRatio = (
                         np.cosh(arg) + coeff*np.sinh(arg)
                     )/(
                            np.cosh(self._m*self._L)
                         +  coeff*np.sinh(self._m*self._L)
                     )

        return self._Tinf + thetaRatio*self._thetaB

    def getHeatTransfer(self):

        # By using the equivalent length
        # the heat transfer can be approximated
        # by the equation for the adiabatic tip
        return self._M*np.tanh(self._m*self._Lc)

    def getEfficiency(self):
        """Compute efficiency of a fin."""

        return self.getHeatTransfer()/self.getMaxHeatTransfer()

    def getEffectiveness(self):
        """Return fin's effectiveness."""

        qBaseNoFin = self._h*self._area_base*self._thetaB

        return self.getHeatTransfer()/qBaseNoFin

class FinNonUniformAtr(Fin):

    def __init__(
            self,
            htc: float,
            k: float,
            Tinf: float,
            Tbase: float
        ):
        """Initiate geometric model of fin with non-uniform section area."""

        Fin.__init__(
            self,
            htc=htc,
            k=k,
            Tbase=Tbase,
            Tinf=Tinf
        )

    def getPerimeter(self, x):
        """Return fin perimeter, given longitudinal position.

        Given a longitudinal position along the length of the fin, return the
        section perimeter. The longitudinal position depends on the fin type:
        an x value of rectangular or piniform fins or a radius for annular
        fins.
        """

        return self._perimeter(x)

    def getTrArea(self, x):
        """Transversal area of annular fin based on longitudinal position.

        Given a longitudinal position along the length of the fin, return the
        section area. The longitudinal position depends on the fin type: an x
        value of rectangular or piniform fins or a radius for annular fins.
        """

        return self._area_tr(x)

    def getTemperature(
            self,
            x
        ):
        """Temperature distribution in an annular fin.

        Return the temperature in an annular fin at a longitudinal position.
        This function is given for the active fin tip (convective boundary
        condition) using the solution of the adiabatic tip with an equivalent
        external radius.  The longitudinal position depends on the fin type: an
        x value of rectangular or piniform fins or a radius for annular fins.
        """
        # Must be in derived class for non-uniform fins
        raise NotImplementedError(
                "You must implement a getTemperature() down the derived class."
            )

    def getHeatTransfer(self):
        """Return fin's total heat transfer.

        For non-uniform section fins, the heat transfer is computed based on
        the efficiency, with the exception of annular fins for which the heat
        transfer was computed explicitly.
        """

        # For fins that only the efficiency is available, compute heat transfer
        # based on it

        return self.getMaxHeatTransfer()*self.getEfficiency()

    def getEffectiveness(self):
        """Return fin's effectiveness.

        For non-uniform section fins, the effectiviness is computed based on
        the efficiency, with the exception of annular fins for which the heat
        transfer was computed explicitly.
        """

        return self.getEfficiency()*(self._area_fin/self._area_base)

    def getEfficiency(self):
        raise NotImplementedError(
                "You must implement a getEfficiency() down the derived class."
            )

class FinPlanarRectangular(FinUniformAtr):

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
        """Initiate geometric model of a rectangular planar fin."""

        FinUniformAtr.__init__(
            self,
            length=length,
            htc=htc,
            k=k,
            Tbase=Tbase,
            Tinf=Tinf
        )

        # Rectangular planar fin
        self._w = width
        self._t = thickness

        # Equivalent length to compute efficiency
        self._Lc = self._L + 0.5*self._t

        self._perimeter = 2*self._w + 2*self._t

        # For uniform section area fins, use the Lc to
        # compute the area (this includes the tip area)
        self._area_fin  = 2*self._w*self._Lc + 2*self._L*self._t
        self._area_tr   = self._w*self._t
        self._area_base = self._area_tr
        self._area_prof = self._t*self._L
        self._volume = self._area_base*self._L

        # Their computattion depends on the geometry
        self._m = self._mCoeffFin()
        self._M = self._MCoeffFin()

class FinPiniformRectangular(FinUniformAtr):

    def __init__(
            self,
            length: float,
            diameter: float,
            htc: float,
            k: float,
            Tinf: float,
            Tbase: float
        ):
        """Initiate geometric model of a piniform rectangular fin."""

        FinUniformAtr.__init__(
            self,
            length=length,
            htc=htc,
            k=k,
            Tbase=Tbase,
            Tinf=Tinf
        )

        # Rectangular piniform fin test
        self._D = diameter

        # Equivalent length to compute efficiency
        self._Lc = self._L + 0.25*self._D

        self._perimeter = np.pi*self._D

        # For uniform section area fins, use the Lc to
        # compute the area (this includes the tip area)
        self._area_fin  = self._Lc*self._perimeter
        self._area_tr   = 0.25*np.pi*self._D**2
        self._area_base = self._area_tr
        self._area_prof = self._D*self._L
        self._volume = self._area_base*self._L

        # Their computattion depends on the geometry
        self._m = self._mCoeffFin()
        self._M = self._MCoeffFin()

    def getProfileArea(self):

        raise NotImplementedError(
                  "Profile area undefined for piniform fins."
              )

class FinAnnular(FinNonUniformAtr):

    def __init__(
            self,
            inner_radius: float,
            outer_radius: float,
            thickness: float,
            htc: float,
            k: float,
            Tinf: float,
            Tbase: float
        ):
        """Initiate geometric model of an annular fin."""

        FinNonUniformAtr.__init__(
            self,
            htc=htc,
            k=k,
            Tinf=Tinf,
            Tbase=Tbase
        )

        # Rectangular piniform fin test
        self._ri = inner_radius
        self._re = outer_radius
        self._t  = thickness
        self._L  = outer_radius - inner_radius
        self._Lc = self._L + 0.5*thickness

        # Equivalent radius
        self._rec = self._re + 0.5*self._t

        # Areas
        self._area_fin  = 2*np.pi*(self._rec**2 - self._ri**2)
        self._area_base = 2*np.pi*(self._ri**2)*self._t
        self._area_prof = self._t*self._Lc
        self._volume    = np.pi*(self._re**2 - self._ri**2)*self._t

        # Their computattion depends on the geometry
        self._m = self._mCoeffFin()

    def _mCoeffFin(self):
        # this version is specific of annular fins
        # it independs on the radius
        return np.sqrt(
                   2.0*self._h/(self._k*self._t)
               )

    def _check_radius(self, radius):

        if radius < self._ri or radius > self._re:
            raise ValueError(
                      "Fin radius must be between inner and outer radius."
                  )
        else:
            pass

    def _perimeter(self, radius):

        self._check_radius(radius)

        return 4.0*np.pi*radius

    def _area_tr(self, radius):

        self._check_radius(radius)

        return 2.0*np.pi*radius*self._t

    def getTemperature(
            self,
            radius
        ):
        """Temperature distribution in an annular fin.

        Return the temperature distribution in an annular fin as a NumPy array.
        This function is given for the active fin tip (convective boundary
        condition) using the solution of the adiabatic tip with an equivalent
        external radius.
        """

        self._check_radius(radius)

        thetaRatio = (
                           i0(self._m*radius)*k1(self._m*self._rec)
                         + k0(self._m*radius)*i1(self._m*self._rec)
                     )/(
                           i0(self._m*self._ri)*k1(self._m*self._rec)
                         + k0(self._m*self._ri)*i1(self._m*self._rec)
                     )

        return self._Tinf + thetaRatio*self._thetaB

    def getEfficiency(self):
        """Return fin's efficiency."""

        # Explicit computation based on solution provided in the text
        c2 = (2*self._ri/self._m)/(self._rec**2 - self._ri**2)

        return c2*(
                     k1(self._m*self._ri)*i1(self._m*self._rec)
                   - i1(self._m*self._ri)*k1(self._m*self._rec)
               )/(
                     k0(self._m*self._ri)*i1(self._m*self._rec)
                   + i0(self._m*self._ri)*k1(self._m*self._rec)
               )

class FinPlanarTriangular(FinNonUniformAtr):

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
        """Initiate geometric model of a planar triangular fin."""

        FinNonUniformAtr.__init__(
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

        # Areas
        self._area_fin  = 2*self._w*np.sqrt(
                                self._L**2 + 0.25*self._t**2
                            )

        self._area_prof = 0.5*self._t*self._L
        self._volume = self._area_prof*self._w

        # Their computattion depends on the geometry
        # These two have to be defined here
        self._m = self._mCoeffFin()

    def _mCoeffFin(self):
        # this version is specific of rectangular fins
        # with width >> thickness
        return np.sqrt(
                   2.0*self._h/(self._k*self._t)
               )

    def _check_x(self, x):

        if x < 0.0 or x > self._L:
            raise ValueError(
                      "Fin longitudinal position, x, must be smaller than "\
                      "its length."
                  )

    def _thickness(self, x):
        self._check_x(x)

        return self._t*(1.0 - x/self._L)

    def _perimeter(self, x):
        return 2*self._w + 2*self._thickness(x)

    def _area_tr(self, x):
        return self._w*self._thickness(x)

    def getTemperature(
            self,
            x
        ):
        raise NotImplementedError(
                  "Temperature field not implemented for triangular fin."
              )

    def getEfficiency(self):
        """Compute efficiency of triangular planar fin."""

        return (1.0/(self._m*self._L))*(
                    i1(2*self._m*self._L)/i0(2*self._m*self._L)
                )

class FinPlanarParabolic(FinNonUniformAtr):

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
        """Initiate geometric model of a planar parabolic fin."""

        FinNonUniformAtr.__init__(
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

        # Areas
        c1 = np.sqrt(
                 1.0 + (self._t/self._L)**2
             )

        self._area_fin  = self._w*(
                              c1*self._L
                            + (self._L**2/self._t)*np.log(
                                  c1 + self._t/self._L
                              )
                          )

        self._area_prof = (1.0/3.0)*self._t*self._L
        self._volume    = self._area_prof*self._w

        # Their computattion depends on the geometry
        # These two have to be defined here
        self._m = self._mCoeffFin()

    def _mCoeffFin(self):
        # this version is specific of rectangular fins
        # with width >> thickness
        return np.sqrt(
                   2.0*self._h/(self._k*self._t)
               )

    def _check_x(self, x):

        if x < 0.0 or x > self._L:
            raise ValueError(
                    "Fin longitudinal position, x, must be smaller then its"\
                    " length."
                  )

    def _thickness(self, x):
        self._check_x(x)

        return self._t*(1.0 - x/self._L)**2

    def _perimeter(self, x):
        return 2*self._w + 2*self._thickness(x)

    def _area_tr(self, x):
        return self._w*self._thickness(x)

    def getTemperature(
            self,
            x
        ):
        raise NotImplementedError(
                  "Temperature field not implemented for parabolic fin."
              )

    def getEfficiency(self):
        """Compute efficiency of parabolic planar fin."""

        return 2.0/(1.0 + np.sqrt(4.0*(self._m*self._L)**2 + 1.0))

class FinPiniformTriangular(FinNonUniformAtr):

    def __init__(
            self,
            length: float,
            diameter: float,
            htc: float,
            k: float,
            Tinf: float,
            Tbase: float
        ):
        """Initiate geometric model of a piniform triangular fin."""

        FinNonUniformAtr.__init__(
            self,
            htc=htc,
            k=k,
            Tbase=Tbase,
            Tinf=Tinf
        )

        # Triangular piniform fin test
        self._L = length
        self._D = diameter

        self._area_fin  = 0.5*np.pi*self._D*np.sqrt(
                                  self._L**2
                              +   0.25*self._D**2
                          )

        self._area_base = 0.25*np.pi*self._D**2
        self._volume = (1.0/3.0)*self._area_base*self._L

        # Their computattion depends on the geometry
        # These two have to be defined here
        self._m = self._mCoeffFin()

    def _mCoeffFin(self):
        return np.sqrt(
                   4.0*self._h/(self._k*self._D)
               )

    def _check_x(self, x):

        if x < 0.0 or x > self._L:
            raise ValueError(
                      "Fin longitudinal position, x, must be smaller than "\
                      "its length."
                  )

    def _section_diameter(self, x):
        self._check_x(x)

        return self._D*(1.0 - x/self._L)

    def _perimeter(self, x):
        return np.pi*self._section_diameter(x)

    def _area_tr(self, x):
        return 0.25*np.pi*self._section_diameter(x)**2

    def getProfileArea(self):

        raise NotImplementedError(
                  "Profile area undefined for piniform fins."
              )

    def getTemperature(
            self,
            x
        ):
        raise NotImplementedError(
                  "Temperature field not implemented for triangular fin."
              )

    def getEfficiency(self):
        """Compute efficiency of triangular piniform fin."""

        return (2.0/(self._m*self._L))*(
                    i2(2*self._m*self._L)/i1(2*self._m*self._L)
                )

class FinPiniformParabolic(FinNonUniformAtr):

    def __init__(
            self,
            length: float,
            diameter: float,
            htc: float,
            k: float,
            Tinf: float,
            Tbase: float
        ):
        """Initiate geometric model of a piniform parabolic fin."""

        FinNonUniformAtr.__init__(
            self,
            htc=htc,
            k=k,
            Tbase=Tbase,
            Tinf=Tinf
        )

        # Triangular piniform fin test
        self._L = length
        self._D = diameter

        c3 = 1.0 + 2*(self._D/self._L)**2
        c4 = np.sqrt(1.0 + (self._D/self._L)**2)

        self._area_fin  = (np.pi*(self._L**3)/(8*self._D))*(
                                c3*c4
                            -   self._L/(2*self._D)*np.log(
                                    c3 + 2*self._D*c4/self._L
                                )
                          )

        self._area_base = 0.25*np.pi*self._D**2
        self._volume = (1.0/5.0)*self._area_base*self._L

        # Their computattion depends on the geometry
        # These two have to be defined here
        self._m = self._mCoeffFin()

    def _mCoeffFin(self):
        return np.sqrt(
                   4.0*self._h/(self._k*self._D)
               )

    def _check_x(self, x):

        if x < 0.0 or x > self._L:
            raise ValueError(
                      "Fin longitudinal position, x, must be smaller than "\
                      "its length."
                  )

    def _section_diameter(self, x):
        self._check_x(x)

        return self._D*(1.0 - x/self._L)**2

    def _perimeter(self, x):
        return np.pi*self._section_diameter(x)

    def _area_tr(self, x):
        return 0.25*np.pi*self._section_diameter(x)**2

    def getProfileArea(self):

        raise NotImplementedError(
                  "Profile area undefined for piniform fins."
              )

    def getTemperature(
            self,
            x
        ):
        raise NotImplementedError(
                  "Temperature field not implemented for triangular fin."
              )

    def getEfficiency(self):
        """Compute efficiency of parabolic piniform fin."""

        return 2.0/(1.0 + np.sqrt((4.0/9.0)*(self._m*self._L)**2 + 1.0))

class FinArray():
    """Model of fin arrays.

    Class representing a specific fin array with a provided fin object, its
    number, and prime surface dimensions.
    """

    def __init__(
            self,
            fin_object,
            prime_area,
            nfins
        ):
        """Initiate model of a fin array."""

        self._fin = fin_object
        self._prime_area = prime_area
        self._nfins = nfins

        self._fin_area = self._fin.getSurfaceArea()

    def getTotalArea(self):
        """Compute total area of finned surface."""

        return self._nfins*self._fin_area + self._prime_area

    def getEfficiency(self):
        """Compute efficiency of fin array."""

        etaf = self._fin.getEfficiency()
        At   = self.getTotalArea()
        Af   = self._fin_area
        N    = self._nfins

        return 1 - (N*Af/At)*(1 - etaf)

    def getHeatTransfer(self):
        """Compute total heat transfer rate of fin array."""

        dictOpCond = self._fin.getOperationalConditions()

        htc     = dictOpCond["htc"]
        thetaB  = dictOpCond["Tbase"] - dictOpCond["Tinf"]
        At      = self.getTotalArea()
        etao    = self.getEfficiency()

        return etao*htc*At*thetaB
