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

"""Test the fins.py."""

import os
import sys
import unittest

import numpy as np
from lib import fins


class TestFinsModule(unittest.TestCase):

    def test_PiniformRectangularFin(self):
        """Test based on exercise form Incropera, 8th ed."""

        kCu = 398.0 # W/mK
        kAl = 180.0 # W/mK
        kSteel = 14.0 # W/mK

        # Dados do problema
        exHtc   = 100.0 # W/m2K
        exTbase = 100.0 # Celsius
        exTinf  = 25.0 # Celsius

        # Dimensions
        exFinRadius = 2.5e-3 # meters
        finLength = 0.3 # meters

        # SOlution of heat transfer per material
        qCu = 8.3 # W
        qAl = 5.6 # W
        qSteel = 1.6 # W

        LCu = 0.19 # meter
        LAl = 0.13 # meter
        LSteel = 0.04 # meter

        print("\n")
        for (k, q, Linf), metal in zip([(kCu, qCu, LCu),
                                        (kAl, qAl, LAl),
                                        (kSteel, qSteel, LSteel)],
                                        ["Cu", "Al", "Steel"]):

            pinFin = fins.FinPiniformRectangular(
                          length=finLength,
                          diameter=2*exFinRadius,
                          htc=exHtc,
                          k=k,
                          Tinf=exTinf,
                          Tbase=exTbase
                     )

            print(
                f"Checking for metal: {metal}",
                end="\n"
            )

            self.assertTrue(
                round(pinFin.getHeatTransfer(), 1) == q
            )

            # Check infinite length
            self.assertTrue(
                round(2.65/pinFin._mCoeffFin(), 2) == Linf
            )

    def test_OtherFinn(self):

        kCu = 398.0 # W/mK
        kAl = 180.0 # W/mK
        kSteel = 14.0 # W/mK

        # Dados do problema
        exHtc   = 100.0 # W/m2K
        exTbase = 100.0 # Celsius
        exTinf  = 25.0 # Celsius

        # Dimensions
        exFinRadius = 2.5e-3 # meters
        finLength = 0.3 # meters

        # Adding a similar planar fin
        exFinWidth = 0.20
        exFinThickness = 0.01

        triFin = fins.FinPlanarTriangular(
                      length=finLength,
                      width=exFinWidth,
                      thickness=exFinThickness,
                      htc=exHtc,
                      k=kCu,
                      Tinf=exTinf,
                      Tbase=exTbase
                 )

        self.assertTrue(
            triFin.getLength() == finLength
        )

        self.assertTrue(
            triFin.getPerimeter(finLength) == 2*exFinWidth
        )

if __name__=='__main__':
    unittest.main()
