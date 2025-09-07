import firedrake as fdrk
from .problem import StaticProblem
from src.tools.elasticity import first_piola_neohookean

# Neo Hookean Potentials
# I_1, I_2, I_3 are the principal invariants of the Cauchy Green deformation tensor C = F^T F
# W_1 = mu/2 * (I_1 - 3) - mu/2 * ln I_3 + kappa/2 * (I_3^(1/2) - 1)^2
# W_2 = mu/2 * (I_1 - 3) - mu/2 * ln I_3 + kappa/8 *ln(I_3)^2

# First Piola stress tensor
# P_1 = mu (F - F^{-T}) + kappa (J^2 - J) F^{-T}
# P_2 = mu (F - F^{-T}) + kappa ln(J) F^{-T}

class ConvergenceStatic(StaticProblem):

    def __init__(self, n_elem_x, n_elem_y, quad=False):

        self.domain = fdrk.UnitSquareMesh(n_elem_x, n_elem_y, quadrilateral=quad)
        self.dim = self.domain.topological_dimension()

        self.coordinates_mesh = fdrk.SpatialCoordinate(self.domain)

        self.x, self.y = self.coordinates_mesh
        self.normal_versor = fdrk.FacetNormal(self.domain)

        mu = 1  #N/mm^2
        lamda = 1 #N/mm^2

        self.parameters = {"mu": mu, "lamda": lamda}

    def get_exact_solution(self):

        exact_displacement = fdrk.as_vector([0.5*self.y**3 + 0.5*fdrk.sin(0.5 * fdrk.pi * self.y), 0])
        
        exact_disp_grad = fdrk.grad(exact_displacement)
        exact_first_piola = first_piola_neohookean(exact_disp_grad, parameters=self.parameters)

        return {"displacement" : exact_displacement,
                "disp_grad": exact_disp_grad,
                "first_piola": exact_first_piola}


    def get_essential_bcs(self) -> dict:
        """
        Cantilever beam
        Zero velocity on left boundary 
        Traction along the y axis on the right boundary =
        
        """
        essential_dict = {"displacement x": {3: fdrk.Constant(0)},
                          "displacement y": {3: fdrk.Constant(0)}}
        
        return essential_dict
    

    def get_natural_bcs(self) -> dict:
        first_piola_exact = self.get_exact_solution()["first_piola"]

        traction_x = first_piola_exact[0, :]
        traction_y = first_piola_exact[1, :]

        return {"traction x": {1: traction_x, 2: traction_x, 4: traction_x},
                "traction y": {1: traction_y, 2: traction_y, 4: traction_y}}


    def get_forcing(self):

        first_piola_exact = self.get_exact_solution()["first_piola"]
        exact_forcing = - fdrk.div(first_piola_exact)
        return exact_forcing
    
    
    def __str__(self):
        return "ConvergenceStatic2D"