import firedrake as fdrk


def stiffness_tensor(strain, young_modulus, poisson_ratio):
    dim = strain.ufl_shape[0]

    stress = young_modulus/(1+poisson_ratio)*\
            (strain + poisson_ratio/(1-2*poisson_ratio)*fdrk.Identity(dim)*fdrk.tr(strain))

    return stress 


def compliance_tensor(stress, young_modulus, poisson_ratio):
    dim = stress.ufl_shape[0]
    # Compliance tensor for generic dimensions
    strain = 1 /(young_modulus) * ((1+poisson_ratio)*stress \
                                           - poisson_ratio * fdrk.Identity(dim) * fdrk.tr(stress))
    return strain


def def_gradient(vector):
    dim = vector.ufl_shape[0]
    return fdrk.Identity(dim) + fdrk.grad(vector)


def green_lagrange_strain(vector):
    return 1/2*(fdrk.grad(vector).T + fdrk.grad(vector) + fdrk.dot(fdrk.grad(vector).T, fdrk.grad(vector)))


def first_piola_neohookean(grad_disp, parameters, dim = 2):
    mu = parameters["mu"]
    lamda = parameters["lamda"]
    def_grad = fdrk.Identity(dim) + grad_disp
    inv_F_transpose = fdrk.inv(def_grad).T
    return mu*(def_grad - inv_F_transpose) + lamda * fdrk.ln(fdrk.det(def_grad)) * inv_F_transpose


def second_piola_neohookean(green_strain, parameters, dim = 2):
    mu = parameters["mu"]
    lamda = parameters["lamda"]
    inv_cauchy_strain = fdrk.inv(green_strain)
    return mu * (fdrk.Identity(dim) - inv_cauchy_strain) \
            + lamda/2 * fdrk.ln(fdrk.det(green_strain))*inv_cauchy_strain


def derivative_first_piola_neohookean(tensor, grad_disp, parameters, dim = 2):
    mu = parameters["mu"]
    lamda = parameters["lamda"]
    def_grad = fdrk.Identity(dim) + grad_disp
    invF = fdrk.inv(def_grad)
    inv_Ftr = fdrk.inv(def_grad).T

    return mu * tensor + (mu - lamda * fdrk.ln(fdrk.det(def_grad))) \
            * fdrk.dot(inv_Ftr, fdrk.dot(tensor.T, inv_Ftr)) \
            + lamda * fdrk.tr(fdrk.dot(invF, tensor)) * inv_Ftr


def natural_control_follower(test, displacement, traction_data_dict : dict):
    F = def_gradient(displacement)

    natural_control = 0 

    if traction_data_dict:

        for item in traction_data_dict.items():
            if item[0]=="follower":
                pass
            else:
                id = item[0]
                value_traction = item[1]


                if traction_data_dict["follower"]:
                    if id == "on_boundary":
                        natural_control +=fdrk.inner(test, fdrk.dot(F, value_traction))*fdrk.ds
                    else: 
                        natural_control +=fdrk.inner(test, fdrk.dot(F, value_traction))*fdrk.ds(id)
                else:
                    if id == "on_boundary":
                        natural_control +=fdrk.inner(test, value_traction)*fdrk.ds
                    else: 
                        natural_control +=fdrk.inner(test, value_traction)*fdrk.ds(id)

    return natural_control

