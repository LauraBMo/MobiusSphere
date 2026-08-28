module MobiusSphereNemoExt

using MobiusSphere
import Nemo: CalciumFieldElem, complex_normal_form

# Simplify Nemo CalciumField expressions; triggered when Nemo is loaded.
MobiusSphere.__normalize(z::CalciumFieldElem) = complex_normal_form(z)

end
