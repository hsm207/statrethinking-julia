using WebIO
using IJulia

jupyter = ENV["JUPYTER"]
WebIO.install_jupyter_nbextension(`$jupyter`)

installkernel("Julia (Turing)", "--project=/workspaces/statrethinking-gen/turing", specname = "julia-turing")