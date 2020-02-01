using WebIO
cmd = WebIO.find_jupyter_cmd(force_conda_jupyter=true)
WebIO.install_jupyter_nbextension(cmd)

ENV["R_HOME"] = "C:\\Program Files\\R\\R-3.6.2"
