using WebIO
cmd = WebIO.find_jupyter_cmd(force_conda_jupyter = true)
WebIO.install_jupyter_nbextension(cmd)
