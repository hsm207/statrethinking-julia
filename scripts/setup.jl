using WebIO
jupyter = ENV["JUPYTER"]
WebIO.install_jupyter_nbextension(`$jupyter`)