import compute_rhino3d
from compute_rhino3d import Util
import setupsecrets
secrets=setupsecrets.setup_secrets()
Util.url="http://"+secrets['RHINO_COMPUTE_URL']+":"+secrets['RHINO_COMPUTE_URL']+"/"