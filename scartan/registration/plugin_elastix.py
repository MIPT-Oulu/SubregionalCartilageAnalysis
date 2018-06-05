import os
from pathlib import Path
import subprocess
import logging


logging.basicConfig()
logger = logging.getLogger('registration')
logger.setLevel(logging.DEBUG)


class Elastix(object):
    def __init__(self, path_root):
        """

        Args:
            path_root:
        """
        self._dir_exec = Path(path_root, 'bin')
        self._path_exec_elastix = Path(self._dir_exec, 'elastix')
        self._path_exec_transformix = Path(self._dir_exec, 'transformix')
        self._dir_lib = Path(path_root, 'lib')

    def fit(self, path_scan_fixed, path_scan_moving, path_param_file, path_root_out,
            num_threads=12):

        if not os.path.exists(path_root_out):
            os.makedirs(path_root_out, exist_ok=True)

        cmd = [self._path_exec_elastix,
               '-f', path_scan_fixed,
               '-m', path_scan_moving,
               '-p', path_param_file,
               '-out', path_root_out,
               '-threads', str(num_threads)]

        # Run in a subprocess to hide the large output
        proc = subprocess.run(cmd, check=True,
                              env=dict(os.environ, LD_LIBRARY_PATH=self._dir_lib),
                              stdout=subprocess.PIPE, encoding='utf8')

        if "Total time elapsed" in proc.stdout:
            logger.info(f"Registration successful: {path_scan_moving}")
            return 0
        else:
            logger.info(f"Registration failed: {path_scan_moving}")
            return 1

    def predict(self, path_scan, path_root_out, path_transf, num_threads=12):

        if not os.path.exists(path_root_out):
            os.makedirs(path_root_out, exist_ok=True)

        cmd = [self._path_exec_transformix,
               '-in', path_scan,
               '-out', path_root_out,
               '-tp', path_transf,
               '-threads', str(num_threads)]

        # Run in a subprocess to hide the large output
        proc = subprocess.run(cmd, check=True,
                              env=dict(os.environ, LD_LIBRARY_PATH=self._dir_lib),
                              stdout=subprocess.PIPE, encoding='utf8')

        if "Total time elapsed" in proc.stdout:
            logger.info(f"Transformation successful: {path_scan}")
            return 0
        else:
            logger.info(f"Transformation failed: {path_scan}")
            return 1

    def fit_predict(self, path_scan_fixed, path_scan_moving, path_param_file,
                    path_root_out):

        # Estimate the warping transform
        ret = self.fit(path_scan_fixed=path_scan_fixed,
                       path_scan_moving=path_scan_moving,
                       path_param_file=path_param_file,
                       path_root_out=path_root_out)
        if ret != 0:
            return ret

        # Find the transformation parameters file
        path_transf = Path(path_root_out).glob('TransformParameters.*.txt')
        path_transf = sorted(path_transf)[-1]

        # Warp the atlas mask
        ret = self.predict(path_scan=path_scan_moving,
                           path_root_out=path_root_out,
                           path_transf=path_transf)
        if ret != 0:
            return ret

        res = Path(path_root_out).glob('result.*')
        return res
