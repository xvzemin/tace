import os
import subprocess
import sys
import textwrap


def test_nvalchemi_is_an_isolated_optional_interface():
    script = textwrap.dedent(
        r"""
        import builtins

        original_import = builtins.__import__

        def import_without_nvalchemi(name, *args, **kwargs):
            if name == "nvalchemi" or name.startswith("nvalchemi."):
                raise ModuleNotFoundError("blocked optional dependency")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = import_without_nvalchemi

        import tace
        import tace.models
        from tace.interface.ase import TACEAseCalc

        assert tace is not None
        assert tace.models is not None
        assert TACEAseCalc is not None

        try:
            from tace.interface.nvalchemi import TACEWrapper
        except ImportError as error:
            message = str(error)
            assert "nvalchemi-toolkit" in message
            assert "nvalchemi-toolkit-ops" in message
            assert (
                "pip install nvalchemi-toolkit nvalchemi-toolkit-ops" in message
            )
        else:
            raise AssertionError(TACEWrapper)
        """
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr
