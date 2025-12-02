# -*- mode: python ; coding: utf-8 -*-

# -*- mode: python ; coding: utf-8 -*-

import os
import PySide6

pyside_dir = os.path.dirname(PySide6.__file__)

datas = [
    (os.path.join(pyside_dir, "plugins"), "PySide6/plugins"),
    (os.path.join(pyside_dir, "resources"), "PySide6/resources"),
]

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=["PySide6.QtGui", "PySide6.QtWidgets", "PySide6.QtCore"],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="DentalLabeler",
    windowed=True,
    icon=None
)


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
