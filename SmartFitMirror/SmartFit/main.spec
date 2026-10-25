# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

# Путь к корню проекта (где лежит main.spec)
BASE_DIR = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    ['main.py'],
    pathex=[BASE_DIR],
    binaries=[],
    datas=[
        (os.path.join(BASE_DIR, 'assets', 'images'), 'assets/images'),
        (os.path.join(BASE_DIR, 'assets', 'videos'), 'assets/videos'),
    ],
    hiddenimports=['src', 'src.app', 'src.config', 'src.exercises',
                   'src.splash_screen', 'src.stats_panel', 'src.utils',
                   'src.video_player'],
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
    [],
    exclude_binaries=True,
    name='SmartFitMirror',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SmartFitMirror',
)