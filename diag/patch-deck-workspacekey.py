#!/usr/bin/env python3
"""修 dsh-mattpocock-skills-deck 的工作区路径小写化 bug。

lib/workspaceKey.js 里 normalizeWorkspacePath() 在 win32 平台无条件
    n = n.toLowerCase()
对 UNC 路径（\\\\wsl.localhost\\...）来说，背后是大小写敏感的 Linux 文件系统，
小写化会把 ...\\github\\WanGP 拼成 ...\\github\\wangp，导致插件在错误目录里找 .git。

用法：python3 diag/patch-deck-workspacekey.py [--revert]
插件升级会覆盖，需要重跑；改完重启 DeepSeek Harness 生效。
"""
import sys, pathlib, shutil, datetime

TARGET = pathlib.Path("/mnt/c/Users/ASUS/.dsh/profiles/desktop/node_modules/dsh-mattpocock-skills-deck/lib/workspaceKey.js")
MARK = "// [wangp-patch:unc-case]"
NEEDLE = '      if (platform.os === "win32") n = n.toLowerCase();'
REPLACEMENT = (
    '      // [wangp-patch:unc-case] UNC 路径(\\\\\\\\wsl.localhost\\...)背后可能是大小写敏感的\n'
    '      // Linux 文件系统，不能按 Windows 惯例小写化，否则 .../WanGP 会被拼成 .../wangp。\n'
    '      const __uncRaw = /^\\\\\\\\/.test(String(raw).trim());\n'
    '      if (platform.os === "win32" && !__uncRaw) n = n.toLowerCase();'
)

def main():
    text = TARGET.read_text(encoding="utf-8")
    if "--revert" in sys.argv:
        orig = TARGET.with_suffix(".js.orig")
        if not orig.exists():
            print("没有 .orig 备份，无法 revert"); return 1
        shutil.copy2(orig, TARGET)
        print(f"已还原: {TARGET}"); return 0
    if MARK in text:
        print("已经是打过补丁的状态，未改动"); return 0
    if NEEDLE not in text:
        print("找不到目标行，插件版本可能变了，未改动"); return 1
    orig = TARGET.with_suffix(".js.orig")
    if not orig.exists():
        shutil.copy2(TARGET, orig)
    TARGET.write_text(text.replace(NEEDLE, REPLACEMENT, 1), encoding="utf-8")
    print(f"已打补丁: {TARGET}")
    print(f"备份: {orig}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
