#!/usr/bin/env python3
r"""deck 插件仓库预检补丁（UNC 工作区专用）

现象：工作区是 \\wsl.localhost\... 时，DSH 执行命令的 cwd 会退到 C:\Windows，
插件的 `gh repo view` 因此报 not a git repository，预检一直红、repo.refId 空。

做法：在 gh repo view 失败后，直接读工作区 .git/config 的 origin（Node fs，
不依赖子进程/cwd），把 owner/name 填进 refId。补丁用标记包裹，插件升级覆盖后重跑：
    python3 diag/patch-deck-preflight.py           # 打补丁
    python3 diag/patch-deck-preflight.py --revert  # 还原
重启 DeepSeek Harness 后生效。
"""
import glob
import os
import sys

ROOTS = glob.glob("/mnt/c/Users/*/.dsh/profiles/desktop/node_modules") + [
    os.path.expanduser("~/.dsh/profiles/desktop/node_modules"),
]
CANDIDATES = []
for base in ROOTS:
    CANDIDATES.extend(glob.glob(os.path.join(base, "dsh-mattpocock-skills-deck/tracker/rooms/github/preflight.js")))

ANCHOR = r'''      if (rr.ok) {
        const s = (rr.data.stdout || "").trim();
        if (s && s.includes("/")) refId = s;
      }
    }
'''

BLOCK = r'''    /* deck-unc-patch:start */
    if (!refId) {
      let fsmod = null;
      let pathmod = null;
      try { fsmod = await import("node:fs"); pathmod = await import("node:path"); } catch (e0) {
        try { fsmod = (typeof process !== "undefined" && process.getBuiltinModule) ? process.getBuiltinModule("node:fs") : null; } catch (e1) { fsmod = null; }
      }
      if (fsmod) {
        const pm = (pathmod && (pathmod.posix || pathmod.default || pathmod)) || null;
        const joinp = pm ? function (a, b) { return pm.join(a, b); } : function (a, b) { return String(a).replace(/\/+$/, "") + "/" + b; };
        const readCfg = function (p) { try { return fsmod.readFileSync(p, "utf8"); } catch (e) { return null; } };
        const fixCase = function (root) {
          try {
            const parent = pm ? pm.dirname(root) : String(root).replace(/\/[^/]+$/, "");
            const base = pm ? pm.basename(root) : String(root).split("/").pop();
            const ents = fsmod.readdirSync(parent);
            for (let i = 0; i < ents.length; i++) {
              if (String(ents[i]).toLowerCase() === String(base).toLowerCase()) return joinp(parent, ents[i]);
            }
          } catch (e) {}
          return root;
        };
        const roots = [];
        const raw = String(cwd || "");
        if (raw) roots.push(raw.replace(/[\\/]+$/, ""));
        const stripped = raw.replace(/^\\\\wsl\.localhost\\[^\\/]+/i, "").replace(/\\/g, "/");
        if (stripped && stripped !== raw) roots.push(stripped.replace(/\/+$/, ""));
        for (let i = 0; i < roots.length; i++) {
          let root = roots[i];
          let cfg = readCfg(root + "/.git/config");
          if (!cfg) {
            const fixed = fixCase(root);
            if (fixed !== root) { root = fixed; cfg = readCfg(root + "/.git/config"); }
          }
          if (cfg) {
            const m = cfg.match(/\[remote\s+"origin"\][^[]*url\s*=\s*([^\r\n]+)/);
            const um = m && String(m[1]).match(/github\.com[\/:]([^\/\s]+)\/([^\/\s]+?)(?:\.git)?\s*$/);
            if (um) { refId = um[1] + "/" + um[2]; break; }
          }
        }
        try {
          fsmod.appendFileSync("C:/Users/ASUS/AppData/Local/Temp/deck-preflight-debug.log", JSON.stringify({ cwd: raw, roots: roots, refId: String(refId || "") }) + "\n");
        } catch (e3) {}
      }
    }
    /* deck-unc-patch:end */
'''

START = "    /* deck-unc-patch:start */"
END = "    /* deck-unc-patch:end */\n"


def revert(path):
    src = open(path, encoding="utf-8").read()
    i = src.find(START)
    j = src.find(END)
    if i >= 0 and j > i:
        src = src[:i] + src[j + len(END):]
        open(path, "w", encoding="utf-8", newline="\n").write(src)
        return "reverted"
    return "nothing to revert"


def patch(path):
    src = open(path, encoding="utf-8").read()
    if START in src:
        return "already patched"
    if ANCHOR not in src:
        return "SKIP: anchor not found (插件结构变了，需要人工看)"
    src = src.replace(ANCHOR, ANCHOR + BLOCK, 1)
    open(path, "w", encoding="utf-8", newline="\n").write(src)
    return "patched"


if not CANDIDATES:
    sys.exit("找不到插件目录（试过：%s）" % ", ".join(ROOTS))
do_revert = "--revert" in sys.argv
for p in CANDIDATES:
    print("%s -> %s" % (p, revert(p) if do_revert else patch(p)))
if not do_revert:
    print("重启 DeepSeek Harness 后生效")
